# Image_MultiResolutionSampler.py
# 原始opensora逻辑:
# world_batch_index_list [[batch1 * num_gpu], [batch2 * num_gpu], ....]
# local_batch_index_list 根据world_batch_index_list拆开: [[batch1],[batch1],.. 共num_gpu个, [batch2], [batch2], [batch2], 共num_gpu个]
# 然后用于accelerate.prepare, 不同的rank拿到不同的数据 
# ** 如果没有accelerate.prepare, 所有rank的local_batch_index_list都是一样的 !! **

# 现在需要DDP实现需要改动:
# 在__iter__中使用get_rank()取当前rank的数据, ** 而不是使用 拆开 + prepare **
# 参考FastGen/fastgen/datasets/samplers.py 使用 start_idx 来计算当前epoch 和 idx
# 在当前框架需要使用infinite_iter

import pandas as pd
import torch
from torch.utils.data import Sampler
from collections import OrderedDict, defaultdict
from typing import Iterator, List
from pprint import pformat
from fastgen.utils.distributed import get_rank, world_size
# Import ImageBucket

from fastgen.datasets.custom.bucket_image import ImageBucket 


def format_numel_str(num):
    if num >= 1000000:
        return f"{num/1000000:.2f}M"
    elif num >= 1000:
        return f"{num/1000:.2f}K"
    return str(num)


class ImageMultiResolutionSampler(Sampler):
    """
    "idx-H-W-condition_num"
    """

    def __init__(
        self,
        dataset_path: str,
        bucket: ImageBucket,
        seed: int = 42,
        drop_last: bool = True,
        shuffle: bool = True,
        verbose: bool = True,
        start_idx: int = 0,
    ):
        self.dataset_path = dataset_path
        self.bucket = bucket
        self.world_size = world_size()
        self.rank = get_rank()
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.verbose = verbose and (self.rank == 0)
        self.start_idx = start_idx

        if self.verbose:
            print(f"📖 Loading CSV from {dataset_path}...")
        self.df = pd.read_csv(dataset_path)

        required_cols = ['id', 'height', 'width', 'condition_num']
        for col in required_cols:
            assert col in self.df.columns, f"Missing required column: {col}"

        if self.verbose:
            print(f"✅ CSV loaded: {len(self.df)} samples")

        if self.verbose:
            print("🔧 Assigning buckets to samples...")

        self.df['bucket_id'] = self.df.apply(self._assign_bucket, axis=1)

        valid_df = self.df[self.df['bucket_id'].notna()].copy()
        dropped = len(self.df) - len(valid_df)

        if dropped > 0 and self.verbose:
            print(f"⚠️ Dropped {dropped} samples (cannot assign to bucket)")

        self.df = valid_df.reset_index(drop=True)

        if self.verbose:
            print(f"✅ Total valid samples: {len(self.df)}")
            
        self.num_global_batches = len(self.get_sample_index_list(0))
        

        self.real_hw_info = None
        self.epoch = self.start_idx // self.num_global_batches

        self.world_batch_index_list = self.get_sample_index_list(self.epoch)
       
    def _assign_bucket(self, row):
        try:
            bucket_id = self.bucket.get_bucket_id(
                height=int(row['height']),
                width=int(row['width']),
                has_demo=False, 
                condition_num=int(row['condition_num']),
            )
            return bucket_id
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error assigning bucket for id={row['id']}: {e}")
            return None

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()
        real_hw_info = {}

        for idx, row in self.df.iterrows():
            bucket_id = row['bucket_id']
            sample_id = int(row['id'])

            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []

            bucket_sample_dict[bucket_id].append(sample_id)

            real_h, real_w = self.bucket.get_hw(bucket_id)
            condition_num = bucket_id[2]   # bucket_id = (res_key, False, cond_num, ar_key)

            real_hw_info[sample_id] = (real_h, real_w, False, condition_num)

        self.real_hw_info = real_hw_info
        return bucket_sample_dict

    def get_sample_index_list(self, epoch):
        bucket_sample_dict = self.group_by_bucket()

        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + epoch)

        world_batch_index_list = []

        for bucket_id, data_list in bucket_sample_dict.items():
            bs_per_gpu = int(self.bucket.get_batch_size(bucket_id))
            bs_world = self.world_size * bs_per_gpu

            if len(data_list) < bs_world:
                if self.verbose:
                    print(f"⚠️ Bucket {bucket_id} has only {len(data_list)} samples (< {bs_world}), skipped")
                continue

            remainder = len(data_list) % bs_world
            if remainder > 0:
                if self.drop_last:
                    data_list = data_list[:-remainder]
                else:
                    data_list = data_list + data_list[:bs_world - remainder]

            if self.shuffle:
                indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in indices]

            assert len(data_list) % bs_world == 0

            for start_idx in range(0, len(data_list), bs_world):
                world_batch_index_list.append(data_list[start_idx:start_idx + bs_world])

        if self.shuffle:
            batch_indices = torch.randperm(len(world_batch_index_list), generator=g).tolist()
            world_batch_index_list = [world_batch_index_list[i] for i in batch_indices]

        return world_batch_index_list

    # infinite + rank_split  version
    def __iter__(self) -> Iterator[List[str]]:
        step = self.start_idx
        # shuffle by epoch and seed

        while True:
            if step % self.num_global_batches == 0:
                self.epoch += 1
                self.world_batch_index_list = self.get_sample_index_list(self.epoch)

            global_batch = self.world_batch_index_list[step % self.num_global_batches]
            batch_size = len(global_batch) // self.world_size
            local_batch = global_batch[self.rank * batch_size:(self.rank + 1) * batch_size]
            real_h, real_w, _, condition_num = self.real_hw_info[local_batch[0]]
            batch_str = [f"{idx}-{real_h}-{real_w}-{condition_num}" for idx in local_batch]

            yield batch_str
            step += 1

    def __len__(self) -> int:
        # 这里的len会被用到后续函数吗 ? 乘以world_size似乎没有意义..无法直观看出一个epoch需要多少步, num_global_batches代表一个epoch需要多少步
        return self.num_global_batches

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        total_samples = 0
        total_batch = 0

        image_buckets = defaultdict(lambda: [0, 0])

        for bucket_id, samples in bucket_sample_dict.items():
            res_key, _, cond_num, ar_key = bucket_id
            num_samples = len(samples)
            bs = self.bucket.get_batch_size(bucket_id)
            num_batches = int(num_samples // (bs * self.world_size))

            total_samples += num_samples
            total_batch += num_batches

            key = f"{res_key}_{ar_key}_cond{cond_num}"
            image_buckets[key][0] += num_samples
            image_buckets[key][1] += num_batches

        image_buckets = dict(sorted(image_buckets.items()))

        print("=" * 80)
        print("📊 Bucket Information")
        print("=" * 80)
        print(f"Total samples: {format_numel_str(total_samples)}")
        print(f"Total batches (per GPU): {format_numel_str(total_batch)}")
        print(f"World size: {self.world_size}")
        print(f"Non-empty buckets: {len(bucket_sample_dict)}")
        print("-" * 80)

        if image_buckets:
            print("🖼️ Image Buckets [#samples, #batches]:")
            for key, (n_samples, n_batches) in image_buckets.items():
                print(f"  {key}: [{n_samples}, {n_batches}]")
            print("-" * 80)

        print("=" * 80)

# ===== 测试代码 =====
if __name__ == "__main__":
    import os
    from pathlib import Path
    import torch.distributed as dist
    import torch


    dist.init_process_group(backend='nccl')   # 如果没有 GPU 可以改成 'gloo'

    rank = dist.get_rank()


    repo_root = Path(__file__).resolve().parents[3]
    dataset_path = os.getenv("FASTGEN_SAMPLE_DATASET", str(repo_root / "data" / "raim_phase1_cond1.csv"))

    if not os.path.exists(dataset_path):
        if rank == 0:
            print(f"❌ CSV 文件不存在: {dataset_path}")
        dist.destroy_process_group()
        exit(1)

    if rank == 0:
        print(f"\n使用真实数据集: {dataset_path}")

    # 创建 ImageBucket（所有 rank 都要创建相同的 bucket）
    if rank == 0:
        print("\n🪣 Creating ImageBucket...")
    bucket = ImageBucket(
        resolutions=["512", "768", "1024"],
        batch_size_config={"1024": 8},
        max_condition_num=3,
        downsample_prob=0.0,
        exact_resolutions=[
            "1024x768",
            "1024x1024",
            "768x1024",
        ],
        align_to_multiple_of=16,
        fuzzy_match_pixel_diff=8,
    )

    # 创建 Sampler（传入自动获取的 world_size 和 rank）
    if rank == 0:
        print("\n🎲 Creating Sampler...")

    sampler = ImageMultiResolutionSampler(
        dataset_path=dataset_path,
        bucket=bucket,
        seed=42,
        drop_last=True,
        shuffle=True,
        verbose=(rank == 0),            # 只让 rank 0 打印详细日志
        start_idx = 327
    )
    # if rank == 0:
    #     import pdb; pdb.set_trace()
    if rank == 0:
        print(len(sampler),sampler.num_global_batches)

    for i, batch in enumerate(sampler):
        if i >= 5:
            break
        if rank == 0:
            print("rank0")
            print(f"\nBatch {i+1}:")
            print(f"  Size: {len(batch)}")
            print(f"  Format example: {batch[0]}")

            parts = batch[0].split('-')
            idx, h, w, cond_num = parts
            print(f"  → idx={idx}, H={h}, W={w}, condition_num={cond_num}")

            for item in batch:
                p = item.split('-')
                assert p[1] == h and p[2] == w and p[3] == cond_num, \
                    f"Inconsistent batch item: {item}"
            print("  ✅ All samples in batch have same H/W/condition_num")

        if rank == 1:
            print("rank1")
            print(f"\nBatch {i+1}:")
            print(f"  Size: {len(batch)}")
            print(f"  Format example: {batch[0]}")

            parts = batch[0].split('-')
            idx, h, w, cond_num = parts
            print(f"  → idx={idx}, H={h}, W={w}, condition_num={cond_num}")

            for item in batch:
                p = item.split('-')
                assert p[1] == h and p[2] == w and p[3] == cond_num, \
                    f"Inconsistent batch item: {item}"
            print("  ✅ All samples in batch have same H/W/condition_num")

    # 所有进程同步等待 rank 0 完成打印（可选，防止 rank 0 先退出）
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 80)
        print("测试结束")
        print("=" * 80)

    # 清理分布式进程组
    dist.destroy_process_group()
