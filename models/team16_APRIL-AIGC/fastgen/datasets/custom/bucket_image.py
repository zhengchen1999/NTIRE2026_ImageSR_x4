# image_bucket.py
from collections import OrderedDict
import random


from fastgen.datasets.custom.aspect import (
    ASPECT_RATIOS,
    get_closest_ratio,
    get_num_pixels,
)


class ImageBucket:
    def __init__(
        self,
        resolutions=None,
        batch_size_config=None,
        max_condition_num=3,
        downsample_prob=0.0,
        exact_resolutions=None,       # e.g. ["1024x768", "1024x1024", "768x1024"]
        align_to_multiple_of=16,
        fuzzy_match_pixel_diff=8,
    ):
        if resolutions is None:
            resolutions = ["512", "768", "1024"]

        if batch_size_config is None:
            batch_size_config = {"1024": 1,"768": 2, "512": 4,}

        self.resolutions = resolutions
        self.batch_size_config = batch_size_config
        self.max_condition_num = max_condition_num
        self.downsample_prob = downsample_prob
        self.align_to_multiple_of = align_to_multiple_of
        self.fuzzy_match_pixel_diff = fuzzy_match_pixel_diff

        # 处理 exact_resolutions, 不会去不同bucket里面resize了
        self.exact_resolutions = set()
        if exact_resolutions:
            for res in exact_resolutions:
                if isinstance(res, str):
                    parts = res.replace(" ", "").lower().split("x")
                    if len(parts) != 2:
                        continue
                    try:
                        h, w = int(parts[0]), int(parts[1])
                    except ValueError:
                        continue
                else:
                    try:
                        h, w = int(res[0]), int(res[1])
                    except:
                        continue

                h = round(h / align_to_multiple_of) * align_to_multiple_of
                w = round(w / align_to_multiple_of) * align_to_multiple_of
                if h > 0 and w > 0:
                    self.exact_resolutions.add((h, w))

        for res in self.resolutions:
            if res not in ASPECT_RATIOS:
                raise ValueError(f"Resolution key '{res}' not found in ASPECT_RATIOS")

        self.bucket_id_to_info = OrderedDict()
        self._build_buckets()

        print(f"ImageBucket initialized with {len(self.bucket_id_to_info)} buckets")
        print(f"  Resolutions (fallback): {self.resolutions}")
        print(f"  Exact preserved: {sorted(self.exact_resolutions)}")
        print(f"  Max condition num: {max_condition_num}")
        print(f"  Downsample prob: {downsample_prob:.2f}")
        print(f"  Align to ×{align_to_multiple_of}")

    def _estimate_base_batch_size(self, h: int, w: int) -> int:
        pixels = h * w
        ref_pixels = 1024 * 1024
        ref_bs = self.batch_size_config.get("1024", 1)

        if pixels <= ref_pixels * 0.4:
            base = ref_bs * 4
        elif pixels <= ref_pixels * 0.65:
            base = ref_bs * 2
        elif pixels <= ref_pixels * 1.1:
            base = ref_bs
        elif pixels <= ref_pixels * 1.6:
            base = max(1, ref_bs // 2)
        else:
            base = max(1, ref_bs // 4)

        return max(1, base)

    def _build_buckets(self):

        cnt = 0

        for res_key in self.resolutions:
            _, ar_dict = ASPECT_RATIOS[res_key]
            for cond in range(0, self.max_condition_num + 1):
                for ar_key, (h, w) in ar_dict.items():
                    bucket_id = (res_key, False, cond, ar_key)
                    base_bs = self.batch_size_config[res_key]
                    bs = max(1, base_bs // (2 ** max(0, cond - 1)))
                    self.bucket_id_to_info[bucket_id] = {
                        'id': cnt,
                        'height': h,
                        'width': w,
                        'resolution_key': res_key,
                        'has_demo': False,
                        'condition_num': cond,
                        'aspect_ratio': ar_key,
                        'batch_size': bs,
                    }
                    cnt += 1

        for h, w in sorted(self.exact_resolutions):
            res_key = f"{h}x{w}"
            ar_key = f"{h}/{w}"

            base_bs = self.batch_size_config.get(res_key, self._estimate_base_batch_size(h, w))

            for cond in range(0, self.max_condition_num + 1):
                bucket_id = (res_key, False, cond, ar_key)
                bs = max(1, base_bs // (2 ** max(0, cond - 1)))
                self.bucket_id_to_info[bucket_id] = {
                    'id': cnt,
                    'height': h,
                    'width': w,
                    'resolution_key': res_key,
                    'has_demo': False,
                    'condition_num': cond,
                    'aspect_ratio': ar_key,
                    'batch_size': bs,
                }
                cnt += 1

        self.num_bucket = cnt

    def get_bucket_id(
        self,
        height: int,
        width: int,
        has_demo: bool = False,
        condition_num: int = 0,
        approx: float = 0.7,
    ):
        if has_demo:
            raise NotImplementedError("Demo mode is not supported in current version")

        if condition_num < 0 or condition_num > self.max_condition_num:
            return None

        target_h = height
        target_w = width
        for eh, ew in self.exact_resolutions:
            if (abs(height - eh) <= self.fuzzy_match_pixel_diff and
                abs(width - ew) <= self.fuzzy_match_pixel_diff):
                target_h, target_w = eh, ew
                break

        if (target_h, target_w) not in self.exact_resolutions:
            target_h = round(target_h / self.align_to_multiple_of) * self.align_to_multiple_of
            target_w = round(target_w / self.align_to_multiple_of) * self.align_to_multiple_of

        res_key = f"{target_h}x{target_w}"
        ar_key = f"{target_h}/{target_w}"
        bucket_id = (res_key, False, condition_num, ar_key)

        if bucket_id in self.bucket_id_to_info:
            return bucket_id

        sorted_res = sorted(
            self.resolutions,
            key=lambda x: get_num_pixels(x),
            reverse=True
        )

        res_key = None
        target_pixels = target_h * target_w
        for cand in sorted_res:
            if get_num_pixels(cand) * approx <= target_pixels:
                res_key = cand
                break
        if res_key is None:
            res_key = sorted_res[-1]

        if self.downsample_prob > 0 and random.random() < self.downsample_prob:
            idx = sorted_res.index(res_key)
            if idx < len(sorted_res) - 1:
                res_key = sorted_res[idx + 1]

        _, ar_dict = ASPECT_RATIOS[res_key]
        ar_key = get_closest_ratio(target_h, target_w, ar_dict)
        if ar_key is None:
            ar_key = next(iter(ar_dict))

        bucket_id = (res_key, False, condition_num, ar_key)
        if bucket_id in self.bucket_id_to_info:
            return bucket_id

        return None

    def get_batch_size(self, bucket_id):
        if bucket_id not in self.bucket_id_to_info:
            raise ValueError(f"Bucket not found: {bucket_id}")
        return self.bucket_id_to_info[bucket_id]['batch_size']

    def get_hw(self, bucket_id):
        if bucket_id not in self.bucket_id_to_info:
            raise ValueError(f"Bucket not found: {bucket_id}")
        info = self.bucket_id_to_info[bucket_id]
        return info['height'], info['width']

    def get_bucket_info(self, bucket_id):
        if bucket_id not in self.bucket_id_to_info:
            raise ValueError(f"Bucket not found: {bucket_id}")
        return self.bucket_id_to_info[bucket_id]

    def get_all_bucket_ids(self):
        return list(self.bucket_id_to_info.keys())

    def get_bucket_count(self):
        return len(self.bucket_id_to_info)


if __name__ == "__main__":
    bucket = ImageBucket(
        resolutions=["512", "768", "1024"],
        batch_size_config={"1024": 1},
        max_condition_num=4,
        downsample_prob=0.0,
        exact_resolutions=[
            "1024x768",
            "1024x1024",
            "768x1024",
        ],
        align_to_multiple_of=32,
        fuzzy_match_pixel_diff=8,
    )

    test_cases = [
        (1024, 768),
        (1024, 767),
        (768, 1024),
        (1023, 1025),
        (512, 512),
        (1200, 800),
    ]

    print("\nTest bucket assignment:")
    for h, w in test_cases:
        bid = bucket.get_bucket_id(h, w, condition_num=1)
        if bid:
            bh, bw = bucket.get_hw(bid)
            bs = bucket.get_batch_size(bid)
            print(f"{h:4d} × {w:4d}  →  {bh:4d} × {bw:4d}    bs={bs:2d}    {bid}")
        else:
            print(f"{h:4d} × {w:4d}  →  None")