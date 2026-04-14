import os
import glob

# ===================== 请修改为你的数据集实际路径 =====================
GT_FOLDER = "datasets/DIV2K/DIV2K_train_HR"    # HR/GT图片路径
LQ_FOLDER = "datasets/DIV2K/DIV2K_train_LR_bicubic/X4"  # X4下采样LQ图片路径
# =====================================================================

def get_clean_filename(filename):
    """清理文件名：去掉后缀、x4标识，只保留数字序号（如 0001x4.png → 0001）"""
    # 去掉文件后缀
    name_without_ext = os.path.splitext(filename)[0]
    # 去掉x4标识（DIV2K LQ文件的命名格式）
    name_clean = name_without_ext.replace('x4', '').strip()
    return name_clean

# 1. 读取GT文件夹的所有图片序号
gt_files = glob.glob(os.path.join(GT_FOLDER, "*.[pj][np]g"))  # 匹配png/jpg
gt_ids = set()
for file in gt_files:
    filename = os.path.basename(file)
    gt_id = get_clean_filename(filename)
    gt_ids.add(gt_id)

# 2. 读取LQ文件夹的所有图片序号，并找出多余的文件
lq_files = glob.glob(os.path.join(LQ_FOLDER, "*.[pj][np]g"))
lq_ids = set()
extra_lq_files = []  # 存储多余的LQ文件路径
missing_gt_ids = []  # 存储LQ有但GT缺失的序号

for file in lq_files:
    filename = os.path.basename(file)
    lq_id = get_clean_filename(filename)
    lq_ids.add(lq_id)
    
    # 检查该序号是否在GT中存在
    if lq_id not in gt_ids:
        extra_lq_files.append(file)
        missing_gt_ids.append(lq_id)

# 3. 打印结果（直观展示）
print("="*60)
print(f"GT文件夹图片数量：{len(gt_ids)}")
print(f"LQ文件夹图片数量：{len(lq_ids)}")
print(f"\n❌ 发现 {len(missing_gt_ids)} 个GT缺失的序号：")
for idx in sorted(missing_gt_ids):
    print(f"  - {idx}")

print(f"\n🗑️  对应的多余LQ文件（共{len(extra_lq_files)}个）：")
for file in sorted(extra_lq_files):
    print(f"  - {file}")
print("="*60)

# 可选：自动删除多余的LQ文件（确认无误后取消下面两行的注释）
# for file in extra_lq_files:
#     os.remove(file)
#     print(f"已删除：{file}")