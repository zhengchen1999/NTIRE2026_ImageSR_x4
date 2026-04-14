# [NTIRE 2026 Challenge on Image Super-Resolution (x4)] Team10_AIMLAB

## Installation

```sh
conda create -n SwinIR-Diff python=3.8
conda activate SwinIR-Diff
pip install -r requirements.txt
```

## Executable Commands

```sh
python test.py --scale 4 --model_path ../../model_zoo/team10_AIMLAB/team10_swinir_pretrained.pth --folder_lq $test_dataset_dir --diff_swinir_model_path ../../model_zoo/team10_AIMLAB/team10_swinir_diff.pth --output_dir $result_dir


python eval.py \
--output_folder "/home/NTIRE26/data2/NTIRE2026/SR/Team10/NTIRE2026_ImageSR_x4-main/result" \
--target_folder "/home/NTIRE26/data2/NTIRE2026/SR/dataset/DIV2K/Test/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 1 

```

## Submission Result download
```sh
https://drive.google.com/file/d/1-ytQ43ChiKnyAksEvFVG63qv8_Zszf-I/view?usp=drive_link
```

## Pre-trained checkpoints download
```sh
https://drive.google.com/file/d/1TqFmnbP18oge5BtKUzQhtvTy8YW0u77u/view?usp=drive_link
```
