# Team 30: FusionHero

FusionHero is a two-branch transformer fusion method for 4× image super-resolution. It combines two SR branches with a fixed global weight (0.04 for the second branch) to produce the final output.

**Weights:** `branch_a.pth`, `branch_b.pth` (in `model_zoo/team30_FusionHero/`).  
Because each file exceeds 100 MB, download links are provided in `model_zoo/team30_FusionHero/team30_FusionHero.txt` if the checkpoint files are not included in the repository.

---

## Dependencies (how to run FusionHero inference)

**Official way (recommended for first-time setup):**  
The repository root contains a single `requirements.txt` for the whole challenge. The official README asks you to use:

- **Python 3.8**, then `pip install -r requirements.txt`

That file includes everything needed to run **FusionHero inference**: `torch`, `torchvision`, `einops`, `numpy`, `opencv-python`, and the repo’s `utils` (which also use `matplotlib`). So **using the official `requirements.txt` is enough** to run test-set inference for Team 30.

**What FusionHero actually uses (for reference):**  
Only the following are required to run `test.py --model_id 30`:

- `torch`, `torchvision`
- `einops`
- `numpy`, `opencv-python`
- `matplotlib` (used by `utils.utils_image`)

So if you already have a working environment (e.g. a conda env like `team18_xiaomi` with Python 3.10 and PyTorch 2.x), you can use it as long as these are installed. You do **not** need to match the official Python 3.8 or the exact versions in `requirements.txt` for FusionHero; the official file is meant for the whole repo (baseline DAT, eval scripts, etc.) and may include extra packages (e.g. `pytorch-lightning`, `menpo`, `mxnet`) that FusionHero does not use.

**If the official `requirements.txt` fails** (e.g. on `menpo`, `mxnet`, or `bcolz-zipline`): install PyTorch with CUDA from [pytorch.org](https://pytorch.org), then run `pip install -r requirements_team30_FusionHero.txt` from the repo root. That file lists only the packages needed for FusionHero.

**Summary:** Use `pip install -r requirements.txt` from the repo root for a one-shot setup; if that fails, use `requirements_team30_FusionHero.txt`. Or reuse an existing PyTorch env that has `torch`, `torchvision`, `einops`, `numpy`, `opencv-python`, and `matplotlib`.

---

## Where do test images come from?

**Test/validation images are not stored in this repository.** The organizers (or you) provide them separately.

- **At evaluation time:** Organizers will place the challenge test set (LR images) in a folder on their machine and pass that folder to `test.py` via `--test_dir`.
- **On your side:** You run the same script with `--test_dir` pointing to any folder that contains LR images in PNG (or JPG) format. There is no default path; you must set `--test_dir` and `--save_dir` yourself.

So: **input images are read from the path given by `--test_dir`**. The repo only contains code and (optionally) checkpoint weights or links to them.

---

## How to run inference (Team 30 FusionHero)

1. **Environment** (from repository root). Either use the official setup:
   ```bash
   conda create -n NTIRE-SR python=3.8
   conda activate NTIRE-SR
   pip install -r requirements.txt
   ```
   Or use an existing conda env with PyTorch and the dependencies listed in the "Dependencies" section above.

2. **Download weights** (if not in the repo):  
   Get `branch_a.pth` and `branch_b.pth` from the links in `model_zoo/team30_FusionHero/team30_FusionHero.txt` and place them inside `model_zoo/team30_FusionHero/`.

3. **Run restoration** on a folder of LR images:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python test.py \
     --test_dir /path/to/test/LQ \
     --save_dir /path/to/output \
     --model_id 30
   ```
   - `--test_dir`: folder containing input LR images (e.g. `0901x4.png`, `0902x4.png`, ...).
   - `--save_dir`: base directory for results. SR images will be written to `{save_dir}/team30_FusionHero/test/` with the same filenames as the inputs.

   **Create submission zip** (e.g. for challenge upload): from the repo root,  
   `cd results/team30_FusionHero/test && zip -r ../res_fusion_global_w0.04.zip *.png && cd ../../..`

   You can also run on a validation set:
   ```bash
   python test.py --valid_dir /path/to/valid/LQ --save_dir /path/to/output --model_id 30
   ```

4. **Optional – compute PSNR/SSIM** (if you have ground-truth HR images):
   ```bash
   python eval.py \
     --output_folder "/path/to/output/team30_FusionHero/test" \
     --target_folder "/path/to/test/HR" \
     --metrics_save_path "./IQA_results" \
     --gpu_ids 0
   ```
   This compares the restored images in `output_folder` to the HR images in `target_folder` and saves metrics (e.g. PSNR) to `metrics_save_path`.

---

## Summary for organizers

- **Inference:** Run `test.py` with `--test_dir` = folder of test LR images, `--save_dir` = your output root, `--model_id 30`. SR images are saved under `{save_dir}/team30_FusionHero/test/`. No test images or paths are hardcoded in the repo.
- **Evaluation:** Run `eval.py` on the generated SR folder and the test HR folder to obtain PSNR and other metrics as per the main repository instructions.
