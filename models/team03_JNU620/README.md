# ProCata (Team03_ProCata)
# TeamName: JNU620

Overview
--------
ProCata is Team03's submission for the NTIRE 2026 Image Super-Resolution (×4) challenge. The method builds on CATANet as the baseline and introduces architectural and training adjustments to improve perceptual quality and high-frequency detail recovery.

Repository layout
-----------------
- `basicsr/` — BasicSR-based training/testing utilities and the ProCata implementation.
- `options/` — YAML configs; main test config: `options/test/test_ProCata_x4.yml`.
- `datasets/` — dataset placeholders (data not included in repo).
- `model_zoo/` — put pretrained checkpoint at `model_zoo/team03_ProCata/ProCata.pth`.

Pretrained weights
------------------
Place the trained checkpoint here:

```
model_zoo/team03_ProCata/ProCata.pth
```

you can get the checkpoint with a download URL in `team03_ProCata.txt`

Environment
-----------
Recommended setup (same as repository root):

```bash
conda create -n ProCata python=3.9 -y
conda activate ProCata
pip install -r requirements.txt
pip install -r models/team03_ProCata/requirements.txt
```

Quick test (repository root)
----------------------------
The repository provides a unified test entry `test.py` in the root. Team03 is registered as `--model_id 3`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --test_dir /path/to/test_dir \
  --save_dir /path/to/save_results \
  --model_id 3
```

Notes:
- `--test_dir` can point to a folder that contains `LQ/` (low-quality inputs) and optionally `HR/` (ground-truth) subfolders.
- Outputs are saved to `--save_dir/<model_name>/test`.

BasicSR-style test
-------------------
You can also run the BasicSR test script inside the model folder using the provided YAML:

```bash
python models/team03_ProCata/basicsr/test.py -opt models/team03_ProCata/options/test/test_ProCata_x4.yml
```

Evaluation
----------
Use the repository-level `eval.py` to compute IQA metrics on generated outputs:

```bash
python eval.py --output_folder /path/to/your/output_dir --target_folder /path/to/HR --metrics_save_path ./IQA_results --gpu_ids 0
```

Configuration
-------------
- Edit `models/team03_ProCata/options/test/test_ProCata_x4.yml` to change dataset paths, `pretrain_network_g`, or evaluation metrics.

Acknowledgements & citation
---------------------------
This implementation uses CATANet as the baseline and relies on BasicSR utilities. Please cite the original CATANet paper when using this code.

