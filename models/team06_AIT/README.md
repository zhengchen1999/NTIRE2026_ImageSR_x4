# team06_AIT

This folder provides the NTIRE interface wrapper for AIT (SeeSR-based inference backend).

Required interface:
- `main(model_dir, input_path, output_path, device)`

How this wrapper resolves paths:
1. Environment variable `SEESR_INFER_ROOT`
2. `model_zoo/team06_ait/team06_ait.json` -> `inference_root`
3. `model_zoo/team06_ait/inference_only`
4. fallback `/home/jiyang/jiyang/Projects/inference_only`

For challenge submission, prefer packaging all dependencies under:
- `model_zoo/team06_ait/inference_only`

Then set in `team06_ait.json`:
```json
{
  "inference_root": "model_zoo/team06_ait/inference_only"
}
```
