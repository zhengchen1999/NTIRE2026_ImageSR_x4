# Inference Only

这个目录是从 SeeSR 项目中提取出来的推理子集，只保留推理入口和依赖模块。

保留内容:
- `infer.sh`: 推理启动脚本
- `test_seesr.py`: 主推理脚本
- `models/`, `pipelines/`, `ram/`: 推理依赖模块
- `utils/`: 仅保留推理所需的工具文件
- `requirements.txt`: 原项目依赖清单

说明:
- 该目录只复制代码，不复制模型权重与数据集。
- `infer.sh` 已改为基于脚本位置解析路径，默认仍引用上一级仓库中的 `preset/` 和 `experience/`。
- 未包含训练、数据集构建、评测相关脚本。
