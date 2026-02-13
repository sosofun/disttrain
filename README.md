# disttrain

基于 PyTorch 的分布式大模型训练框架示例，支持：

- 逻辑三阶段 Pipeline（Encoder / LLM / Decoder），且 Encoder / Decoder 可按配置启停
- 每阶段独立 TP/DP 配置与独立 Process Group
- 多模态输入（image/video/audio）与多模态输出（image/audio）插件化扩展
- GPipe / 1F1B 两种调度策略

## 目录结构

```text
disttrain/
  config.py
  checkpoint.py
  dist/
  models/
  pipeline/
configs/
  multimodal_tri_stage.yaml
  text_llm_only.yaml
  text_llm_only_local.yaml
train.py
```

## 依赖

- Python 3.10+
- PyTorch 2.x
- PyYAML（读取 YAML 配置）

## 快速开始（单机 smoke）

```bash
python train.py --config configs/text_llm_only_local.yaml --max-steps 2
```

或使用脚本（fake 数据）：

```bash
bash scripts/run_fake_data.sh local-llm
```

三阶段 CPU 多进程 fake 数据：

```bash
bash scripts/run_fake_data.sh tri-stage-cpu
```

## E2E 多进程 smoke / 回归脚本（torchrun + fake 数据）

2-stage smoke（`encoder+llm` + `llm+decoder`）：

```bash
bash scripts/run_e2e_2stage_smoke.sh
```

3-stage smoke（`encoder+llm+decoder`）：

```bash
bash scripts/run_e2e_3stage_smoke.sh
```

一键 4 拓扑回归（更严格）：

```bash
bash scripts/run_e2e_all_topologies.sh
```

可选环境变量：

- `STEPS=5`：每个 case 的训练步数
- `FORCE_CPU=1`：默认开启，强制 CPU（避免 Gloo + CUDA 混用）
- `TIMEOUT_SEC=240`：单 case 超时时间（秒）
- `LOG_DIR=/path/to/logs`：日志输出目录

## 分布式启动示例

纯文本 LLM（8 卡）：

```bash
torchrun --nproc_per_node=8 train.py --config configs/text_llm_only.yaml
```

多模态三阶段（16 卡）：

```bash
torchrun --nproc_per_node=16 train.py --config configs/multimodal_tri_stage.yaml
```

## 测试

```bash
python -m unittest discover -s tests -v
```