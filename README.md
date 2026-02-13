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

支持 JSON 日志格式（便于接入监控）：

```bash
python train.py --config configs/text_llm_only_local.yaml --max-steps 2 --log-format json
```

将结构化日志落盘（json lines）：

```bash
python train.py --config configs/text_llm_only_local.yaml --max-steps 2 \
  --log-format json --log-file artifacts/train_metrics.jsonl
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

DDP DP 同步专项回归（2 进程 LLM-only，校验 `sync_impl=ddp`）：

```bash
bash scripts/run_e2e_ddp_dp_sync.sh
```

ZeRO-1 分布式优化器专项回归（2 进程 LLM-only，校验 `sync_impl` 包含 `zero1`）：

```bash
bash scripts/run_e2e_zero1_dp_optim.sh
```

1F1B 100+ steps 稳定性回归（默认 120 steps，校验 optimizer-step 记录数与数值有限性）：

```bash
bash scripts/run_e2e_1f1b_100_steps.sh
```

GPipe vs 1F1B 自动基线对比（导出 json + md 报告）：

```bash
bash scripts/run_benchmark_schedule_compare.sh
```

SP 基线对比（LLM 阶段，`sequence_parallel on/off` 自动对比并导出报告）：

```bash
bash scripts/run_benchmark_sp_compare.sh
```

可选环境变量：

- `STEPS=5`：每个 case 的训练步数
- `FORCE_CPU=1`：默认开启，强制 CPU（避免 Gloo + CUDA 混用）
- `TIMEOUT_SEC=240`：单 case 超时时间（秒）
- `LOG_DIR=/path/to/logs`：日志输出目录
- `REQUIRE_1F1B_NOT_WORSE=1`：benchmark 门禁，要求 1F1B 不劣于 GPipe（可关闭为 0）
- `TOKENS_RATIO_MIN=0.98`：`1f1b/gpipe` 最低 tokens/s 比例
- `STEP_TIME_RATIO_MAX=1.02`：`1f1b/gpipe` 最高 step_time 比例

## 训练配置补充项

- `training.grad_clip_norm`：梯度裁剪阈值（`0` 表示关闭）
- `training.optimizer.stage_lrs`：按阶段覆盖学习率，例如：

```yaml
training:
  optimizer:
    type: adamw
    lr: 2.0e-4
    stage_lrs:
      encoder: 1.5e-4
      llm: 2.0e-4
      decoder: 1.0e-4
```

Checkpoint 恢复时默认会恢复 RNG 状态（保证可复现），如需关闭可加：

```bash
python train.py --config configs/text_llm_only_local.yaml --resume ckpt.pt --no-restore-rng
```

默认保存/恢复的 RNG 状态包括：
- Python `random`
- NumPy `np.random`（若环境安装了 numpy）
- PyTorch CPU RNG
- PyTorch CUDA RNG（`torch.cuda.get_rng_state_all`）

如需进一步降低算子层面的非确定性，可启用 deterministic 模式：

```bash
python train.py --config configs/text_llm_only_local.yaml --deterministic
```

- `distributed.grad_sync_bucket_mb`：梯度 all-reduce bucket 大小（MB，`0` 表示按参数逐个同步）
- `training.deterministic`：是否启用确定性模式（默认 `false`）
  - 开启后会设置 `torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、
    `cudnn.benchmark=False`、关闭 TF32，并设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `training.optimizer.zero_stage`：优化器分片等级，当前支持：
  - `0`：常规 AdamW（默认）
  - `1`：ZeRO-1 Distributed Optimizer（连续参数/主梯度 buffer + `reduce_scatter/all_gather`）
- `training.loss_weights`：多任务损失权重，支持键：`text/image/audio`，示例：

```yaml
training:
  loss_weights:
    text: 1.0
    image: 0.3
    audio: 0.5
```

> 训练时会按启用输出分支做加权归一化：`sum(w_i * loss_i) / sum(w_i)`（仅统计 `w_i > 0` 的分支）。

- `stages.<stage>.activation_checkpoint`：按阶段启用 activation checkpoint（true/false）
- `stages.<stage>.sequence_parallel`：在 TP 基础上启用序列并行（要求 `tp_size > 1`）
- `training.io`：I/O 占位优化开关（prefetch/pin_memory）：

```yaml
training:
  io:
    enable_prefetch: true
    prefetch_size: 2
    pin_memory: true
    num_workers: 0
```

ZeRO-1 最小配置示例（建议配合 `dp_size > 1`）：

```yaml
training:
  optimizer:
    type: adamw
    zero_stage: 1
    lr: 1.0e-3
    weight_decay: 0.01
```

ZeRO-1 Distributed Optimizer（本实现）关键步骤：

1. backward 完成后，将模型梯度拷贝到 **fp32 main gradient contiguous buffer**。
2. 在 DP 组上执行 `reduce_scatter`，每个 rank 仅保留本地 shard 的已规约梯度。
3. 在本地 shard 上用 **fp32 main parameter shard** 执行 AdamW 更新。
4. 将本地更新后的 fp32 参数 shard cast 回模型参数 dtype（bf16/fp16/fp32）并写入参数 buffer。
5. 在 DP 组执行 `all_gather`，恢复完整参数 buffer；模型参数视图直接指向该 buffer，可立即进入下一轮前向。

实现细节：参数 buffer 按 dtype 分桶管理（bf16/fp16/fp32 各自独立 contiguous buffer），避免混合 dtype 参数写入同一 buffer。

注意：当 `zero_stage=1` 时，DP 通信由优化器内部处理，训练会自动跳过 DDP 的 DP all-reduce 路径。

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