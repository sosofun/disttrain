# 基于 PyTorch 的三阶段分布式大模型训练框架设计需求说明书

## 1. 文档信息

- **文档版本**：v1.0
- **状态**：需求设计稿
- **面向对象**：训练框架研发工程师、分布式系统工程师、算法工程师
- **技术基线**：PyTorch 2.x（`torch.distributed` / `torchrun` / NCCL）

---

## 2. 背景与目标

### 2.1 背景
在多阶段模型中（Encoder -> LLM -> Decoder），各阶段计算特征差异明显：
- Encoder 偏表示学习、序列压缩；
- LLM 偏大参数密集计算；
- Decoder 偏任务头与生成/分类输出。

单一并行策略通常无法同时兼顾吞吐、显存与易用性，因此需要构建**分阶段可独立配置并行度**的统一训练框架。

### 2.2 目标
本项目需实现一个基于 PyTorch 的分布式训练框架示例，满足：

1. **三阶段 Pipeline 架构**：Encoder / LLM / Decoder 分别独立拥有 Process Group，且 GPU 独占。
2. **阶段独立并行度**：每个阶段拥有独立的 `tp_size` 与 `dp_size`。
3. **严格卡数关系**：
   - `stage_world_size = tp_size * dp_size`
   - `world_size = encoder_world + llm_world + decoder_world`
4. **训练效率优先**：关注吞吐、通信开销、显存占用、流水线气泡率。
5. **算法易用性优先**：配置清晰、接口统一、可诊断、可扩展。

---

## 3. 范围与非目标

### 3.1 范围（In Scope）
- 三阶段异构并行拓扑设计与初始化流程。
- 阶段内 TP/DP 组构建规则。
- 跨阶段前向/反向通信规范。
- Pipeline 调度（支持至少 GPipe 与 1F1B）。
- 配置系统、运行时日志、监控指标、容错与 checkpoint 方案。
- 示例级代码结构与最小可运行样例要求。

### 3.2 非目标（Out of Scope）
- 不要求实现全量生产级调度器（如集群资源编排器）。
- 不强制绑定特定 Transformer 内核库（可选 Megatron-style 切分策略）。
- 不要求在本需求文档中给出完整业务模型细节。

---

## 4. 术语定义

- **Stage**：Pipeline 中的阶段，本设计固定为 Encoder / LLM / Decoder。
- **TP（Tensor Parallel）**：单层内张量切分并行。
- **DP（Data Parallel）**：多副本数据并行。
- **PG（Process Group）**：PyTorch 分布式通信组。
- **Micro-batch**：一个 global batch 中参与流水并行调度的微批次。
- **Virtual Pipeline Slot（VPS）**：用于对齐不同阶段 DP 数量的逻辑槽位。

---

## 5. 总体架构需求

## 5.1 逻辑架构

框架采用固定三阶段串行 Pipeline：

`Encoder Stage -> LLM Stage -> Decoder Stage`

每个 Stage 由独立 GPU 子集承载，并包含：
- 1 个 Stage PG（阶段全体 rank）
- N 个 TP PG（按 `tp_size` 切分）
- M 个 DP PG（按 `dp_size` 切分）

> 约束：任意 rank 只能归属于一个 Stage；不得跨 Stage 复用同一 GPU。

## 5.2 world_size 计算约束

对任意阶段 `s in {enc, llm, dec}`：

- `stage_world_size_s = tp_size_s * dp_size_s`

全局：

- `world_size = stage_world_size_enc + stage_world_size_llm + stage_world_size_dec`

示例：

- encoder(tp=2, dp=2) -> 4
- llm(tp=4, dp=2) -> 8
- decoder(tp=2, dp=2) -> 4
- **world_size = 4 + 8 + 4 = 16**

## 5.3 Rank 编排原则

建议采用连续 rank 区间分配，便于拓扑理解和部署：
- Encoder: `[0, enc_world-1]`
- LLM: `[enc_world, enc_world+llm_world-1]`
- Decoder: `[enc_world+llm_world, world_size-1]`

每阶段内部再映射为二维坐标：
- `stage_rank = global_rank - stage_rank_start`
- `tp_idx = stage_rank % tp_size`
- `dp_idx = stage_rank // tp_size`

---

## 6. 功能需求（FR）

## 6.1 配置与校验

### FR-01 配置文件
框架必须支持 YAML/JSON 配置，至少包含：
- 三阶段模型定义（模块类名、参数路径、超参）。
- 每阶段 `tp_size`、`dp_size`。
- 训练参数（global batch、micro batch、梯度累积步数、精度策略）。
- 通信参数（backend、timeout、bucket size）。

### FR-02 参数校验
启动前必须完成以下硬校验：
1. `world_size` 是否匹配三阶段卡数之和。
2. 所有 `tp_size >= 1` 且 `dp_size >= 1`。
3. `stage_world_size == tp_size * dp_size`。
4. rank 分配无重叠、无遗漏。
5. micro-batch 数满足所选 Pipeline 调度最小要求。

校验失败时应抛出可读错误（包含期望值/实际值/修复建议）。

## 6.2 Process Group 管理

### FR-03 组创建
系统必须创建以下通信组：
- `pg_stage_{enc|llm|dec}`
- `pg_tp_{stage}_{dp_idx}`
- `pg_dp_{stage}_{tp_idx}`
- 跨阶段边界通信组（Encoder<->LLM，LLM<->Decoder），用于激活与梯度点对点传输。

### FR-04 生命周期
- 初始化顺序：`dist.init_process_group` -> stage 判断 -> 子组创建。
- 支持统一销毁接口，确保异常退出时释放通信资源。

## 6.3 模型与运行时

### FR-05 阶段模型接口统一
每阶段模型必须遵循统一接口：
- `forward(inputs, meta) -> outputs, meta`
- `backward(grad_outputs, meta) -> grad_inputs`
- `state_dict()/load_state_dict()`

### FR-06 Pipeline 调度
至少支持两种调度：
1. **GPipe（全部前向后全部反向）**
2. **1F1B（稳态一前一后）**

支持通过配置切换，默认 1F1B。

### FR-07 阶段独立 DP/TP
框架必须允许阶段间并行度不同（例如 LLM 的 TP 更大），并保持：
- 调度可执行；
- 数据路径可追踪；
- 不引入跨阶段参数同步耦合。

## 6.4 数据与优化器

### FR-08 数据分发
- DP 维度内应使用标准分布式采样（`DistributedSampler`）。
- 当三阶段 DP 数不一致时，系统需通过 VPS 逻辑映射保证样本路由均衡。

### FR-09 优化器与精度
- 支持 AdamW。
- 支持 BF16 或 FP16 混合精度。
- 支持梯度裁剪、梯度累积。
- 支持按阶段独立学习率组（可选）。

## 6.5 Checkpoint 与恢复

### FR-10 Checkpoint
支持：
- 阶段内分片保存（TP/DP aware）。
- 训练状态保存（optimizer、scaler、step、rng）。
- 增量或全量保存策略（至少支持全量）。

### FR-11 恢复
恢复时必须校验：
- 阶段并行拓扑是否兼容。
- 参数分片形状是否一致。
- 不兼容时给出明确报错与重映射建议。

## 6.6 可观测性与诊断

### FR-12 指标
至少采集：
- step time、tokens/s、samples/s
- GPU 显存峰值
- 各阶段前向/反向耗时
- 跨阶段通信耗时与带宽估计
- pipeline bubble ratio

### FR-13 日志
日志应具备：
- 全局 step 粒度摘要
- rank=0 汇总 + 异常 rank 明细
- 可选 JSON 格式，便于接入监控平台

---

## 7. 性能与效率需求（PR）

## 7.1 吞吐目标
- 在示例规模下，1F1B 相较 GPipe 在稳态吞吐应有可观提升（目标 >= 10%，具体与模型结构相关）。
- Pipeline 预热后，bubble ratio 需可观测并可优化。

## 7.2 通信效率
- 阶段边界通信应采用异步发送/接收（`isend/irecv`）并支持与计算重叠。
- 梯度同步应进行 bucket 化，减少小包通信。
- 优先同机高速互联 rank 编排（如 NVLink 域内优先）。

## 7.3 显存效率
- 支持 activation checkpoint。
- 支持按阶段独立的重计算策略（LLM 阶段可更激进，Encoder/Decoder 可更保守）。

## 7.4 扩展性
- 允许三阶段分别增减 `tp_size` 与 `dp_size`，不需要改动核心调度代码。
- 模块接口保持稳定，支持替换不同 Encoder/Decoder 头部实现。

---

## 8. 易用性需求（UR）

## 8.1 配置易用
- 提供最小配置模板与完整配置模板。
- 错误提示必须包含具体键路径（如 `stages.llm.tp_size`）。

## 8.2 启动易用
- 提供统一启动命令，示例：
  - `torchrun --nproc_per_node=16 train.py --config configs/tri_stage.yaml`
- 启动后输出拓扑摘要表（每阶段 rank、tp/dp 划分）。

## 8.3 开发易用
- 阶段模型以插件方式注册（registry），新增阶段实现不改框架核心。
- 通信、调度、优化器解耦，便于单模块调试。

---

## 9. 关键设计细节

## 9.1 多阶段独立 DP 的 VPS 映射

当 `dp_enc`、`dp_llm`、`dp_dec` 不同时，引入：

- `vps_size = lcm(dp_enc, dp_llm, dp_dec)`

对每个 `vps_id in [0, vps_size)`：
- `enc_replica = vps_id % dp_enc`
- `llm_replica = vps_id % dp_llm`
- `dec_replica = vps_id % dp_dec`

作用：
- 将输入样本先映射到 VPS，再映射到各阶段副本；
- 保证长期统计下负载均衡；
- 避免要求三阶段 DP 必须完全相等。

## 9.2 Pipeline 批次约束

建议最小微批次数：

- `num_micro_batches >= pipeline_depth`（基础可运行）
- 推荐 `num_micro_batches >= 2 * pipeline_depth`（降低气泡）

其中 `pipeline_depth = 3`（固定三阶段）。

## 9.3 典型训练步流程（1F1B）

1. Encoder 对 micro-batch 前向，激活发送至 LLM。
2. LLM 前向后发送至 Decoder。
3. Decoder 计算 loss 并触发反向，梯度逐级回传至 LLM、Encoder。
4. 每阶段在 DP 组内完成梯度同步。
5. 梯度累积步满足后执行 `optimizer.step()` 与 `zero_grad()`。

---

## 10. 参考配置示例（与题设一致）

```yaml
distributed:
  backend: nccl
  world_size: 16
  init_method: env://

stages:
  encoder:
    tp_size: 2
    dp_size: 2
    model_cls: EncoderModel
  llm:
    tp_size: 4
    dp_size: 2
    model_cls: LLMModel
  decoder:
    tp_size: 2
    dp_size: 2
    model_cls: DecoderModel

pipeline:
  schedule: "1f1b"          # gpipe | 1f1b
  num_micro_batches: 8
  overlap_p2p_comm: true

training:
  global_batch_size: 256
  micro_batch_size: 4
  grad_accum_steps: 8
  precision: bf16           # bf16 | fp16 | fp32
  optimizer:
    type: adamw
    lr: 2e-4
    weight_decay: 0.01
```

---

## 11. 模块划分要求（示例工程）

建议代码结构：

```text
disttrain/
  configs/
    tri_stage.yaml
  dist/
    topology.py          # rank/stage/tp/dp 映射
    groups.py            # process group 创建与管理
    p2p.py               # 跨阶段通信封装
  pipeline/
    scheduler.py         # GPipe / 1F1B
    engine.py            # 训练主循环
  models/
    encoder.py
    llm.py
    decoder.py
    registry.py
  train.py
```

模块职责：
- `topology.py`：计算 world_size、校验配置、导出 rank 映射。
- `groups.py`：创建 stage/tp/dp PG，并提供查询 API。
- `scheduler.py`：负责 micro-batch 级流水调度。
- `engine.py`：组织前后向、梯度同步、优化器步进、日志上报。

---

## 12. 验收标准（Acceptance Criteria）

## 12.1 功能验收
- 能以题设示例拓扑启动 16 卡训练。
- 日志中可打印三阶段 PG、TP PG、DP PG 的 rank 列表。
- 至少一种调度策略（1F1B）可稳定运行 100+ step。
- checkpoint 保存与恢复可用，恢复后 loss 连续。

## 12.2 性能验收
- 提供 step time、tokens/s、显存峰值统计。
- 在相同配置下，1F1B 吞吐不低于 GPipe。
- 阶段间通信耗时可观测且可定位瓶颈 rank。

## 12.3 易用性验收
- 错配配置可在启动前被拦截并显示可读错误。
- 新增一个自定义 Decoder 类，仅通过注册和配置即可接入训练。

---

## 13. 风险与缓解

1. **阶段负载不均导致流水线阻塞**
   - 缓解：支持阶段耗时 profiling、动态调大 micro-batch、优化最慢阶段算子。
2. **跨阶段通信成为瓶颈**
   - 缓解：异步通信与计算重叠、融合传输、拓扑感知 rank 编排。
3. **DP 不一致带来数据路由复杂度**
   - 缓解：引入 VPS 映射并提供调试可视化输出。
4. **恢复时并行拓扑变化**
   - 缓解：checkpoint 中保存拓扑元信息并在恢复前强校验。

---

## 14. 里程碑建议

- **M1（基础可跑）**：完成拓扑/PG/最小前后向串联。
- **M2（训练闭环）**：完成 1F1B、优化器、梯度同步、日志。
- **M3（工程化）**：完成 checkpoint、配置校验、故障诊断。
- **M4（性能优化）**：通信重叠、重计算策略、瓶颈分析工具。

---

## 15. 结论

该设计通过“**三阶段独立 PG + 阶段内独立 TP/DP + 可配置 Pipeline 调度**”实现异构并行训练框架，既满足题设中的严格卡数关系和架构约束，又在训练效率与算法易用性上具备可扩展落地路径。作为示例工程，可在较小实现复杂度下覆盖大模型分布式训练的核心机制，并为后续生产化演进提供清晰边界。
