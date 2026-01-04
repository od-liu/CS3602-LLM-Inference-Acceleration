# Attention Head Specialization & Pruning (注意力头特化与剪枝)

## 1. 核心动机

Pythia模型规模较小，头的数目比较少且功能相对固定。这些头的功能可能高度重复或有些是"死头"。进行这个创新的核心动机是希望可以在不进行训练的情况下加速pythia系列模型的推理速度。

## 2. 创新思路

1. **动态探查：** 在token生成过程中，我们对每个头的行为进行"探查"。记录每个头的注意力熵（entropy）、最大注意力值等统计数据。

2. **实时分类与剪枝：** 根据这些统计数据，实时给所有个头分类：
   - **"定位头" (Positional Heads)：** 熵很低，总是关注固定的相对位置（如前一个token）。
   - **"汇聚头" (Gathering Heads)：** 熵较高，关注内容相关的token。
   - **"死头" (Dead Heads)：** 注意力分布接近均匀分布，或者值很低。

3. **应用特化策略：**
   - 对**"定位头"**，我们**强制**它只关注极小的窗口（比如前3个token），极大地减少它的计算量。
   - 对**"汇聚头"**，我们才应用我们前面讨论的复杂KV Cache策略（如预测性缓存）。
   - 对**"死头"**，在后续的生成中，我们可以**完全跳过它们的计算**！或者用一个固定的、随机的向量来填充它们的输出，代价极小。

**为什么创新：** 它把"一刀切"的KV压缩策略，变成了"因头制宜"的精细化手术。它利用了模型自身的结构冗余，是真正意义上的动态、自适应优化。

---

## 3. 环境约束分析 (Mac + MPS)

### 3.1 Pythia 模型架构参数

| 模型 | 层数 | 每层注意力头数 | 总头数 | 隐藏维度 | 头维度 |
|------|-----|--------------|--------|---------|-------|
| pythia-70m | 6 | 8 | 48 | 512 | 64 |
| pythia-2.8b | 32 | 32 | 1024 | 2560 | 80 |

### 3.2 MPS 特有挑战

| 挑战 | 详情 | 应对策略 |
|------|------|---------|
| **算子支持不完整** | `unfold_backward` 等操作未实现 | 避免使用不支持的操作，使用标准 PyTorch 操作 |
| **内存限制** | MPS 会尝试占用所有可用内存 | 分批处理，限制 batch size 为 1 |
| **精度问题** | float16 支持可能不稳定 | 使用 float32，必要时再尝试混合精度 |
| **调试困难** | GPU 错误信息不如 CUDA 详细 | 添加充分的 CPU fallback 代码进行验证 |
| **output_attentions 开销** | 返回注意力权重会显著增加内存使用 | 仅在分析阶段启用，剪枝阶段关闭 |

### 3.3 可行性结论

**整体可行性：✅ 可行，但需分阶段谨慎实施**

- 分析阶段（Step 1-2）：完全可行，主要是数据收集和可视化
- 剪枝阶段（Step 3-4）：中等难度，需要修改计算图但不涉及 MPS 不支持的操作

---

## 4. 细化行动方案

### 第一步：Pythia-70m 注意力分布分析

**目标：** 验证 Pythia-70m 模型是否存在注意力头特化现象

**实现细节：**

```python
# 核心代码框架
def analyze_attention_heads(model, tokenizer, text, max_tokens=2048):
    """
    收集每个 head 在处理序列时的注意力分布统计
    
    返回统计指标：
    - attention_entropy: 每个 head 的平均注意力熵
    - max_attention_position: 最大注意力位置的分布（相对位置）
    - sink_attention_ratio: 分配给前 4 个 token 的注意力比例
    - uniform_score: 与均匀分布的 KL 散度（识别死头）
    """
    # 启用 attention 输出
    outputs = model(input_ids, output_attentions=True, use_cache=True)
    attentions = outputs.attentions  # tuple of (batch, heads, seq_len, seq_len)
    
    # 计算统计指标...
```

**可视化方案：**

1. **热力图矩阵**：每层每头的平均熵值 (layers × heads)
2. **位置偏好图**：每个 head 对不同相对位置的注意力分配
3. **时序变化图**：随序列长度增长，各 head 特性的变化
4. **聚类分析图**：基于统计指标的 head 分类结果

**预期输出文件：**
- `results/attention_analysis_70m/`
  - `entropy_heatmap.png`
  - `position_preference.png`
  - `head_classification.json`

**难点与应对：**

| 难点 | 原因 | 解决方案 |
|------|------|---------|
| 内存压力 | attention 矩阵 O(n²) | 分段处理：每 256 tokens 收集一次，聚合统计 |
| MPS 兼容性 | output_attentions 可能有问题 | 先在 CPU 上验证代码正确性，再迁移到 MPS |
| 统计指标设计 | 需要有意义的指标 | 参考 SnapKV 的 attention pattern 分析方法 |

**预计工作量：** 1-2 天

---

### 第二步：Pythia-2.8b 验证

**目标：** 确认注意力头特化现象在更大模型上依然存在

**实现细节：**
- 复用 Step 1 的分析代码
- 由于头数更多（1024 vs 48），需要更智能的聚合策略

**难点与应对：**

| 难点 | 原因 | 解决方案 |
|------|------|---------|
| 模型加载慢 | 2.8b 参数量大 | 使用 `low_cpu_mem_usage=True`，已在 benchmark.py 中实现 |
| 内存不足 | 2.8b + attention 输出 | 只分析部分层（如每隔 4 层采样），或使用更短序列 |
| 1024 头分析 | 可视化复杂 | 按层聚合，或只展示典型层的详细分析 |

**预计工作量：** 0.5-1 天

---

### 第三步：Pythia-70m 头剪枝实现

**目标：** 基于 Step 1 的分析结果，实现头级别的优化策略

**3.1 实现 "死头" 剪枝**

```python
class DeadHeadPruner:
    """
    对于被识别为"死头"的注意力头，跳过其计算
    
    实现方式：
    1. 在 forward 之前设置 attention mask 为 0
    2. 或者直接用零向量替换输出
    """
    def __init__(self, dead_head_indices: Dict[int, List[int]]):
        # dead_head_indices: {layer_idx: [head_idx1, head_idx2, ...]}
        self.dead_heads = dead_head_indices
```

**3.2 实现 "定位头" 优化**

```python
class PositionalHeadOptimizer:
    """
    对于只关注固定相对位置的头，限制其 KV cache 大小
    
    实现方式：
    1. 为这类头维护一个极小的滑动窗口（如 window_size=8）
    2. 保持其他头的完整 KV cache
    """
    def compress_positional_heads(self, past_key_values, positional_heads):
        # 只压缩特定头的 KV cache
        pass
```

**3.3 Benchmark 集成**

在现有 `kvcompress/methods/` 目录下新建：
- `head_pruning.py`：实现上述两种优化器
- 修改 `__init__.py` 暴露新方法
- 修改 `scripts/benchmark.py` 添加新的测试配置

**难点与应对：**

| 难点 | 原因 | 解决方案 |
|------|------|---------|
| 头级别 KV 操作 | 需要按 head 维度切片 | 利用已有的 tensor 操作，KV shape 为 (batch, **heads**, seq, dim) |
| 计算图修改 | 跳过死头计算 | 不修改模型代码，通过 attention mask 或后处理实现 |
| 性能验证 | 需要排除噪声 | 使用现有的 warmup 机制，多次采样取平均 |

**预期性能收益：**
- 死头剪枝：如果 10% 的头是死头，理论上可减少 10% 的注意力计算
- 定位头优化：如果 30% 的头是定位头，可显著减少这部分 KV cache

**预计工作量：** 2-3 天

---

### 第四步：Pythia-2.8b 验证与优化

**目标：** 将 Step 3 的实现应用到 2.8b 模型

**实现细节：**
- 使用 Step 2 的分析结果确定死头和定位头
- 运行完整 benchmark 比较性能

**关键验证指标：**
- TTFT 变化
- TPOT 变化
- Throughput 变化
- PPL 变化（需确保质量损失可控）

**预计工作量：** 1 天

---

## 5. 代码结构建议

```
kvcompress/
├── methods/
│   ├── __init__.py          # 添加 head_pruning 导出
│   ├── head_pruning.py      # 新增：头剪枝实现
│   └── ...
├── analysis/                 # 新增目录
│   ├── __init__.py
│   ├── attention_analyzer.py # 注意力分布分析
│   └── visualize.py         # 可视化工具
scripts/
├── analyze_attention.py     # 新增：运行注意力分析
├── benchmark.py             # 修改：添加头剪枝测试
└── plot_attention_analysis.py # 新增：生成分析图表
results/
├── attention_analysis_70m/  # 70m 分析结果
└── attention_analysis_2.8b/ # 2.8b 分析结果
```

---

## 6. 风险评估与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|---------|
| Pythia 模型不存在明显的头特化 | 中 | 高 | 先完成 Step 1 验证再继续后续步骤 |
| MPS 不支持某些操作 | 中 | 中 | 准备 CPU fallback，牺牲速度换取正确性 |
| 剪枝导致 PPL 恶化过大 | 中 | 高 | 设计保守的剪枝阈值，逐步调整 |
| 2.8b 模型内存不足 | 高 | 中 | 减少分析的序列长度，使用 gradient checkpointing |

---

## 7. 时间线估计

| 阶段 | 内容 | 预计时间 | 里程碑 |
|------|------|---------|--------|
| Step 1 | 70m 注意力分析 | 1-2 天 | 可视化图表 + 初步分类 |
| Step 2 | 2.8b 验证 | 0.5-1 天 | 确认理论基础 |
| Step 3 | 剪枝实现 | 2-3 天 | 可运行的 benchmark |
| Step 4 | 2.8b 应用 | 1 天 | 最终性能报告 |
| **总计** | | **5-7 天** | |

---

## 8. 参考资源

- [SnapKV 注意力分析方法](../SnapKV/snapkv/monkeypatch/snapkv_utils.py)：参考其 attention pattern 计算方式
- [StreamingLLM 实验结果](./STREAMINGLLM_ANALYSIS.md)：作为性能基准对比
- 项目现有 benchmark 基础设施：`scripts/benchmark.py`、`kvcompress/evaluate.py`
