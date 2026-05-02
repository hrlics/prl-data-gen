# Design notes — data generator RL

Open design questions / rationale that's worth keeping around. The runnable how-to is in [readme_data_generator.md](readme_data_generator.md); this file is the "why and what's still open."

---

## Q1. Fittability：F1 vs 纯 recall

**State**: 当前 ([align_z_judge_v1.md](conf/evaluator_prompts/align_z_judge_v1.md) + [verifier_api.judge_alignment](pipelinerl/domains/math/verifier_api.py#L893)) 用 genvf_v6 风格的 F1（recall × precision）。Dry run 阶段先不动，等 wandb 上看到信号再决定要不要换。

### 1.1 角色映射（已固化在 prompt 里）

`verify_proof(summaries=[z], generation=z_prime, ...)`：

- **CRITERION (`{rubric}`)** = `z` — 4B 看到 GT completion 后写的 bullet list
- **MODEL_SUMMARY (`{model_summary}`)** = `z'` — gpt-oss 在没 GT 的情况下重新写的 bullet list

### 1.2 三个分数到底在量什么

| 指标 | 定义 | 该分低意味着 |
|---|---|---|
| Recall | 遍历 z 的每条 bullet `c_i`，问 z' 里有没有 bullet 覆盖它 | **z' 没复现 z** ⇒ z 含 z' 不会自然产生的 GT-specific 内容（z 超出 z'） |
| Precision | 遍历 z' 的每条 bullet `m_j`，问它是否是某条 c_i 的最佳匹配 | **z' 有 z 没有的 bullet** ⇒ z 缺漏 / z' 啰嗦（z' 超出 z） |
| F1 | `2PR/(P+R)` | 任一方向偏离都拉低 F1 |

### 1.3 设计意图 vs F1 实现的差距

**意图**: "z 不能比 z' 多/含独有的 GT-specific 内容；多了说明 4B 在用 GT 信息泄漏，非 fittable，应该降 reward。"

| 方向 | F1 是否捕获 | 是否符合意图 |
|---|---|---|
| z 超出 z'（recall ↓） | ✅ | ✅ 这就是想要的主轴 |
| z' 超出 z（precision ↓） | ✅ | ⚠️ **额外**约束，不在原始意图里 |

precision 拉低分两种情形：
- (a) **z 太精简**漏掉了不需要 GT 也能想到的 bullet — 罚得合理
- (b) **z' 啰嗦/胡扯**了 z 不需要的 bullet — 罚错了 4B

F1 不区分 (a) 和 (b)，一律用 precision 把分扣下去。

### 1.4 副作用（思想实验）

**场景 A — 完美对齐但 z 更短**：
z 是 z' 的子集（z 只是更精简、更聚焦）。
- recall = 1（z 全被 z' 覆盖）
- precision < 1（z' 比 z 多）
- F1 < 1，reward 被压
- 即使 4B 没有任何不 fittable 的内容，也会被罚

**场景 B — 4B 学会"模仿 z' 篇幅 + 加 GT 特例"**：
- z 长度 ≈ z'
- recall 中等（GT-specific 部分被 z' 漏掉）
- precision 高（z 覆盖了 z' 大部分）
- F1 可能反而比"短而准的 z"更高

⇒ **F1 的优化方向偏向"z 长度 ~ z' 长度"，可能鼓励冗余而非精确。**

### 1.5 两条路的取舍

| 选项 | 优势 | 劣势 |
|---|---|---|
| **保持 F1（现状）** | 抑制 4B 输出过短/退化的 z；与 genvf_v6 一致 | reward landscape 偏向 z 篇幅 ≈ z' 篇幅，可能鼓励冗余；多罚一个不在原始意图里的方向 |
| **改成纯 recall** | 与 "z 不能超过 z'" 语义一对一 | 4B 可能学会输出极短 z（"一条最稳的 bullet"）；不过 performance 项会自然惩罚（短 z 信息量少，gpt-oss 续写 score 上不去），所以未必致命 |

### 1.6 改成纯 recall 的实现路径（先不动）

[verifier_api.py:judge_alignment](pipelinerl/domains/math/judge_alignment) 把
```python
score=float(verification.score)
```
改成
```python
score=float(verification.metrics.get("verifier/scores/recall", 0.0))
```
prompt 不用动（recall block 还是要让 judge 输出，因为没它就算不了 recall）。

### 1.6.5 Dry run 实地观察（2026-04-30）

第一批 rollout 跑出来一个典型 case（图论题，"max-degree-5 graph 必有 ≥3/5 edges 是 bichromatic"）：

- **z (4B w/ GT, CRITERION)** — 3 条短 bullet，思路是 *averaging argument*：
  1. greedy 6-coloring（用 max degree 5）
  2. 把 6 色分两组生成 20 个 2-coloring
  3. 在 20 个里取平均 → 至少一个达到 3/5
- **z' (gpt-oss w/o GT, MODEL_SUMMARY)** — 6 条长 bullet 带 LaTeX，思路是 *switching argument on a locally-optimal 2-coloring*：local optimality ⇒ s(v) ≥ ⌈d(v)/2⌉ ⇒ 全局 D ≥ 3/5 E。

两套都是**合法证明**，最终结论一样，但**思路完全不同**。bullet 之间无 1-1 对应：
- recall ≈ 0（z 的 bullet 没在 z' 里）
- precision ≈ 0（z' 的 bullet 没在 z 里）
- F1 ≈ 0；即使换成纯 recall 也是 ≈ 0

⇒ **4B 这次没有 breach fittability，只是 gpt-oss 这次抽中了另一条等价策略**，但 reward 依然为 0。

### 1.6.6 这是比 F1 vs recall 更深的一个问题：multi-strategy mismatch

我们的 fittability 隐含假设：**"z 的内容 ⊆ gpt-oss 不看 GT 也能想到的内容"**。

但 IMO 难题常多解，gpt-oss 一次采样只抽一种。如果 GT 用策略 A、gpt-oss 抽中策略 B，z 复刻 A 时**和 z 真的"GT-leaky"无关**地对不上 z'。

| 缓解方案 | 思路 | 代价 |
|---|---|---|
| **z' 多采样取并** | gpt-oss 调 N 次拿 z'_1, ..., z'_N，把并集当 MODEL_SUMMARY | judge prompt 变长；gpt-oss bill ×N |
| **改 judge 任务** | 不再做 bullet 对齐，改问 "z 是不是 plausible 策略？(0/1)" | 信号变粗，可能 judge 倾向都给 1 |
| **z 也多采样 + best-of-K** | 4B 每 prefix 多采 K 条 z；与 z' 取最大 F1 | 实现复杂；但公平 |
| **接受噪声 + 大 attempts** | 同一 prefix 多 rollout，至少有一条蒙对策略 | 不解决根本问题，只平均 |

### 1.6.7 [Big idea] 用 4B 自己产 z'，而不是 gpt-oss

**State**: 候选改动，dry run 数据看完后做决定。

**改动**: A 调用从 `gpt-oss(no GT) → z'` 改成 `4B(no GT) → z'`。同一个 trainable model 的两次条件采样：一次看 GT (出 z)，一次不看 (出 z')。

**为什么 semantically 更对**:
- 我们要训的对象就是这个 data generator，fittability 的字面定义是 "这个 model 没看 GT 时是否也能想到 z" — 那 reference 就该是同一个 model 的 no-GT 版本。
- gpt-oss 是固定 baseline，bar 比较高也比较死；4B 自己当 reference 会跟着训练 bootstrap 上去 — reference 持续进化，fittability 持续测 marginal GT contribution。

**对 §1.6.6 multi-strategy mismatch 的缓解**:
- 同一个 model 的 with-GT vs no-GT 策略分布**显著重叠**（同样的偏好 + temperature 多样性），不像跨模型那样容易抽到完全不同的证明套路。
- 不是 0 风险但比跨模型好很多。

**主要风险**:
- **Mode collapse to generic z**: 4B 学到一条 generic strategy，无论看不看 GT 都吐它 → z ≈ z' → fittability=1。被 `performance × fittability` 的乘法挡住（generic z 让 performance 起不来），但要监控。
- **狭窄 base policy**: 4B 策略偏好太单一时，z 和 z' 天然重叠 → fittability 太容易满分。需要看 attempts 内的 z diversity。
- **Cold start**: 训练初期 4B(no GT) 可能弱 → z 和 z' 都烂 → F1 噪声大但都低，不太干扰。

**Compute 净收益**:

| 项 | 当前 | 改后 | Δ |
|---|---|---|---|
| 4B local call | 1 (带 GT, ~16k) | 2 (带 GT + 不带 GT) | **+1 次本地** |
| gpt-oss A (predict z') | 1 (~32k) | 删 | **-1 次远程** |
| gpt-oss B/C1/C2 | 不变 | 不变 | 0 |

gpt-oss 是远程瓶颈，砍一次几乎肯定净加速；两次 4B 调用可并发同一 vLLM batch。

**实现路径** (落代码时):
1. 新 prompt `predict_z_no_gt_v1.md` ≈ 抄 [predict_z_v1.md](conf/evaluator_prompts/predict_z_v1.md)
2. `_fittability_chain` 里把 `generate_strategy(...)` 换成 `llm_async_generate(llm, no_gt_prompt, session)` + `parse_summary`
3. `verifier_api.generate_strategy` 可保留（切换方便）也可删
4. 可选：no-GT 采样温度调低（比如 0.5），让 reference 稳一点

**Tradeoff 一句话**:
- "用强模型当 anchor 拉 4B" → 保留 gpt-oss A
- "测这个 model GT-conditioning 的 marginal 收益" (= self-fittability) → 改成 4B

后者更贴 "data generator RL" 的训练目标。

### 1.7 Dry run 时盯哪几个指标判断要不要换

| wandb 指标 | 信号 |
|---|---|
| `verifier/scores/recall_mean` 和 `verifier/scores/precision_mean` 谁拉低了 F1 | recall 主导 → F1 工作正常；precision 主导 → F1 在罚错方向 |
| `tables/verifier_last_k` 里 z 的长度 vs z' | z 一直被压到很短 → 现在说明 fittability 单纯靠"z' 比 z 长"在拉分；z 越来越长 → 鼓励冗余 |
| `verifier/scores/performance_mean` | 如果 fittability ↑ 但 performance 持平 → 只是在学 prompt 风格，不是真在 distill GT |

---

## Q2. Performance 用 proofbench-no-gt 的两个潜在风险

(待 dry run 后填)

- C1 让 gpt-oss 把 z 当 hint 续写 IMO 证明，温度 1.0 + 40k tokens — 方差有多大？
- C2 用 [/home/aviralku/haoranl4/QED-Nano/eval/configs/prompts/gradingbench/proofbench-no-gt.txt](../QED-Nano/eval/configs/prompts/gradingbench/proofbench-no-gt.txt) 评分，会不会有 bias（比如倾向给短证明高分 / 短证明低分）

---

## Q3. attempts=8 + judge 多次温度 1.0 调用 → group-mean 的方差量级

(待 dry run 后填)

- 每个 group = 8 rollouts × 4 judge calls = 32 个独立随机 OpenAI 调用
- 单 sample reward = `fitt × perf` 的乘积，方差被放大
- 想要稳定的训练信号需要 group-mean 的 std < 几个百分点。如果 dry run 看到 group-mean 在大区间跳，要考虑：
  1. judge temperature 降到 0.3
  2. attempts 提到 16
  3. 这俩同时
