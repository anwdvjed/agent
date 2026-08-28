# Agent New 工程总结

> 项目版本：`0.1.0`  
> 总结日期：2026-08-25  
> 工程目录：`/Users/viper3/Documents/Federated Learning/agent-new`

## 1. 项目概述

Agent New 是一个面向工具型智能体的执行前安全预测与准入工程。系统在候选工具动作产生真实副作用之前，联合分析：

- 可信任务目标与政策条款；
- 历史观察、工具动作和来源信息；
- 当前尚未执行的候选动作；
- 权限、审批、目标地址和运行环境；
- 来源、数据、动作、目的地、权限和政策之间的图关系。

模型输出未来时间窗内首次风险事件的发生概率、风险类型和预计发生动作步，并将硬安全约束、模型预测、校准结果和不确定性组合为运行时决策。

当前仓库实现的是完整的工程骨架和训练/评测流水线，但不包含经过真实数据训练的正式 checkpoint，也不声称已经获得优于现有论文方法的实验结果。

## 2. 总体架构

```text
可信目标 + 政策 + 历史轨迹 + 未执行候选动作
                     │
                     ▼
             可信工具能力注册表
     operation / scopes / argument schema / effects
                     │
                     ▼
       严格输入验证与可信上下文签名验证
                     │
                     ▼
       持久化 Provenance-Permission 异构图
  Actor / Source / Data / Action / Destination / Permission / Policy
                     │
                     ├───────── 六类干预分支 ─────────┐
                     │ none / approval / redact       │
                     │ destination / sandbox / deny   │
                     ▼                                ▼
        Step-Time 时序编码器               关系感知图编码器
                     │                                │
                     └───────── 融合与证据掩码 ───────┘
                                      │
                                      ▼
                    因子化 Competing-Risk 风险头
                 P(event at t) × P(cause | event at t)
                                      │
                                      ▼
                   Ensemble + Group Conformal 校准
                                      │
                                      ▼
                  ALLOW / HOLD / REPLAN / CONFIRM / BLOCK
```

## 3. 核心算法改进

### 3.1 因子化首次事件模型

旧实现将“无事件”和多个风险类型放在同一个 softmax 中，风险类型较多时会导致未训练模型的累计风险接近 1。新模型将事件发生和事件原因拆开：

```text
q_t       = sigmoid(event_logit_t)
pi_t,k    = softmax(cause_logits_t,k)
P(no event at t) = 1 - q_t
P(cause k at t)  = q_t × pi_t,k
```

模型使用低基础事件率初始化。当前 smoke 中未训练模型的初始五步累计风险约为 `0.0500`，不再出现初始化饱和。

主要实现：[agent_new/model.py](agent_new/model.py)、[agent_new/losses.py](agent_new/losses.py)。

### 3.2 稳定的右删失损失

训练支持：

- 已观察到的首次事件时间与类型；
- 安全轨迹的实际观察截止时间；
- 只有时间窗标签的弱干预监督；
- 多个风险类型之间的竞争风险。

损失直接在 logit、log-survival 和 log-softmax 空间计算，不再对已饱和概率执行 `clamp + BCE`。极端 logits 的有限值和梯度均有测试覆盖。

### 3.3 干预分支

每个候选动作最多构造六个共享参数分支：

| 分支 | 含义 |
|---|---|
| `none` | 原始候选动作 |
| `verified_approval` | 获得可信审批后的假设状态 |
| `redact_sensitive` | 敏感数据脱敏后的动作 |
| `constrain_destination` | 将目标限制到明确允许的地址 |
| `sandbox` | 在受控环境中执行可沙箱化动作 |
| `restrict_capability` | 禁止该能力，本质上对应拒绝动作 |

无实际字段变化的干预会被识别为 no-op，并复用原始分支语义。不同分支不使用独立 dropout，因此分支风险差不会混入随机 mask 噪声。

当前校准工件只认证 `none` 分支。其他分支的风险输出仅用于研究和排序，在完成分支条件校准及可信后置条件验证前，运行时不会自动提交修复动作。

### 3.4 证据掩码

模型先对来源图边进行评分，再将证据权重直接用于消息传递，而不是只生成事后解释。损失包含：

- 证据监督；
- sufficiency；
- necessity；
- sparsity。

运行时输出可读的 `evidence_edges`，包含边关系及脱敏后的源/目标节点信息。系统明确标记 `evidence_path_verified=false`：当前边分数解释的是图分支贡献，不等于已经验证的完整因果路径。

### 3.5 安全概念与局部单调性

关键概念包括权限过量、敏感数据流、异常目的地、缺少审批、不可逆性和未知能力。对于固定融合状态，关键概念只能以非负权重提高任意时间窗内的事件发生 logit。

这是一项局部结构保证，不代表端到端全局安全证明。

### 3.6 攻击意图与任务效用辅助头

在首次风险预测之外，当前模型还实现：

1. 隐藏攻击/side-task 意图概率；
2. 主任务成功/效用概率。

LinuxArena 的 `has_side_task` 用于攻击意图监督，`side_task_success` 用于危害结果，`main_task_success` 用于效用监督。side task、policy 和最终结果字段不会进入模型输入。

## 4. 可信数据与图构造

### 4.1 工具能力注册表

[agent_new/registry.py](agent_new/registry.py) 是工具语义的唯一可信来源。注册表定义：

- 操作类型；
- 必需 scope；
- 是否要求验证审批；
- 参数名称及语义角色；
- 副作用；
- 工具别名。

Agent 自报的 `operation`、`required_scopes`、`approval_required` 和 `side_effects` 会被忽略。未知工具、未知参数和不满足安全不变量的注册表条目均 fail closed。

### 4.2 Unicode 特征

[agent_new/features.py](agent_new/features.py) 使用角色分离的 Unicode 字符 n-gram 和 UTF-8 byte n-gram 稳定哈希，能够处理中文和其他非 ASCII 输入。

输入具有单字段、集合、嵌套深度、历史长度和全请求文本预算限制。安全相关文本或参数超限时会拒绝请求，而不是静默截断后继续执行原动作。

### 4.3 持久异构图

[agent_new/graph.py](agent_new/graph.py) 构建以下节点：

- Actor；
- Source；
- Data；
- Action；
- Destination；
- Permission；
- Policy；
- Intervention。

相同实体 ID 在轨迹内复用。对象敏感性作为持久 taint 保留；权限、审批和来源信任等动态状态记录在时间局部边上。边时间使用相对候选动作的 age，而不是可能饱和的 Unix 绝对时间。

## 5. 运行时安全边界

### 5.1 Hard constraints

以下情况不会交给模型自由决定是否放行：

- 未注册工具；
- 未注册参数；
- 权限回执未验证；
- 缺少 scope；
- Actor 未验证；
- 缺少所需审批；
- 目的地不在明确 allowlist；
- 政策评估未验证或政策条款被拒绝；
- 历史中出现未知能力；
- 能力已被限制。

安全布尔值必须是 JSON `true/false`；字符串 `"false"` 会被拒绝。

### 5.2 目的地规则

裸 allowlist 值表示精确匹配：

```text
alice@example.com
```

域级权限必须显式声明：

```text
domain:internal.example
```

因此允许 `alice@example.com` 不会自动允许 `mallory@example.com`。

### 5.3 可信上下文证明

[agent_new/trust.py](agent_new/trust.py) 提供请求级 HMAC attestation：

- payload SHA-256；
- issuer；
- subject；
- issued/expiry time；
- 单次 nonce；
- HMAC signature。

签名覆盖任务、候选动作、权限、政策和环境上下文。重放、篡改、错误 issuer 和过期请求均被拒绝。

生产环境应使用外部持久 nonce 存储、密钥服务或非对称签名；内置 HMAC 主要用于提供清晰的可信边界和本地集成示例。

### 5.4 Ensemble 与校准

运行时要求：

- 至少两个内容不同的模型状态；
- 不同 `training_run_id`；
- 不同 seed；
- 相同模型配置、事件类型、注册表 digest 和校准 digest；
- 校准记录的 `ensemble_z` 与运行时完全一致。

重复 checkpoint 不能伪装成 ensemble。未校准、单模型或无法确认独立训练 lineage 时，模型不能产生 `ALLOW`。

### 5.5 Commit token

`ALLOW` 结果不会把敏感 action payload、权限回执或政策全文打印到 stdout。普通调用方只获得：

- event ID；
- tool name；
- opaque SHA-256；
- `require_digest_match_before_execution=true`。

Executor 必须在实际提交前重新构造 canonical action，并调用 digest 校验。任何 TOCTOU 变更都会使提交授权失效。

CLI 决策退出码：

| 决策 | 退出码 |
|---|---:|
| `ALLOW` | 0 |
| `CONFIRM` | 20 |
| `REPLAN` | 21 |
| `HOLD` | 22 |
| `BLOCK` | 23 |

输入、完整性或配置错误使用普通 Python 非零错误码。

## 6. 数据流水线

### 6.1 数据角色

| 数据集 | 角色 | 当前处理 |
|---|---|---|
| ASSEBench safety/security | 训练 | 轨迹和风险结果；事件时间标为弱 terminal proxy |
| LinuxArena | 训练 | 动作轨迹、attack intent、harm outcome、main-task utility |
| ATBench | 冻结外部测试 | 不进入训练；当前 pinned 配置尚待授权下载后补 SHA-256 |
| OpenAgentSafety | OOD 测试 | 不进入初始训练；作为 legacy evaluation-only 来源 |

数据配置位于 [configs/datasets.json](configs/datasets.json)。下载脚本默认只显示计划，只有显式添加 `--execute` 才会进行网络和磁盘写入。

### 6.2 防泄漏设计

- 完整 conversation/task group 只能属于一个 split；
- 内部分割为 80/10/10 train/validation/calibration；
- ASSEBench/LinuxArena 训练 batch 目标比例为 70%/30%；
- side task、policy、source label 和结果标签不会进入模型 request；
- LinuxArena 私有 action reasoning 不进入模型观察；
- ATBench/OpenAgentSafety 只生成 `test_*` split；
- 测试脚本拒绝使用 validation 代替 test。

### 6.3 训练与评测入口

```bash
# 仅查看下载计划
.venv/bin/python scripts/download_datasets.py --profile train
.venv/bin/python scripts/download_datasets.py --profile eval

# 在已授权且文件就绪后转换
.venv/bin/python scripts/prepare_datasets.py --profile train
.venv/bin/python scripts/prepare_datasets.py --profile eval --overwrite

# 检查实际输入模型的 batch
.venv/bin/python scripts/feed_model.py --split train --forward

# 独立 seed 训练
.venv/bin/python scripts/train_model.py \
  --seed 101 --run-id seed-101 --output-dir artifacts/runs/seed-101

.venv/bin/python scripts/train_model.py \
  --seed 202 --run-id seed-202 --output-dir artifacts/runs/seed-202

# 多模型校准
.venv/bin/python scripts/validate_model.py \
  --checkpoint artifacts/runs/seed-101/best \
  --checkpoint artifacts/runs/seed-202/best \
  --output-dir artifacts/validation

# 冻结测试
.venv/bin/python scripts/test_model.py \
  --checkpoint artifacts/validation/checkpoints/seed-101 \
  --checkpoint artifacts/validation/checkpoints/seed-202 \
  --calibration artifacts/validation/calibration.json \
  --operating-threshold artifacts/validation/operating_threshold.json
```

训练、验证和测试脚本显示明确 epoch 进度以及 hazard、attack-intent、utility 等指标。

## 7. Checkpoint 与构建

[agent_new/checkpoint.py](agent_new/checkpoint.py) 将 JSON metadata 和 tensor state 分开存储：

- state 文件按 digest 命名；
- metadata 作为原子 commit point；
- 加载前验证 SHA-256；
- 绑定模型配置、事件类型、注册表和校准 lineage；
- 只调用 `torch.load(weights_only=True)`。

Wheel 已验证包含：

- 所有 `agent_new` 模块；
- `agent_new.datasets` 子包；
- 内置 smoke assets；
- console entrypoint。

## 8. 当前验证结果

截至 2026-08-25：

| 验证项 | 结果 |
|---|---|
| Python compile | 通过 |
| 单元/集成测试 | `31/31` 通过 |
| Forward/backward smoke | 通过 |
| 梯度有限性 | 通过 |
| 六干预分支 | `[1, 6]` 输出正常 |
| 未训练初始累计风险 | 约 `0.0500` |
| Hard fail-closed 原因 | scope、approval、destination、policy 正确触发 |
| Wheel 构建 | 通过 |
| Wheel 数据适配器/资产 | 已包含 |

Smoke 输出中的风险来自随机未训练权重，只用于验证工程调用链，不代表模型安全性能。

测试覆盖的关键安全性质包括：

- Agent 自报能力字段无效；
- 未知工具和参数 fail closed；
- 字符串布尔绕过被拒绝；
- 精确邮箱不会扩大为域级授权；
- 敏感文本不会因 redaction 分支进入普通 metadata；
- 相同/no-op 分支保持一致；
- 证据 mask 实际影响图消息传递；
- 极端 hazard logits 仍有有限梯度；
- 无 outcome supervision 时拒绝训练；
- 重复 checkpoint 不能伪造 ensemble；
- 注册表加载后篡改会被检测；
- 签名篡改、过期和 nonce replay 被拒绝；
- 独立、校准、签名完整的安全请求可产生 `ALLOW`；
- 数据下载默认为 dry run；
- tiny 数据转换、batch 和 loss 闭环通过。

## 9. 当前边界与限制

1. **没有正式训练结果。** 当前没有真实数据训练 checkpoint，不能报告模型优越性。
2. **首次事件标签仍弱。** ASSEBench 和部分 LinuxArena 数据只能提供终点代理，需要人工标注真实首次违规动作。
3. **干预分支尚未认证。** 当前校准仅适用于 `none` 分支；自动 CONFIRM/REPLAN 需要分支条件或同时校准、可信 transform 和后置条件验证。
4. **证据并非完整因果解释。** 证据边只解释图通道，时序通道仍可独立预测。
5. **Conformal bound 不是条件事故概率。** 它是交换性假设下的安全准入界，应避免解释为因果风险置信区间。
6. **可信上下文仍依赖部署基础设施。** HMAC 示例不能替代生产级身份、策略引擎、权限服务和持久防重放存储。
7. **缺少真实执行拦截实验。** 尚未在真实工具中报告被阻止的副作用、ASR、任务完成率和 over-defense。
8. **缺少强基线和消融。** 需要规则、flags-only、GRU/Transformer、graph-only、标准 survival、LLM judge 以及相关论文方法的公平比较。
9. **外部测试文件尚未获取。** ATBench 的 SHA-256 需在授权下载后固定；OpenAgentSafety 也未在本次实现中下载。
10. **高风险部署需签名 checkpoint。** 当前 SHA-256 保证完整性，不保证发布者真实性。

## 10. 后续优先级

### P0：形成可信实验结论

- 建立人工 gold subset：首次违规点、风险类型、证据路径、权限和真实结果；
- 下载并固定 ATBench/OpenAgentSafety 外部测试文件；
- 训练至少 5 个独立 seed；
- 在相同 safe-trajectory FPR 下运行简单强基线；
- 报告多 seed 和 trajectory/scenario cluster 置信区间。

### P1：启用可信干预选择

- 为每个 ToolSpec 声明可支持的 intervention；
- 使用可信 transform，而不是仅修改内存字段；
- 验证干预后置条件；
- 对各分支做 simultaneous/conditional calibration；
- CONFIRM 后取得新签名 receipt 并重新评估；
- 只允许 executor 提交 digest 完全匹配的动作。

### P2：解释性与系统评测

- 联合时序事件与图边 attribution；
- 加入连通证据路径约束；
- 做 deletion/insertion、necessity/sufficiency 外部实验；
- 接入真实 pre-side-effect middleware；
- 报告 ASR、benign utility、误阻断、延迟、吞吐和内存。

## 11. 交付状态

当前工程已具备：

- 可安装 Python package；
- 因子化 competing-risk 模型；
- 可信工具注册表；
- 严格请求和图构造；
- 多干预研究分支；
- 稳定训练损失；
- 多 seed checkpoint lineage；
- group-aware 校准；
- HMAC trusted-context 示例；
- runtime 和 CLI；
- ASSEBench/LinuxArena/ATBench/OpenAgentSafety 数据适配器；
- 训练、校准、测试脚本；
- 31 项自动化测试；
- 中文/英文对比报告文档。

最准确的工程定位是：**已完成可复现、安全默认、可训练的研究平台；尚未完成真实数据上的性能与干预有效性证明。**

