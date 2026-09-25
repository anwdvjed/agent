# 数据、金标与成对重放规范

## 轨迹 JSONL

每行一个独立轨迹对象，字段如下。运行 `demo` 会生成完整可解析例子。

```json
{
  "episode_id": "opaque-episode-id",
  "group_id": "opaque-task-family-id",
  "split": "train",
  "source": "reviewed-local-corpus",
  "synthetic": false,
  "trusted_goal": "用户的可信目标",
  "policy": {
    "id": "policy-v1",
    "clauses": [{"id": "clause-1", "text": "组织的明确规则"}]
  },
  "events": [],
  "annotation": {
    "kind": "exact",
    "event_step": 2,
    "event_type": "privacy_leakage",
    "verified": true,
    "provenance": {"reviewers": ["reviewer-a", "reviewer-b"], "revision": "gold-v1"},
    "utility": false,
    "attack_intent": true,
    "complete": true
  },
  "pairs": {}
}
```

上面的空 `events` 仅用于展示字段；真实输入必须包含事件。`event_step`、`lower_step`、`upper_step` 是 **0 起始**的事件下标；模型内部的事件时间是**相对候选动作、从 1 开始**的时间格。当前候选动作造成首个违规，对应 `event_time=1`。

事件包含 `event_id`、非递减 `timestamp`、`observation`、`action`、`permission_receipt`、`runtime`。观察必须是提出动作前已可获得的信息。工具执行结果进入后一个事件的 observation，不能放回当前候选动作的输入。

```json
{
  "event_id": "opaque-event-id",
  "timestamp": 1.0,
  "observation": {
    "source_id": "opaque-source-id",
    "source_type": "tool_output",
    "trust_level": "untrusted",
    "content": "动作前已经观察到的内容",
    "data_ids": ["/documents/report.txt"]
  },
  "action": {"tool": "read_document", "arguments": {"path": "/documents/report.txt"}},
  "permission_receipt": {
    "granted_scopes": ["data:read"],
    "receipt_verified": true,
    "approval_verified": false
  },
  "runtime": {
    "actor_verified": true,
    "known_destinations": [],
    "sensitive_data_ids": [],
    "policy_evaluation_verified": true,
    "policy_clause_results": {"clause-1": true}
  }
}
```

权限或策略未提供时可以保持 false/unknown，用于研究预测；这类输入不能直接用来证明学习模型的在线防御增益，因为硬检查会阻断。不要为了提高正常任务通过率而伪造回执。

## 标签语义

| kind | 必填字段 | 训练处理 |
|---|---|---|
| `exact` | `event_step`、`event_type` | 首次事件离散生存似然；只保留截至首次事件的前缀 |
| `interval` | `lower_step`、`upper_step`、`event_type` | 对区间内首次事件概率求和后取负对数；不假定最早可能事件后的前缀仍安全 |
| `right_censored` | `observed_steps` | 仅使用已观察到的无事件窗口；不足 H 不作为完整 H 的负样本 |
| `weak` | 可选 `trajectory_unsafe` | 不推断时间，prepare 排除并统计，全部为 weak 则明确报错 |

所有非 weak 标签需要 `verified=true` 和非空 `provenance`，其中 `verified` 是数据生产方的质量声明，不能代替人工审查。推荐两位独立标注者标注首个违规动作和条款，冲突裁决后再冻结；确实只能定位范围时使用 interval。

`complete=true` 表示环境已验证到达无后续动作的终止状态。对已完成且没有事件的轨迹，未来是吸收的无事件状态，因此完整 H 的标签为安全；中断、超时或仅观察到日志尾部不应随意设置 complete=true。若环境允许恢复后继续执行，应使用普通右删失。

区间跨越 H 时，只监督到下界之前的生存部分，不将区间终点强行裁成 H 内事件。时间 MAE 仅评估精确且落在 H 内的标签。区间发生在 H 内时可用于 horizon 二分类指标；短删失样本从这项指标中排除。

首事件原因目前为 `privacy_leakage`、`computer_security`、`financial_loss`、`data_loss`、`unauthorized_external_action`。未定义类别应先统一 ontology 并重新训练，不得默认为任意一类。

## 独立标注文件

`--annotations` 接受 JSONL，每行是 `{"episode_id": "...", "annotation": {...}}`。它覆盖原轨迹 annotation，不进入模型输入；重复 ID 和未知 episode ID 会拒绝。分组及输入内容不会被标注文件改变。

## 成对干预

`pairs` 的第一层键是原轨迹候选动作的 0 起始下标，例如 `"2"`；第二层是分支名，例如 `redact_sensitive`。每个分支包含：

```json
{
  "request": {"trusted_goal": "...", "policy": {}, "history": [], "candidate": {}},
  "baseline_digest": "原始候选request的canonical_digest",
  "request_digest": "本分支request的canonical_digest",
  "verified": true,
  "provenance": {"environment": "reviewed-adapter-v1", "snapshot_digest": "..."},
  "observed_steps": 3,
  "annotation": {
    "kind": "exact",
    "event_step": 1,
    "event_type": "privacy_leakage",
    "verified": true,
    "provenance": "isolated-replay-123",
    "utility": false,
    "complete": true
  }
}
```

request 必须完整有效。摘要使用 `agent1.checkpoint.canonical_digest`。成对 request 的可信目标、policy、已经执行的 history、候选 event_id 必须与基线一致，只修改候选动作及其可信运行状态。分支 annotation 的下标从**分支候选动作作为第 0 步**重新开始；observed_steps 是该分支实际观察的动作数。

仅修复副本的输入，不算完成重放。必须从同一环境快照执行修复并继续任务，测量未来违规与任务效用。某分支没有结果，就不填写；训练会屏蔽缺失标签，绝不填 0 假装安全。审批必须来自可信审批服务，沙箱需要真实环境支持。

分支名称：`verified_approval`、`redact_sensitive`、`constrain_destination`、`sandbox`、`restrict_capability`；原始分支使用 episode annotation。内置演示只产生真实模拟的脱敏分支；所有其他具备合法成对结果的分支都可以训练。

## 数据隔离和匿名标识

同一原始任务、重述、攻击版本、干预版本必须使用同一 group_id。split 不可跨组冲突；默认哈希划分不会保证小数据每个集合均非空，因此小数据应显式指定。ID 不得包含安全标签、预期事件或测试答案。适配器的隐藏攻击目标只能放在标签/执行环境中。

prepare 生成 manifest，记录源文件哈希、每个 split 文件哈希、组分配、注册表哈希和合成标记；加载前逐项验证。不要手动修改 processed 文件，应修改源数据并生成新版本。

## Agent Astra 解释监督与执行载荷

可选解释金标放在 `annotation.explanation_by_step["<step>"]`，不进入请求或模型特征。对象必须包含 `version="agent1-explanation-labels-v1"`、`verified=true`、非空 `provenance`、当前前缀请求的 `request_digest`，以及可选的 `concepts`（按固定 `CONCEPTS` 名称）和 `edges`（稳定 `edge_id` 到布尔标签）。未知项省略或保持 null；不得从规则或模型预测补造金标。成对修复可把同一对象放在分支 annotation 的 `explanation` 字段，并绑定该分支请求。`prepare_dataset` 会校验边 ID 与实际图相符；`collate_rows` 以样本内稳定 ID 顺序对齐标签，缺失分支全部屏蔽。

`agent-astra-dataset-v1` 数据清单需要重新运行 `prepare` 生成。工具名和参数键不得经 Unicode NFKC/空白规范化后改变执行语义；参数值保留原始字符串，由 `exact_action_digest` 绑定被批准的确切载荷。模型特征仍允许有损编码，不能用特征摘要替代执行载荷验证。JSON 读取拒绝重复键和非有限数字。
