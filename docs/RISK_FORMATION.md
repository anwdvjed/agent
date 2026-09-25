# 动作条件下的风险形成预测（Agent1 0.2）

本次扩展聚焦预测模块：同时预测危险条件状态、首次条件齐备时间和实际风险事件时间。它不生成完整未来工具轨迹，也不自动证明因果效应。首先实现隐私泄露场景，实际事件头仍保留原有五类事件。

## 三类时间与状态不能混淆

- **当前条件状态**：现在条件为真还是为假；输入或标签没有证据时使用 null。
- **首次条件齐备时间**：四个条件第一次同时为真。以后条件可以再次失效，但首次齐备这一历史事实不会被撤销。
- **实际事件时间**：真正发生隐私泄露的动作时间。条件齐备后可能脱敏、取消或一直没有发送，所以不一定发生实际事件。

四个条件的固定顺序见 `constants.RISK_CONDITIONS`：

1. `sensitive_data_available`：相关敏感数据已可供后续动作使用。
2. `sensitive_payload_prepared`：敏感内容已加入待披露载荷。
3. `recipient_selected`：已确定载荷的接收方。
4. `disclosure_not_authorized`：该敏感内容的披露不在可信任务授权范围内。

这里的接收方可以是组织内或组织外；“具备发送工具权限”与“有权披露具体敏感内容”不同。四个条件是本版本对一类隐私风险的定义，并非所有隐私事件的普适充分条件。

## 模型输出

共享时序/GNN 编码器与融合状态。新增 `RiskFormationHead` 输出：

- `current_logits [B,4]`：当前四个条件的边际预测。
- `future_logits [B,H,4]`：未来每个动作执行后四个条件的边际预测，允许真→假与假→真。
- `formed_before_logit [B]`：候选动作执行前是否已曾齐备。
- `hazard_logits [B,H]`：此前未齐备条件下，未来首次齐备的离散风险率。

令 q 为“此前曾齐备”概率，h_j 为未来第 j 格的条件风险率，则：

```text
P(首次齐备在过去) = q
P(首次齐备在第 j 格) = (1-q) × product(i<j, 1-h_i) × h_j
P(截至 H 仍未首次齐备) = (1-q) × product(i<=H, 1-h_i)
```

三部分总和为 1。q 与“当前四个条件全真”不是同一量；例如曾经准备好敏感附件、后来又删除附件，此时 q 的目标为 1，而当前载荷条件为 false。

**没有将四个条件的边际概率相乘。** 齐备时间头直接学习联合事件。未来条件头目前是直接多时域预测头，不是递归世界模型，也不描述条件之间的完整联合分布。

实际事件仍由竞争风险头预测。仅对具备本定义监督的样本增加软一致性损失：隐私事件累计概率不应超过已齐备累计概率。这是训练约束，不是概率校准或硬性安全保证。

## 新增标注

在原 `annotation` 中可选增加 `formation`。没有该字段的旧轨迹继续用于实际事件训练，所有新增条件/时间监督为缺失，不补零。

```json
{
  "formation": {
    "version": "privacy-readiness-v1",
    "verified": true,
    "provenance": "reviewed-state-instrumentation-v1",
    "complete": false,
    "initial": {
      "sensitive_data_available": false,
      "sensitive_payload_prepared": false,
      "recipient_selected": null,
      "disclosure_not_authorized": true
    },
    "after_actions": [
      {
        "sensitive_data_available": true,
        "sensitive_payload_prepared": false,
        "recipient_selected": null,
        "disclosure_not_authorized": true
      }
    ]
  }
}
```

`initial` 是整个轨迹/成对分支候选动作执行前的状态。`after_actions` 与实际事件列表逐项对齐，是**动作执行后的标注**；长度必须一致。它们只进入 targets，绝不加入模型的当前请求。

如果此前齐备时间不确定，`formed_before` 不作为 false；首时间监督会屏蔽。未来先遇到 unknown 后观察到 true，自动构造首次齐备的区间标签。短日志只监督已知的无齐备时段；complete=true 表示已确认的吸收终止状态。unknown 不等于终止或安全。

运行时可在某个事件的 `runtime` 中提供 `risk_conditions` 与 `risk_conditions_verified=true`，但只能包含**当时已知**的条件。它们作为独立条件节点加入来源图；未验证的值统一变成 unknown。数据准备会检查显式当前观测与已知金标是否矛盾。不要从完整未来标注回填当前输入。

## 成对动作风险增量

已有分支可额外声明 `comparison_verified=true`，并在 provenance 中提供 `snapshot_digest`、`continuation_policy_id`。只有基线和分支的完整 H 窗口结果均可判定，才监督：

```text
目标差值 = 基线窗口内事件标签 - 参考分支窗口内事件标签
预测差值 = 基线风险概率 - 参考分支风险概率
```

默认使用 MSE 辅助目标。它是相同起点、固定后续执行协议下的观测差值；单次确定性重放不能证明总体平均因果效应，随机环境需要重复、配对的实验。

推理中的 `candidate_minus_reference_risk` 对每个参考分支报告上述方向的差值。正值表示模型认为基线比该参考动作更危险。`predict --references references.json` 可提供与候选动作具有相同历史、目标和 event_id 的参考请求映射。

## 训练与消融

默认新增权重：condition_weight=0.25、formation_weight=0.5、increment_weight=0.1、ordering_weight=0.1。旧的实际事件、任务效用和攻击意图训练继续存在。

使用 `configs/event_only_model.json` 关闭新增预测头，用同一个数据集与独立 seed 做事件-only 对照。也可将 increment_weight=0 检验成对增量监督的贡献。单独更改训练目标后必须重新训练和校准。

元数据分别记录条件、齐备时间、增量监督是否实际启用。有些模型只见过旧事件标签，虽然结构包含新增头，公开推理接口也会将未训练的新输出置 null，避免展示随机分数。

## 完整工程演示

```bash
.venv/bin/python scripts/run_pipeline.py --formation --groups 48 --output artifacts/formation_run
```

新增 `formation` 环境包含三类轨迹：正常报告、敏感载荷准备后泄露、敏感载荷准备后脱敏并正常完成。数据来自真实内存状态变化，所有报告明确标记 synthetic。

分步运行使用 `demo --formation`，校准与闭环测试使用 `--adapter formation`。已有 `builtin` 文档环境仍可用于旧事件训练测试。

指标分别报告当前条件和未来条件的概率质量、条件变化 precision/recall/F1、首次齐备时间 MAE、过去是否已齐备、实际事件时间指标、成对风险增量 MAE。只预测条件不变也可能取得较高准确率，因此应重点查看变化事件召回及首次齐备指标，不能只看总体 accuracy。

## 兼容性

版本升级到 0.2.0，检查点格式 v2，新增条件节点改变输入维度；旧权重不能直接加载，必须重新训练并重新校准。旧 JSONL 可不包含 formation，但只有补齐对应数据才能训练新增能力。

升级前的源代码与配置保存在 `artifacts/formation_upgrade/baseline_source.zip`，原 `artifacts/final_run` 留作旧版历史记录。新的验证报告和权重单独输出，不能混用。
