# Agent2 实施报告

日期：2026-09-23。来源：同级 `agent1` 项目；本文件夹为独立的可安装工程副本。`CODEX_START_PROMPT.txt` 的执行顺序和 `CODEX_REFACTOR_SPEC.md` 的 T01–T10 是本轮实施依据，附件中的文件名后缀未进入 Python 导入。包名仍为 `agent1`，项目版本 0.3.0。

## 任务与实际改动

| 任务 | 实现文件 | 结果 |
|---|---|---|
| T01 时间契约 | `policy.py`, `tests/test_policy_v2.py`, `tests/test_v2_contracts.py`；保留原 `labels.py` | 保留 `event_step-step+1`；bin 1 为当前未执行候选；仅输出动作格和条件事件时间，不输出安全剩余步数。 |
| T02 完整风险曲线 | `runtime.py`, `policy.py` | 先按集成成员求首次事件 CDF，再求逐格均值与总体标准差；公开 event mass、survival、CDF、类型累计发生率（仅训练可用时）和条件事件格；异常值在评分前拒绝。 |
| T03 v2 策略与告警 | `policy.py`, `runtime.py`, `replay.py`, `configs/policies.json` | 冻结近端/全窗阈值、running-max 分数、效用与独立 alert；共享纯选择器供回放和运行时使用；readiness 默认 off。 |
| T04 逐头监督 | `formation.py`, `objective.py`, `engine.py`, `runtime.py` | 最佳 checkpoint 对应训练批次的有效标签数、更新步数、格点与条件覆盖写入 `agent1-head-contract-v2`；未训练的辅助输出为 null，需要该头的策略 HOLD。 |
| T05 校准与血缘 | `calibration.py`, `cli.py`, `runtime.py` | v2 artifact，冻结策略族、训练/校准组隔离、完整相同 episode 清单、计数重算、实现/执行器/修复生成器血缘；v1 artifact 被拒绝。 |
| T06 执行许可 | `data.py`, `io.py`, `runtime.py` | 原始动作的严格 JSON/Unicode 检查与独立 digest；随机 permit ID，单进程内一次消费、重评估撤销、过期与上下文 revision 校验；无真实验证服务时 HOLD。 |
| T07 可逆联合就绪 | `formation.py`, `objective.py`, `engine.py`, `runtime.py`, `policy.py` | current/future 直接联合头及独立掩码监督；历史首次形成分布保留且不触发永久告警；只有训练并重新认证的策略可启用就绪告警。 |
| T08 实际训练损失 | `objective.py`, `engine.py`, `dataset.py` | `research_loss` 仍保留 exact/interval/censored 首次事件似然；full 分支也有真实 NLL，充分性比较 H×K 累计发生率并 detach 参考；概念/边金标进入数据、batch 和实际训练入口；无监督占位分支不施加稀疏损失。 |
| T09 图证据与脱敏 | `graph.py`, `evidence.py`, `data.py` | 完整受限结构生成节点身份摘要、稳定 edge ID、最终敏感标记覆盖旧 preview；公开来源 ID 脱敏；解释报告区分事实、预测、结构边和模型敏感性。未经可信谱系的 `argument_sources` 不再修改实验动作。 |
| T10 回归与交付 | `tests/`, `README.md`, `docs/`, `pyproject.toml`, 本报告 | 版本迁移说明、单元/集成测试、完整合成流水线及 wheel 构建。 |

P2 的显式未来路径模拟未实现。当前模型预测的是学习到的续行分布下首次不安全事件，不能把预测危险格解释为在该格前可安全继续执行。下一实际动作必须重新评估。

## 契约与迁移

预测接口为 `agent1-forecast-v2`，关键字段包括 `branch_id`、`candidate_event_id`、`horizon_bins`、`time_unit=action_bin`、`bin_one_semantics=current_candidate`、`event_mass_mean`、`survival_mean`、`event_cdf_mean/std`、`cause_names`、`head_availability`、`prepared_digest`、`expected_event_bin_conditional`。保留 `risk_mean` 与 `ensemble_disagreement`，分别等于末格 CDF 均值/标准差；无可信 cause 训练时类型累计发生率为 null。概率守恒容差为 `1e-6`；条件期望分母门槛为 `1e-8`。

策略为 `agent1-temporal-policy-v2`，`score_method=cdf_member_mean_std_running_max_v1`。raw score 是 `min(1, mean+z*std)`，decision score 是其逐格运行最大值；两者均保留在 `branch_checks`，不称为概率上界。概率门槛等于阈值时允许，超过才拒绝；告警大于等于其阈值触发。仅 `decision=ALLOW` 可以得到 permit。

数据集、checkpoint、校准分别升至 `agent1-dataset-v3`、`agent1-checkpoint-v3`、`agent1-policy-calibration-v2`。旧数据要重新 prepare，旧模型要严格重新训练，旧策略要在相同受信组上重跑完整闭环；不能静默补近端阈值或 `strict=False` 加载旧权重。详细迁移见 `docs/MIGRATION_V2.md`。`artifacts/` 中的复制品都是历史 v1 输出，不用于 v2 认证。

许可记录绑定确切原始动作字节、准备后请求摘要、候选 event/actor、上下文 revision、策略 artifact digest 和部署血缘。`authorize_and_consume` 最多返回一次 `AuthorizedAction`；公开 SHA-256 仅用于一致性检查，不是身份认证或消费索引。运行时默认没有 `artifact_verifier` 与 `current_context_verifier` 时 HOLD。内存锁仅支持单进程；外部副作用的原子提交须由真实执行器实现。

## 实际验证

在 `agent2` 工作目录执行：

```text
../agent1/.venv/bin/python -m pytest -q --tb=short
77 passed

../agent1/.venv/bin/python scripts/run_pipeline.py --groups 16 --output /tmp/agent2-final2-smoke-20260923
完成：合成数据准备、两 seed 各 2 轮训练、验证、v2 校准、测试、闭环回放；校准 no_feasible_policy，测试回放执行动作 0。

../agent1/.venv/bin/python scripts/run_pipeline.py --formation --groups 16 --output /tmp/agent2-final2-formation-smoke-20260923
完成：formation 合成数据全流程；两 seed 的 current/future joint-ready、formation hazard、future conditions 均在最佳 checkpoint 标为有训练覆盖；校准 no_feasible_policy。

../agent1/.venv/bin/python -m pip wheel . --no-deps --no-build-isolation --no-cache-dir -w /tmp/agent2-final-wheels-20260923
成功构建 agent1-0.3.0-py3-none-any.whl。
```

测试中使用的固定概率、假 ensemble、假 artifact verifier 和模拟环境都只存在于 `tests/` 或明确标记为 synthetic 的演示路径；它们不证明真实预测精度或部署安全。上述流水线的训练是**合成数据重训**，不是实际数据重训。

## 准入与剩余依赖

- 实际数据重训：**否**。本轮只运行了明确标为 synthetic 的普通/formation 工程演示。
- 真实完整闭环校准认证：**否**。合成校准均为 `no_feasible_policy`，无可用生产策略。
- 当前生产准入：**HOLD**。缺真实数据、可信闭环 runner、artifact 受控分发验证、实时撤权服务、生产执行器和分布式原子 permit/事件账本。
- 解释支持稳定结构边、可信当前条件（若独立验证）、模型预测和有限预算敏感性实验；不证明因果数据流或事故原因。未经验证的派生参数谱系未建立；`verified_replay` 默认为 null。删除边仍保留节点、时序和全图池化信息。
- `necessity_weight` 继续禁用；提供正权重会报错，直到有真实可移除证据标签和定义清楚的删除实验。未做 old/full/temporal-only/graph-only 的真实数据消融或效果评估。
- 外部工具提交与本地许可消费之间的竞态、进程重启后的存储丢失和多进程并发，需要生产执行器/共享事务解决；本仓库不声称跨服务原子执行。人工确认、沙箱、审批和回放都不能由请求自报布尔值生成。
