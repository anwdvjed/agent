# Agent2 v2 准入迁移

本目录保留 `agent1` Python 包名和 `agent1` CLI，项目版本升级至 0.3.0。复制的 `artifacts/final_run` 与 `artifacts/formation_upgrade` 是 0.2 历史输出，仅供离线对照。

| 对象 | 原版本 | 本版 | 迁移动作 |
|---|---|---|---|
| 请求与动作 | `agent1-input-v1` | `agent1-input-v2` | 重新验证原始动作载荷与 JSON；不改变工具/键拼写 |
| 数据集 | `agent1-dataset-v2` | `agent1-dataset-v3` | 重新运行 `prepare`，稳定边 ID 和可选解释标签进入 target |
| 检查点 | `agent1-checkpoint-v2` | `agent1-checkpoint-v3` | 重新训练；新增联合就绪参数，不能以 `strict=False` 加载旧权重 |
| 预测头 | 粗粒度布尔标记 | `agent1-head-contract-v2` | 根据最佳 checkpoint 之前实际训练批次与逐格覆盖生成 |
| 预测输出 | 最终窗口标量为主 | `agent1-forecast-v2` | 输出事件质量、survival、CDF、成员分歧及条件事件时间 |
| 策略 | `threshold` 单窗 | `agent1-temporal-policy-v2` | 显式冻结近期、窗口、告警、效用与 readiness 配置 |
| 校准 | `agent1-policy-calibration-v1` | `agent1-policy-calibration-v2` | 在新选择器与实际修复/执行流程上重跑完整闭环 |
| 许可 | 公开摘要索引 | 随机 `permit_id` | 执行前绑定原始动作字节与实时上下文并一次性消费 |

`bin 1` 为当前尚未执行的候选动作，后续 bin 是动作格，不是秒数或“还能安全执行的步数”。告警与 ALLOW 分离，P0 策略默认 `readiness_warning.mode=off`。开启 `joint_occupancy_v2` 需要相应真实标签训练、逐格覆盖与新的闭环认证。

`configs/policies.json` 是未认证研究示例，只用于演示完整流程。运行时不从文件自哈希推断可信认证：部署方必须提供受控分发验证、活体上下文检查、执行器版本和共享原子账本/受控网关。旧 artifact 即使自行重算公开 digest 也不获准进入新运行时。

迁移顺序：准备真实带来源的标签 → v3 数据集 → v3 独立模型检查点 → 冻结 v2 策略族 → 在未用开发组上通过可信 runner 实际执行每个策略的完整闭环 → 受控分发 artifact → 接入执行前授权与事务边界 → 冻结测试。缺少任一步时保持 HOLD。
