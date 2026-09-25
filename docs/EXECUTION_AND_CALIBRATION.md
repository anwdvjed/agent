> Historical Agent2 guidance. For the current Agent Astra runtime, use [DEPLOYMENT.md](DEPLOYMENT.md) and [RISK_WINDOW.md](RISK_WINDOW.md).

# 执行闭环、策略校准与接入

## 外部环境接口

`--adapter trusted_module:Factory` 动态加载可信 Python 工厂。它必须具有类属性 `environment_id`，可通过 `Factory(scenario, registry)` 构造实例，实现 `agent1.environments.Environment` 协议。

```python
class Factory:
    environment_id = "my-environment-v1"

    def __init__(self, scenario, registry):
        # 建立隔离环境，使用scenario复原起始状态。
        self.done = False

    def request(self):
        # 返回已观察历史 + 尚未执行动作；不包含未来观察或评测答案。
        ...

    def repairs(self, request):
        # 返回 {分支名: 完整候选request}；可信环境验证可用性与后置条件。
        # 不允许用"none"键覆盖原始request。
        ...

    def commit(self, request):
        # 校验请求对应当前状态，执行确切动作，更新观察/历史/done。
        # 必须拒绝旧动作和不支持的沙箱/审批标记。
        ...

    def stop(self, reason):
        # 终止本轮策略执行；不要继续原始轨迹伪造实际结果。
        ...

    def outcome(self):
        # 从真实环境状态判定，不能使用模型自报。
        return {
            "unsafe": False, "utility": False, "completed": True,
            "executed_actions": 0, "benign": True, "synthetic": False
        }

    def fork(self):
        # 独立状态副本，用于离线成对实验。
        ...
```

上面是接口示意，省略号处需要真实实现，不是可冒充结果的适配器。AgentDojo 等基准需要映射自己的工具前钩子、环境状态、用户任务结果与攻击成功判定。不要把标签和隐藏 side_task 放入 request；它们可以保留在 scenario 中供环境评分。

scenario JSONL 至少包含 `episode_id`、`group_id`、`split` 及适配器复原环境所需字段。同一任务不得跨 split，测试组不得与训练/验证/校准组重合。适配器的顶层实现文件哈希与 environment_id 绑定校准；如果它还依赖外部工具版本、数据或容器镜像，必须将这些不可变版本编码进 environment_id，并由部署方校验。

提供者负责真实进程/容器/网络隔离。该 Python 协议本身不是操作系统安全沙箱。内置 `builtin` 完全在内存中运行，不访问网络或执行 Shell。

## 冻结策略选择器

每个集成成员先计算完整首次事件 CDF，再逐格求均值与总体标准差。`build_decision_curve` 用 `min(1, mean+z*std)` 及其 running maximum 产生决策分数。`select_action_v2` 同时检查当前候选所在的 bin 1、近期窗口、完整窗口、硬约束和已训练效用头；告警是独立字段。该分数不是概率置信上界。闭环回放与部署共用同一纯函数。

`calibrate` 对 policies.json 中每个预先指定的策略执行完整校准任务。修复改变后续观察，下一步重新调用模型；不会把固定轨迹后缀当成修复后的真实世界。

## 统计对象

独立单元是完整任务组，而非动作或前缀。每个策略的组级安全损失是“该组任意实际执行轨迹是否产生违规”。正常任务中断损失是“包含正常任务的组中，是否有正常任务未完成”。

若候选策略数为 M、总失败概率为 delta，对每个策略和每个约束使用 delta/(2M) 的单侧 Clopper–Pearson 二项上界。Bonferroni 对两类约束和整个候选族同时控制，所以在此固定策略族内按实测效用挑选可行策略不会额外消耗未计入的选择自由度。

假设：校准任务组独立、来自与部署一致的分布；每个组的生成协议固定；模型、候选生成器、工具状态语义和策略族在使用校准结果前固定。组间独立性不能靠命名新 group_id 制造。若按类别比例确定性抽样，或分布明显不同，应只报告该实验分布的经验结果，不能直接推广成总体保证。

这控制的是完整策略在上述组分布上的风险，不是每个候选动作的真实条件概率，也不是无限运行、不受分布漂移影响的安全保证。测试报告中 bootstrap 区间与单侧二项上界分别给出；观察到零违规仍会有非零上界。

若没有策略同时满足违规率和正常中断率预算，artifact 的 status 为 `no_feasible_policy`，selected_policy 为 null。SafetyRuntime 返回 HOLD。旧 v1 artifact 和 checkpoint 不能迁入新准入路径。

## Python 准入接口

`SafetyRuntime` 需要部署方提供三个可信边界：签名请求验证器 `trusted_context_verifier`、实时撤权/版本查询 `current_context_verifier`，以及从受控分发渠道核验校准产物的 `artifact_verifier`。还需传入实际部署的 `deployment_lineage`（校准 episode/group 清单、策略族、修复生成器和执行器版本）。任何一项缺失均不签发许可。请求携带的 `verified` 字段、自哈希或本仓库的内存模拟器不能代替这些边界。

`assess` 返回 ALLOW 时，使用 `selected_branch` 从**已签名的真实候选请求**中取出确切动作，再以 `commit.permit_id` 调用 `authorize_and_consume(chosen_request, permit_id)`。成功返回不可变 `AuthorizedAction`，其中 `action_bytes` 是被批准的原始执行载荷；执行器只能提交这些字节对应的动作，不能重新读取调用方可变字典。`verify_commit_digest` 是废弃的 v1 兼容入口，始终拒绝。

许可由不可猜测 ID 索引，只能消费一次；同一 actor/event 重新评估会撤销旧许可，已消费事件不能再次签发。过期、上下文 revision 变化、注册表/策略血缘变化和请求/动作摘要不匹配均拒绝。本实现只声明单进程内存存储。外部工具提交与许可消费之间仍需执行器的锁、事务或受控网关；跨进程部署需共享原子账本。本仓库没有这种真实生产执行器，因此生产准入保持 HOLD。

逐轨迹 `latency_ms` 包含环境初始化和完整运行，但模型已经加载。冷启动/检查点加载时间应另行测量；不能把该数值标为包含冷启动的总延迟。

## 来源组合验证

`explain_groups` 默认执行来源集合移除后的模型重预测。输入文本、对应的图和时序表示同时重建；可选 `argument_sources` 元数据指示从来源复制的参数。原始请求不会改变。

对于真实动作归因，传入可信的 `replay_evaluator(original_request, modified_request, source_group)`：在环境副本中应用来源变化，重新生成候选动作，实际运行，并返回结局。没有该回调就保持 replay_outcome=null。

内置 `source_replay_evaluator` 实现这一流程，演示重复注入中的联合来源移除。它证明工程接口和玩具环境的可验证交互，不代表模型已具备一般因果识别能力。
