# 研究定位与可证伪实验

Agent1 的候选研究问题：在相同正常任务误拦截预算和验证成本下，来源图能否筛选需要联合处理的风险来源，并改善干预后的多步风险与任务效用预测？当前实现提供实验基础，不声称已证明新颖性或优越性。

## 对应文献

- [AgentSpec，ICSE 2026](https://arxiv.org/abs/2503.18666)：可执行规范与运行时强制约束。
- [MELON，ICML 2025](https://proceedings.mlr.press/v267/zhu25z.html)：遮蔽任务后的反事实重执行。
- [AttriGuard，USENIX Security 2026](https://www.usenix.org/conference/usenixsecurity26/presentation/he-yu)：动作来源归因、对齐历史的重执行。
- [SafeMCP，ACL 2026](https://aclanthology.org/2026.acl-long.522/)：环境动态与前瞻能力约束。
- [CausalArmor，2026 预印本](https://arxiv.org/abs/2602.07918)：来源片段归因与选择性清理，讨论组合来源干预。
- [CORA，2026 预印本](https://arxiv.org/abs/2604.09155)：执行前风险门控与统计风险控制。
- [Recast，2026 预印本](https://arxiv.org/abs/2607.26820)：预测首次风险出现的时间分布。
- [Monitoring Web Agents Without Internal Signals，2026 预印本](https://arxiv.org/abs/2609.02057)：关键错误边界，区分最终失败与有效早期前缀。
- [Conformal Risk Control，ICLR 2024](https://arxiv.org/abs/2208.02814)：对明确定义损失的风险控制。Agent1 当前实现使用有限策略族的精确二项检验与同时控制，不冒充实现该文的所有算法。
- [The Attacker Moves Second，USENIX Security 2026](https://www.usenix.org/conference/usenixsecurity26/presentation/nasr)：必须评估针对防御机制适应的攻击。

## 先建立四组证据

1. 时间标签实验：真实首事件金标 vs 末尾弱标签基线；当前动作分类、长度基线、temporal_only、graph_only、full；固定正常轨迹误报率报告提交前拦截与提前一步预警。若优势在精确标签上消失，不能主张时序预警有效。
2. 来源实验：单来源移除、同预算随机分组、图引导组合移除；比较模型敏感性、真实动作变化、真实任务结局。图方法若没有优于同预算对照，不能把图结构宣称为主要贡献。
3. 干预实验：无防御、始终阻断、固定脱敏、学习选择干预；报告任务完成、安全违规、修复成本和干预后的残余风险。保留任务是评价的一部分，参数变成占位符不等于修复成功。
4. 在线实验：先接 AgentDojo，再测试 AgentDyn 的合法外部指令以及 AgentLAB 的长程/自适应攻击。阈值、模型、攻击预算和确认策略冻结后才运行 test；不能直接用不同论文的数值横比。

## 已实现与待采集

已实现：动作时间监督、全部已标注分支的训练、概率/策略指标分离、完整模拟执行、有限策略族校准、图引导联合来源移除、模拟动作重生成、分组测试和实现溯源。

待由真实研究数据提供：人工动作级金标、真实工具环境的成对分支结果、真实 AgentDojo/AgentDyn/AgentLAB 适配器、跨攻击/跨工具/跨域实验和充分独立的校准组。工程测试中的合成 oracle 只用于验证代码，不能充当真实基线或正式安全结果。
