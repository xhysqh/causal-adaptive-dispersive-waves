# 实验配置索引

每个 YAML 都是一个注册实验协议，配置中的 `output_dir`、源结果路径与冻结文件名
组成可复现性链，故历史 YAML 不移动、不合并。

当前开发配置：

- `m63433524r2431_target_tail_preview.yaml`：R2.4.3 PASS-data/STOP-head 后的
  新鲜 target-tail preview 重构。输入为 `60+8+16=84`；16 维 preview 来自同一条
  candidate-first/frozen-R2.3 四步因果尾分支。数据按 config hash 绑定的 group cache
  可恢复，三分支 hazard 头和四个只读消融只用于 development selection。

- `m63433524r243_policy_reachability.yaml`：R2.4.2 STOP 后的策略条件一致
  可达风险重建。collector policy 只采状态，标签固定为第一步 candidate、随后
  frozen R2.3 controller；输入为 `60+8=68`，输出 K=1/2/4 单调风险。
- `m63433524r242_conditional_risk.yaml`：R2.4.1-A 后的全新 K=4 条件 MoE
  数据与固定 selection 审计，输入布局为 `60+8+4+3=75`；其结果为 STOP，
  仅作为 R2.4.3 的冻结失败归因。

```text
m63433524r241a_readonly_fusion_audit.yaml
```

它登记 R2.4.1 STOP archive 的只读融合/消融诊断，输出路径必须在
`results/_analysis/`。该配置没有 checkpoint、阈值、冻结或外部 LTT 阶段。无论
结果如何，后续 R2.4.2 都必须注册新的数据种子与独立闭环链。

`m63433524r24_onpolicy_margin_certificate.yaml` 登记的 margin-fit、fixed-selection
和 final-LTT 是 R2.4 历史协议；其 STOP 输出保留为失败归因，不应被当作最终部署
配置或证据来源。
