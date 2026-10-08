# 风格回放与盲评

先检查有没有接对人、接对当时的话，再评语气是否像群友。字少、梗多或与真人答案字面一致，都不能单独证明自然。

## 回放

`tools/evaluate_style.py` 只读本地 JSON，使用选定代码的 `build_pack` 生成资料包，不连接服务器或模型。输入为案例列表，每个案例包含：

- `id`：案例名；`rows`：按接收时间排序的 Ambient 原始消息行。
- `current_id`：本次来话的消息 ID；`reference_id`：可选的真实后续答案 ID。
- `excluded_senders`：已确认的其他机器人账号；`limits`：可选参数覆盖。

消息行结构与本插件原话缓存一致，见 `event_message`；不要将不同群的原话混在一个案例。工具按消息位置截到当前行，连同秒但排在后面的消息也留出，避免把真实答案泄漏到样本或近期语境。参考答案不得位于当前行之前。数据不足的案例照实保留，不补造引用或人物关系。

```powershell
python tools/evaluate_style.py replay C:/private/cases.json C:/private/new-packs.json
python tools/evaluate_style.py replay C:/private/cases.json C:/private/old-packs.json --plugin-root C:/reviewed/old-version
```

`--plugin-root` 只用于已经审阅的本地版本。输出含该版本系统规则、逐案资料包和样本/引用/邻近片段数量；这些计数用于检查选样，不是风格分数。比较默认行为时不覆盖 `style_messages`；比较单一排序变量时给两个版本相同的参数。

## 回复对照

在相同 Persona、模型、生成参数及独立新会话下，分别使用两份资料包生成草稿。必要时重复多次，记录失败和不利结果；不要只保留最好的一条。后台测试聊天不会自动经过群插件注入，手动回放资料包只能证明该测试方式下的模型输出，不能代替真实群消息验收。

把草稿保存为以下结构，再打乱版本名称：

```json
[
  {
    "id": "sleep",
    "current": "刚睡醒",
    "reference": "可选：真实群友的后续接法，仅供评阅",
    "variants": {"baseline": "草稿一", "candidate": "草稿二"}
  }
]
```

```powershell
python tools/evaluate_style.py blind C:/private/drafts.json C:/private/review.json --seed 17
```

人工先看 `review.json`，最后再揭开单独的 `review.key.json`。`winner` 留空，工具不会自动选胜者。评阅时分别记录：是否接对对象与话题，是否保留群友常见的省略/短碎句，是否硬加比喻/解释/追问，是否尊重拒绝与严肃请求；允许平局和两条都差。

真实案例、草稿与答案留在仓库之外，发布包只包含合成测试。固定场景用于回归，另留一组未参与调参的场景评估泛化；模型权重未因本插件而训练或改变。
