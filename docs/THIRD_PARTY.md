# Tulpa

Ambient Dialogue 1.3.0 的群聊引用核验、同人同场景优先排序和具体原话优先的提示方式参考并改写自 [fumingyang2004/Tulpa](https://github.com/fumingyang2004/Tulpa)，固定参考提交为 `5476cbfa58986b0dbc5b200f8f69c3239b1cf637`。

参考文件：`chatlocal/reply_history.py`（引用来源校验与排序）、`chatlocal/tulpa.py`（本地场景信号）、`chatlocal/reply_agent.py` 和 `chatlocal/prompts/mcp_chat/behavior.md`（真实互动案例、口吻与参与边界）。

本插件重新实现为每群有界短期窗口上的本地计算，不依赖 Tulpa 或新增模型。群友的真人表达用于口吻参考，机器人自己生成的回复始终排除在学习样本之外；窗口外引用不会被扩大检索到长期档案。普通回复和插话共享这套选择逻辑。

1.3.3 另对照了以下固定版本，仅参考设计和评测方法，未复制其实现代码或引入依赖：

- [Tulpa 的风格评测](https://github.com/fumingyang2004/Tulpa/blob/079ce9e8ba0a59bded346f61116ba118fc83eaf7/scripts/eval_mcp_chat_style.py)：固定场景比较回复；本插件增加消息位置截断及独立留出答案。
- [MaiBot 的情境表达学习](https://github.com/Mai-with-u/MaiBot/blob/f9ec3670a3638d6a4708ee5253f7ffe2838c9415/src/learners/expression_learner.py)及[离线选择评测](https://github.com/Mai-with-u/MaiBot/blob/f9ec3670a3638d6a4708ee5253f7ffe2838c9415/scripts/expression_selection/README.md)：结合情境与来源比较表达，保留人工评阅。本插件以短期原话片段独立实现，不使用表达数据库、向量模型或自动人设学习。

本次调整将场景和话题放在身份之前；相邻原话明确标为时间线线索，引用关系仍须独立核验。持久样本池、对机器人回复的自我模仿、自动给沉默打分均未采用。

以下保留原项目 MIT 许可：

```
MIT License

Copyright (c) 2026 Tulpa contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
