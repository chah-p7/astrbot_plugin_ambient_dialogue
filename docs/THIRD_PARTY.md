# Tulpa

Ambient Dialogue 1.3.0 的群聊引用核验、同人同场景优先排序和具体原话优先的提示方式参考并改写自 [fumingyang2004/Tulpa](https://github.com/fumingyang2004/Tulpa)，固定参考提交为 `5476cbfa58986b0dbc5b200f8f69c3239b1cf637`。

参考文件：`chatlocal/reply_history.py`（引用来源校验与排序）、`chatlocal/tulpa.py`（本地场景信号）、`chatlocal/reply_agent.py` 和 `chatlocal/prompts/mcp_chat/behavior.md`（真实互动案例、口吻与参与边界）。

本插件重新实现为每群有界短期窗口上的本地计算，不依赖 Tulpa 或新增模型。群友的真人表达用于口吻参考，机器人自己生成的回复始终排除在学习样本之外；窗口外引用不会被扩大检索到长期档案。普通回复和插话共享这套选择逻辑。

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
