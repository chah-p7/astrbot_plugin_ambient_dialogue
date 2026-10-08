"""Segmentation quality and actual partial-delivery behavior, without QQ posts."""
import asyncio
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ambient.delivery import send_reply_parts, split_reply
from ambient.policy import Policy


class SplitTests(unittest.TestCase):
    def test_short_reply_and_paragraphs(self):
        self.assertEqual(['健身环先练负重了'], split_reply('健身环先练负重了'))
        self.assertEqual(['第一段先说结论。', '第二段把原因讲清。'],
                         split_reply('第一段先说结论。\n\n第二段把原因讲清。'))

    def test_long_prose_preserves_content_and_complete_sentences(self):
        sentences = ['这是一段需要详细说明的内容，请先保存当前记录。',
                     '接着查看最新日志，确认出错的时间和当时的操作。',
                     '最后核对更改是否生效，不要反复提交同一项任务。']
        text = ''.join(sentences)
        self.assertEqual(sentences, split_reply(text, 35))
        clauses = '这是一条很长很长的消息，没有必要一股脑全塞进去，可以先把前半句讲完，再把后面的事情接着说清楚。'
        parts = split_reply(clauses, 20)
        self.assertEqual(clauses, ''.join(parts))
        self.assertTrue(all(len(p) <= 20 for p in parts))

    def test_code_urls_math_quotes_and_english_words_are_indivisible(self):
        atoms = ['```python\nprint("a.b,c")\n\nprint("你好")\n```',
                 'https://example.org/a-long-path?q=what&n=3.14',
                 '[参考说明](https://example.org/a?a=1)',
                 '“这是完整引用，不能把半句挪到另一条里面。”',
                 r'\(x^2 + y^2 = z^2\)', '$$a+b=c$$', 'supercalifragilisticexpialidocious']
        for atom in atoms:
            with self.subTest(atom=atom):
                parts = split_reply('这里是说明。\n\n'+atom+'\n\n以上是原文。', 12)
                self.assertIn(atom, parts)
        self.assertEqual(['未完代码：', '```py\nx = 1\n\ny = 2'],
                         split_reply('未完代码：\n\n```py\nx = 1\n\ny = 2', 8))

    def test_no_punctuation_and_disabled_setting_lose_nothing(self):
        text = '很长的中文句子没有标点符号'*20
        self.assertEqual(text, ''.join(split_reply(text)))
        self.assertTrue(all(len(p) <= 80 for p in split_reply(text)))
        self.assertEqual(['甲\n\n乙'], split_reply('甲\n\n乙', 0))
        policy = Policy.from_config({'limits': {'reply_segment_chars': 0, 'reply_segment_interval_ms': 0}})
        self.assertEqual(0, policy.reply_segment_chars)
        self.assertEqual(0, policy.reply_segment_interval_ms)


class DeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_platform_allowance_merges_tail_without_dropping_content(self):
        seen, receipts = [], []
        class Sender:
            async def send(self, text, guard):
                await guard()
                seen.append(text)
                return 'receipt-'+str(len(seen))
        async def confirmed(text, receipt):
            receipts.append((text, receipt))
        text = '\n\n'.join('第'+str(i)+'段' for i in range(8))
        await send_reply_parts(Sender(), text, replace(Policy(), reply_segment_interval_ms=0),
                               AsyncMock(), confirmed, max_parts=3)
        self.assertEqual(3, len(seen))
        self.assertEqual(text, '\n\n'.join(seen))
        self.assertEqual(3, len(receipts))
        with self.assertRaisesRegex(ValueError, 'qq_reply_slots_exhausted'):
            await send_reply_parts(Sender(), text, Policy(), AsyncMock(), confirmed, max_parts=0)
        self.assertEqual(3, len(seen))

    async def test_part_is_confirmed_before_next_and_failure_stops_tail(self):
        events = []
        class Sender:
            async def send(self, text, guard):
                await guard()
                events.append('send:'+text)
                if text == '第二段':
                    raise TimeoutError()
                return 'receipt'
        async def confirmed(text, receipt):
            events.append('confirmed:'+text)
        with self.assertRaises(TimeoutError):
            await send_reply_parts(Sender(), '第一段\n\n第二段\n\n第三段',
                replace(Policy(), reply_segment_interval_ms=0), AsyncMock(), confirmed)
        self.assertEqual(['send:第一段', 'confirmed:第一段', 'send:第二段'], events)

    async def test_cancellation_during_gap_does_not_send_next(self):
        delivered = asyncio.Event()
        sender = type('Sender', (), {'send': AsyncMock(return_value='receipt')})()
        async def confirmed(text, receipt):
            delivered.set()
        task = asyncio.create_task(send_reply_parts(sender, '第一段\n\n第二段',
            replace(Policy(), reply_segment_interval_ms=3000), AsyncMock(), confirmed))
        await delivered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        sender.send.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
