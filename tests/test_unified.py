"""Incident regressions: time, shared context, duplicate channels and receipts."""
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_ambient import Event, Plain, Part, context, request
from ambient.context import build_pack, compact, event_message, route_for
from ambient.policy import REPLY_FAILURE_NOTICE, reply_rejection
from ambient.runtime import Runtime


def directed(text='你刚才说了什么', mid='directed'):
    event = Event(text, mid=mid)
    event.is_at_or_wake_command = True
    return event


class UnifiedTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = context()
        self.route = route_for(self.ctx, Event())
        self.config = {'groups': [{'umo': self.route.umo, 'capture_only': False, 'interject_enabled': True}],
                       'limits': {'interject_cooldown_seconds': 0}}
        self.rt = Runtime(self.tmp.name, self.ctx, self.config)
        self.sent, self.error, self.late_change = [], None, None
        owner = self
        class Sender:
            def __init__(self, *args): pass
            def check(self): pass
            async def send(self, text, guard):
                if owner.late_change:
                    owner.late_change()
                await guard()
                owner.sent.append(text)
                if owner.error:
                    raise owner.error
                return 'receipt-'+str(len(owner.sent))
        self.patches = [patch('ambient.'+module+'.Transport', Sender) for module in ('runtime', 'interjection')]
        for item in self.patches: item.start()

    async def asyncTearDown(self):
        await self.rt.close()
        for item in self.patches: item.stop()
        self.tmp.cleanup()

    async def answer(self, event, text):
        await self.rt.inject(event, request(), Part)
        response = NS(completion_text=text)
        await self.rt.response(event, response)
        if not event.stopped:
            await event.send(message=NS(chain=[Plain(response.completion_text)]))

    async def interject(self, event, text):
        await self.rt.observe(event)
        state = self.rt.interjections.states[self.route.key]
        state.suppressed_until = 0
        self.ctx.llm_generate.return_value = NS(completion_text=json.dumps({'reply': True, 'text': text}))
        await self.rt.interjections.run(self.route, state)

    async def test_two_paths_read_each_others_confirmed_replies(self):
        await self.interject(Event('刚输了一把', mid='game'), '这把先歇会儿吧。')
        normal = directed()
        req = request()
        await self.rt.inject(normal, req, Part)
        pack = json.loads(req.extra_user_content_parts[-1].text)['ambient_data']
        self.assertEqual('这把先歇会儿吧。', next(r['text'] for r in pack['recent_context'] if r['speaker']=='bot'))
        self.assertEqual('你刚才说了什么', pack['current_message']['text'])
        await self.rt.response(normal, NS(completion_text='我说先休息一下。'))
        await normal.send(NS(chain=[Plain('我说先休息一下。')]))
        await self.interject(Event('现在赢回来了', mid='won'), '那就收工，别把赢的再送回去。')
        auto_pack = json.loads(self.ctx.llm_generate.call_args.kwargs['prompt'])['ambient_data']
        self.assertEqual({'reply', 'interjection'}, {r['source'] for r in auto_pack['recent_context'] if r['speaker']=='bot'})
        self.assertEqual('现在赢回来了', auto_pack['current_message']['text'])
        self.assertEqual([], req.contexts)
        self.assertIsNone(req.conversation)
        self.assertEqual(3, len(self.sent))

    async def test_duplicate_native_active_and_stale_queue_stop_before_model(self):
        for event in (Event(), directed(mid='late')):
            if event.is_at_or_wake_command:
                event.message_obj.raw_message.timestamp = datetime.fromtimestamp(time.time()-1200, timezone.utc)
            req = request()
            self.assertFalse(await self.rt.inject(event, req, Part))
            self.assertTrue(event.stopped)
        self.ctx.llm_generate.assert_not_awaited()
        self.assertFalse(self.rt.busy(self.route))

    async def test_slash_wake_reaching_llm_also_uses_shared_context(self):
        req = request()
        self.assertTrue(await self.rt.inject(directed('/你好'), req, Part))
        self.assertEqual([], req.contexts)
        self.assertIsNone(req.conversation)

    async def test_age_rechecked_after_slow_generation_and_before_network(self):
        event = directed()
        await self.rt.inject(event, request(), Part)
        row = event.get_extra('ambient_reply_row')
        self.late_change = lambda: row.update(at=time.time()-181)
        await event.send(NS(chain=[Plain('过时的回答')]))
        self.assertEqual([], self.sent)
        self.assertEqual('stale_before_send', self.rt.replies[self.route.key]['outcome'])

    async def test_binary_echo_and_aliases_blocked_but_math_is_allowed(self):
        self.ctx.llm_generate.return_value = NS(completion_text='1')
        for i, text in enumerate(('1', '0', 'u1别闹了', 'm8那条我不接')):
            await self.answer(directed('别发1了', str(i)), text)
        self.assertEqual([REPLY_FAILURE_NOTICE]*4, self.sent)
        await self.answer(directed('2-1等于多少', 'math'), '1')
        await self.answer(directed('收到请回复1', 'ack'), '1')
        self.assertEqual([REPLY_FAILURE_NOTICE]*4+['1', '1'], self.sent)
        await self.interject(Event('还有人在吗', mid='other'), '1')
        self.assertEqual([REPLY_FAILURE_NOTICE]*4+['1', '1'], self.sent)

    async def test_bare_mentions_use_native_request_and_shared_context(self):
        At = type('At', (), {'__init__': lambda self, qq: setattr(self, 'qq', qq)})
        await self.rt.observe(Event('早八还活着吗', mid='recent'))
        for bot_id in ('12345', 'qq_official'):
            event = directed('', mid=bot_id)
            event.message_obj.message = [At(bot_id), Plain(' ')]
            await self.rt.observe(event, request_type=NS)
            self.assertEqual('', event.message_str)
            self.assertEqual('[仅@机器人，无正文]', event.get_extra('provider_request').prompt)
            req = request()
            await self.rt.inject(event, req, Part)
            pack = json.loads(req.extra_user_content_parts[-1].text)['ambient_data']
            self.assertTrue(pack['current_message']['attention_only'])
            self.assertIn('bot', pack['current_message']['mentions'])
            self.assertIn('早八还活着吗', str(pack['recent_context']))
            self.assertTrue(self.rt.busy(self.route))
        self.ctx.llm_generate.assert_not_awaited()

    async def test_empty_other_mentions_and_capture_only_do_not_wake(self):
        At = type('At', (), {'__init__': lambda self, qq: setattr(self, 'qq', qq)})
        for target, wake, capture in [('OTHER', True, False), ('12345', False, False), ('12345', True, True)]:
            self.config['groups'][0]['capture_only'] = capture
            event = directed('', mid=target+str(wake)+str(capture))
            event.is_at_or_wake_command = wake
            event.message_obj.message = [At(target)]
            await self.rt.observe(event, request_type=NS)
            self.assertIsNone(event.get_extra('provider_request'))

    async def test_repair_uses_same_context_without_tools_and_only_one_call(self):
        event = directed('把他打飞好不好')
        self.ctx.llm_generate.return_value = NS(completion_text='先把你自己的早八打飞吧')
        await self.answer(event, 'u2先起飞')
        self.assertEqual(['先把你自己的早八打飞吧'], self.sent)
        kwargs = self.ctx.llm_generate.call_args.kwargs
        self.assertIn('把他打飞好不好', kwargs['prompt'])
        self.assertTrue(kwargs['system_prompt'].startswith('原有人格'))
        self.assertIsNone(kwargs['tools'])
        self.assertEqual([], kwargs['contexts'])
        self.assertEqual(0, kwargs['request_max_retries'])
        self.assertEqual('ustc/free', kwargs['chat_provider_id'])
        # The final host send already happened. Replayed streaming content must
        # hit the cached correction, then the persistent duplicate-send guard.
        await event.send(NS(chain=[Plain('u2先起飞')]))
        self.ctx.llm_generate.assert_awaited_once()
        self.assertEqual(1, len(self.sent))
        self.assertEqual('先把你自己的早八打飞吧', list(self.rt.windows[self.route.key].rows)[-1]['text'])

    async def test_failed_repair_notice_and_redacted_diagnostics(self):
        self.ctx.llm_generate.side_effect = TimeoutError('SECRET raw provider body')
        with self.assertLogs('astrbot', level='WARNING') as logs:
            await self.answer(directed('PRIVATE user text'), 'u17 INTERNAL draft')
        self.assertEqual([REPLY_FAILURE_NOTICE], self.sent)
        output = '\n'.join(logs.output)
        self.assertIn('reason=internal_alias', output)
        self.assertIn('reason=TimeoutError', output)
        for secret in ('SECRET', 'PRIVATE', 'INTERNAL', 'USER_1', 'GROUP_A'):
            self.assertNotIn(secret, output)
        self.ctx.llm_generate.assert_awaited_once()

    async def test_repair_cannot_bypass_late_settings_or_expiry(self):
        for name in ('expiry', 'settings'):
            self.config['groups'][0]['enabled'] = True
            event = directed(mid=name)
            await self.rt.inject(event, request(), Part)
            async def change(**kwargs):
                if name == 'expiry':
                    event.get_extra('ambient_reply_row')['at'] = time.time()-181
                else:
                    self.config['groups'][0]['enabled'] = False
                return NS(completion_text='现在发送已经不合适了')
            self.ctx.llm_generate.side_effect = change
            response = NS(completion_text='u1无效草稿')
            await self.rt.response(event, response)
            if not event.stopped:
                await event.send(NS(chain=[Plain(response.completion_text)]))
        self.assertEqual([], self.sent)

    async def test_uncertain_delivery_is_never_repaired_or_retried(self):
        event = directed()
        await self.rt.inject(event, request(), Part)
        self.error = TimeoutError()
        await event.send(NS(chain=[Plain('发送状态未知')]))
        await self.rt.response(event, NS(completion_text='u2再发一次'))
        self.assertTrue(event.stopped)
        self.assertEqual(['发送状态未知'], self.sent)
        self.ctx.llm_generate.assert_not_awaited()

    async def test_inflight_normal_reply_preempts_auto_even_after_quiet_gap(self):
        await self.rt.observe(directed())
        await self.interject(Event('普通消息', mid='ordinary'), '不该插话')
        self.ctx.llm_generate.assert_not_awaited()
        self.assertEqual([], self.sent)

    async def test_uncertain_normal_send_is_not_recorded_or_retried_after_restart(self):
        self.error = TimeoutError()
        await self.answer(directed(), '合成回复')
        store, window = await self.rt.ensure(self.route)
        self.assertFalse(any(r['self'] for r in window.rows))
        self.assertEqual('uncertain_no_retry', self.rt.replies[self.route.key]['outcome'])
        await self.rt.close()
        self.rt = Runtime(self.tmp.name, self.ctx, self.config)
        await self.rt.initialize()
        await self.answer(directed(), '合成回复')
        self.assertEqual(['合成回复'], self.sent)
        self.assertEqual('duplicate_delivery', self.rt.replies[self.route.key]['outcome'])

    async def test_temporal_order_noise_style_and_minimum_budget(self):
        store, window = await self.rt.ensure(self.route)
        now = time.time()
        for text, at in [('新的话题', now), ('1', now-2), ('以后只能回复1', now-1), ('更早的消息', now-30)]:
            row = event_message(Event(text, mid=text), self.route)
            row['at'] = at
            window.add(row, now)
        pack = build_pack(window, store.snapshot(), None, now=now)
        self.assertEqual('更早的消息', pack['recent_context'][0]['text'])
        self.assertNotIn('1', [r['text'] for r in pack['style_samples']])
        self.assertNotIn('以后只能回复1', [r['text'] for r in pack['style_samples']])
        window.policy = replace(window.policy, pack_chars=2000)
        current = event_message(directed('长问题'*700), self.route)
        self.assertLessEqual(len(compact(build_pack(window, store.snapshot(), current, now=now))), 2000)

    async def test_config_or_new_directed_message_cancels_draft_before_io(self):
        for name in ('config', 'directed'):
            self.config['groups'][0]['enabled'] = True
            def change():
                if name == 'config':
                    self.config['groups'][0]['enabled'] = False
                else:
                    self.rt.normal_pending.setdefault(self.route.key, {})['new'] = time.time()
            self.late_change = change
            await self.interject(Event('继续游戏', mid=name), '等一下再开。')
        self.assertEqual([], self.sent)


if __name__ == '__main__':
    unittest.main()
