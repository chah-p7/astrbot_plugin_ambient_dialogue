"""Real AstrBot/qq-botpy contracts, intercepted at the native HTTP boundary."""
from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ambient.runtime import Runtime
from ambient.transport import Transport
from ambient.context import route_for
from ambient.host import route_send_tool
from test_ambient import Event, context


@unittest.skipUnless(os.environ.get('AMBIENT_NATIVE_TEST'), 'requires installed AstrBot 4.26.7 and native SDK')
class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_proactive_permission_rejection_is_reported_without_passive_retry(self):
        self.status, self.response = 400, {'code':40034105,'message':'private server detail'}
        sender = Transport(self.ctx,None,self.route,proactive=True)
        with self.assertRaisesRegex(ValueError,'qq_proactive_permission_required'):
            await sender.send('scheduled share',AsyncMock())
        self.assertEqual(1,len(self.calls))
        self.assertTrue({'msg_id','msg_seq','event_id'}.isdisjoint(self.calls[0]['json']))

    async def test_proactive_changed_account_stops_before_http(self):
        sender = Transport(self.ctx,None,self.route,proactive=True)
        self.platform.appid = 'other-account'
        with self.assertRaisesRegex(ValueError,'transport_route_expired_or_changed'):
            await sender.send('scheduled share',AsyncMock())
        self.assertEqual([],self.calls)

    async def test_proactive_video_without_incoming_message_has_two_receipts_and_no_passive_fields(self):
        from test_videos import FIRST,raw_video
        from ambient.videos import video_row,big_image_card
        self.response = lambda: ({'file_info':'preview-file'} if self.calls[-1]['url'].endswith('/files')
                                 else {'id':'auto-receipt-'+str(len(self.calls))})
        self.ctx.llm_generate.return_value=NS(completion_text=json.dumps({
            'reply':True,'video_id':FIRST,'text':'企鹅开始上班了','reason':'share'}))
        config={'groups':[{'umo':self.route.umo,'capture_only':False,'interject_enabled':True,
                          'video_sharing_enabled':True,'video_auto_share_enabled':True,'video_share_format':'image'}],
                'limits':{'reply_segment_interval_ms':0}}
        with tempfile.TemporaryDirectory() as tmp:
            rt=Runtime(tmp,self.ctx,config)
            try:
                video=video_row(raw_video())
                rt.videos.candidates={FIRST:video}
                rt.videos.fetched_at=datetime.now().timestamp()
                rt.videos.verify=AsyncMock()
                rt.videos.card=AsyncMock(return_value={'data':b'preview-jpeg','text':big_image_card(video)['text']})
                await rt.ensure(self.route)
                await rt.videos.auto.run(self.route)
                self.assertEqual(3,len(self.calls))
                wire=[c['json'] for c in self.calls]
                self.assertIs(wire[0]['srv_send_msg'],False)
                self.assertEqual([0,7],[wire[1]['msg_type'],wire[2]['msg_type']])
                self.assertIn('bilibili.com/video/'+FIRST,wire[1]['content'])
                self.assertEqual('',wire[2]['content'])
                for body in wire[1:]:
                    self.assertTrue({'msg_id','msg_seq','event_id'}.isdisjoint(body))
                rows=[r for r in rt.windows[self.route.key].rows if r['self']]
                self.assertEqual(2,len(rows))
                self.assertTrue(all(r['quote'] is None and r['source']=='video_share' for r in rows))
                self.assertEqual(1,rt.videos.auto.states[self.route.key]['confirmed'])
            finally:
                await rt.close()

    async def test_native_qq_adapter_preserves_card_for_plugin_even_without_text_or_at(self):
        from astrbot.core.platform.sources.qqofficial.qqofficial_platform_adapter import PatchedGroupMessage, QQOfficialPlatformAdapter
        from astrbot.core.platform.astrbot_message import MessageType
        from astrbot.core.provider.entities import ProviderRequest
        from ambient.context import event_message
        from test_video_cards import miniapp
        data={'id':'native-card','group_openid':'GROUP_A','content':'',
              'author':{'member_openid':'USER_1','username':'群友','bot':False},
              'timestamp':datetime.now(timezone.utc).isoformat(),
              'message_type':110,'msg_elements':[{'type':'json','data':json.dumps(miniapp(),ensure_ascii=False)}]}
        message=PatchedGroupMessage(self.client.api,None,data)
        abm=await QQOfficialPlatformAdapter._parse_from_qqofficial(message,MessageType.GROUP_MESSAGE)
        self.assertEqual('',abm.message_str)
        self.assertFalse(any(type(p).__name__=='Json' for p in abm.message))
        self.event.message_obj=abm
        self.event.message_str=''
        row=event_message(self.event,self.route)
        self.assertIn('奶蛙小剧场',row['text'])
        with tempfile.TemporaryDirectory() as tmp:
            rt=Runtime(tmp,self.ctx,{'groups':[{'umo':self.route.umo,'capture_only':False,'video_sharing_enabled':True}]})
            try:
                await rt.observe(self.event,request_type=ProviderRequest)
                self.assertEqual(1,len(rt.stores[self.route.key].video_taste.examples(now=datetime.now().timestamp())))
                self.assertIsNone(self.event.get_extra('provider_request'))
                self.event.is_at_or_wake_command=True
                await rt.observe(self.event,request_type=ProviderRequest)
                self.assertIn('奶蛙小剧场',self.event.get_extra('provider_request').prompt)
                self.assertEqual([],self.calls)
            finally:
                await rt.close()

    async def test_video_preview_upload_then_separate_text_and_image_without_ark(self):
        from ambient.delivery import send_reply_parts
        from ambient.policy import Policy
        card = {'data': b'preview-jpeg', 'text': '[B站视频：测试]\nhttps://www.bilibili.com/video/BV1234567890/'}
        self.response = lambda: ({'file_info':'preview-file'} if self.calls[-1]['url'].endswith('/files')
                                 else {'id':'receipt-'+str(len(self.calls))})
        confirmed = AsyncMock()
        gate = AsyncMock(side_effect=[10001,10002])
        text = '配文\nhttps://www.bilibili.com/video/BV1234567890/'
        receipts = await send_reply_parts(Transport(self.ctx, self.event, self.route), text,
            Policy(reply_segment_chars=8, reply_segment_interval_ms=0), gate, confirmed,
            max_parts=2, card=card)
        upload, caption, image = [c['json'] for c in self.calls]
        self.assertIs(upload['srv_send_msg'], False)
        self.assertEqual([0,7], [caption['msg_type'],image['msg_type']])
        self.assertEqual([text,''], [caption['content'],image['content']])
        self.assertEqual([10001,10002], [caption['msg_seq'],image['msg_seq']])
        self.assertEqual({'file_info':'preview-file'}, image['media'])
        self.assertFalse(caption['ark'] or image['ark'])
        self.assertEqual(['receipt-2','receipt-3'], receipts)
        self.assertEqual(card['text'], confirmed.await_args_list[-1].args[0])

    async def test_official_big_card_has_separate_bubble_receipt_and_sequence(self):
        from ambient.delivery import send_reply_parts
        from ambient.policy import Policy
        from ambient.videos import big_image_card, video_row
        from test_videos import raw_video
        card = big_image_card(video_row(raw_video()))
        self.response = lambda: {'id': 'receipt-'+str(len(self.calls))}
        gate = AsyncMock(side_effect=[10001, 10002, 10003])
        confirmed = AsyncMock()
        result = await send_reply_parts(Transport(self.ctx, self.event, self.route),
            '第一句。\n\n第二句。\n\n第三句。', Policy(reply_segment_interval_ms=0), gate, confirmed,
            max_parts=3, card=card)
        wire = [c['json'] for c in self.calls]
        self.assertEqual([0, 0, 3], [m['msg_type'] for m in wire])
        self.assertEqual(['第一句。', '第二句。\n\n第三句。', ''], [m['content'] for m in wire])
        self.assertEqual([10001, 10002, 10003], [m['msg_seq'] for m in wire])
        self.assertTrue(all(m['msg_id']=='m1' and not m['media'] for m in wire))
        self.assertTrue(all(not m['ark'] for m in wire[:2]))
        self.assertEqual(card['ark'], wire[-1]['ark'])
        self.assertEqual(['receipt-1', 'receipt-2', 'receipt-3'], result)
        self.assertEqual(card['text'], confirmed.await_args_list[-1].args[0])
        self.assertTrue(all(not c['allow_redirects'] for c in self.calls))

    async def test_card_permission_error_or_timeout_has_no_fallback_or_false_receipt(self):
        from ambient.delivery import send_reply_parts
        from ambient.policy import Policy
        from ambient.videos import big_image_card, video_row
        from test_videos import raw_video
        card = big_image_card(video_row(raw_video()))
        for timeout in (False, True):
            self.calls.clear()
            self.status = 200
            self.failure = None
            self.response = lambda: {'id': 'caption'} if len(self.calls)==1 else {'code': 40034126, 'message': 'private error'}
            async def record(text, receipt):
                if timeout: self.failure = TimeoutError()
                else: self.status = 400
            confirmed = AsyncMock(side_effect=record)
            with self.assertRaises((ValueError, TimeoutError)):
                await send_reply_parts(Transport(self.ctx, self.event, self.route), '配文',
                    Policy(reply_segment_interval_ms=0), AsyncMock(side_effect=[10001,10002]),
                    confirmed, max_parts=3, card=card)
            self.assertEqual(2, len(self.calls))
            confirmed.assert_awaited_once_with('配文', 'caption')

    async def test_card_permission_rejection_exposes_safe_actionable_error(self):
        from ambient.videos import big_image_card, video_row
        from test_videos import raw_video
        self.status, self.response = 400, {'code': 40034126, 'message': 'private response'}
        with self.assertRaisesRegex(ValueError, '^qq_ark_template_permission_required$'):
            await Transport(self.ctx, self.event, self.route).send('', AsyncMock(return_value=10001),
                ark=big_image_card(video_row(raw_video()))['ark'])
        self.assertEqual(1, len(self.calls))

    async def test_card_cannot_mix_text_or_image_in_same_native_request(self):
        from ambient.videos import big_image_card, video_row
        from test_videos import raw_video
        ark = big_image_card(video_row(raw_video()))['ark']
        gate = AsyncMock()
        sender = Transport(self.ctx, self.event, self.route)
        for text, media in (('配文', None), ('', {'file_info': 'image'})):
            with self.assertRaisesRegex(ValueError, 'qq_card_requires_separate_message'):
                await sender.send(text, gate, ark=ark, media=media)
        self.assertEqual([], self.calls)
        gate.assert_not_awaited()

    async def test_native_sticker_upload_does_not_send_or_consume_sequence(self):
        from ambient.transport import upload_qq_image
        self.response = {'file_info': 'native-file', 'file_uuid': 'upload-id', 'ttl': 600}
        media = await upload_qq_image(self.platform, self.route.group, b'test-image')
        self.assertEqual({'file_info': 'native-file'}, media)
        upload = self.calls[0]
        self.assertTrue(upload['url'].endswith('/v2/groups/GROUP_A/files'))
        self.assertIs(upload['json']['srv_send_msg'], False)
        self.assertNotIn('msg_id', upload['json'])
        self.assertNotIn('msg_seq', upload['json'])
        self.assertFalse(upload['allow_redirects'])
        self.response = {'id': 'image-receipt'}
        gate = AsyncMock(return_value=10002)
        receipt = await Transport(self.ctx, self.event, self.route).send('笑死', gate, media=media)
        self.assertEqual('image-receipt', receipt)
        self.assertEqual(7, self.calls[1]['json']['msg_type'])
        self.assertEqual(media, self.calls[1]['json']['media'])
        self.assertEqual(10002, self.calls[1]['json']['msg_seq'])
        self.assertEqual('m1', self.calls[1]['json']['msg_id'])
        gate.assert_awaited_once()

    async def test_native_sticker_invalid_upload_never_posts_message(self):
        from ambient.transport import upload_qq_image
        for response in ({'id': 'not-a-file'}, {'file_info': 'x', 'code': 22009}, {'file_info': ''}):
            self.response = response
            with self.assertRaises(ValueError):
                await upload_qq_image(self.platform, self.route.group, b'image')
        self.assertEqual(3, len(self.calls))
        self.assertTrue(all(c['url'].endswith('/files') for c in self.calls))

    async def test_native_sticker_has_empty_content_and_a_separate_sequence(self):
        from ambient.delivery import send_reply_parts
        from ambient.policy import Policy
        self.response = lambda: ({'file_info': 'native-file'} if self.calls[-1]['url'].endswith('/files')
                                 else {'id': 'receipt'+str(len(self.calls))})
        guard = AsyncMock(side_effect=[10001,10002,10003])
        confirmed = AsyncMock()
        await send_reply_parts(Transport(self.ctx,self.event,self.route), '第一段。\n\n第二段。',
            Policy(reply_segment_interval_ms=0), guard, confirmed, max_parts=3,
            sticker={'data':b'fake-image-for-wire-contract','caption':'合成测试图'})
        messages=[c['json'] for c in self.calls if not c['url'].endswith('/files')]
        self.assertEqual([0,0,7], [m['msg_type'] for m in messages])
        self.assertEqual(['第一段。','第二段。',''], [m.get('content','') for m in messages])
        self.assertEqual([10001,10002,10003], [m['msg_seq'] for m in messages])
        self.assertTrue(all(m['msg_id']=='m1' for m in messages))
        self.assertTrue(all(not m.get('media') for m in messages[:2]))
        self.assertEqual({'file_info':'native-file'}, messages[2]['media'])
        self.assertEqual(['第一段。','第二段。','[表情包：合成测试图]'],
                         [c.args[0] for c in confirmed.await_args_list])

    async def test_real_request_roundtrip_preserves_persona_and_does_not_save_ambient(self):
        import astrbot
        from astrbot.core.agent.message import Message, TextPart, dump_messages_with_checkpoints, CheckpointMessageSegment, CheckpointData
        from astrbot.core.provider.entities import ProviderRequest, LLMResponse
        ctx, event = context(), Event('还能玩吗')
        event.is_at_or_wake_command = True
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, ctx, {'groups':[{'umo':event.unified_msg_origin,'capture_only':False}]})
            try:
                await rt.observe(Event('刚刚输了三把', mid='old'))
                event.message_obj.raw_message.timestamp = datetime.now(timezone.utc)
                req = ProviderRequest(prompt='还能玩吗', system_prompt='RR 原有人格', conversation=NS(cid='conversation'))
                await rt.inject(event, req, TextPart)
                await rt.inject(event, req, TextPart)
                assembled = await req.assemble_context()
                message = Message.model_validate(assembled)
                self.assertEqual(1, sum(getattr(part,'_no_save',False) for part in message.content))
                self.assertIn('刚刚输了三把', json.dumps(message.model_dump(),ensure_ascii=False))
                saved = dump_messages_with_checkpoints([message])
                self.assertNotIn('ambient_data', json.dumps(saved,ensure_ascii=False))
                self.assertIn('还能玩吗', json.dumps(saved,ensure_ascii=False))
                from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
                runner = ToolLoopAgentRunner()
                await runner.reset(provider=NS(provider_config={'id':'synthetic'}), request=req,
                                   run_context=NS(messages=[]), tool_executor=NS(), agent_hooks=NS())
                self.assertEqual(['system','user'], [m.role for m in runner.run_context.messages])
                self.assertIn('刚刚输了三把', str(runner.run_context.messages))
                # Execute the exact deployed history method, without starting a bot.
                path = Path(astrbot.__file__).parent/'core/pipeline/process_stage/method/agent_sub_stages/internal.py'
                source = path.read_text(encoding='utf-8')
                method = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.AsyncFunctionDef) and n.name=='_save_to_history')
                body = '\n'.join(source.splitlines()[method.lineno-1:method.end_lineno])
                env = {'dump_messages_with_checkpoints':dump_messages_with_checkpoints,
                       'LLMResponse':LLMResponse, 'CheckpointMessageSegment':CheckpointMessageSegment, 'CheckpointData':CheckpointData}
                exec('from __future__ import annotations\n'+textwrap.dedent(body),env)
                manager = NS(update_conversation=AsyncMock())
                await env['_save_to_history'](NS(conv_manager=manager),event,req,
                    LLMResponse(role='assistant',completion_text='还能玩'),
                    [Message(role='system',content=req.system_prompt),message,Message(role='assistant',content='还能玩')],None)
                manager.update_conversation.assert_not_awaited()
            finally:
                await rt.close()

    async def asyncSetUp(self):
        from botpy.api import BotAPI
        from botpy.http import BotHttp
        self.ctx, self.event = context(), Event()
        self.http = BotHttp(timeout=5)
        self.token_gate = None
        async def token():
            if self.token_gate: await self.token_gate()
        self.http._token = NS(app_id='12345', check_token=token, get_string=lambda:'synthetic')
        self.client = NS(http=self.http, api=BotAPI(self.http), is_closed=lambda:False)
        self.platform = self.ctx.get_platform_inst('platform')
        self.platform.get_client = lambda:self.client
        self.calls, self.status, self.response, self.failure = [], 200, {'id':'native-receipt'}, None
        owner = self
        class Response:
            async def __aenter__(self):
                if owner.failure: raise owner.failure
                return self
            async def __aexit__(self,*args): return False
            @property
            def status(self): return owner.status
            async def json(self): return owner.response() if callable(owner.response) else owner.response
        def request(**kwargs):
            self.calls.append(kwargs)
            return Response()
        self.http._session = NS(closed=False,request=request)
        self.route = route_for(self.ctx, self.event)
    async def asyncTearDown(self):
        self.http._session.closed = True

    async def test_native_receipt_account_anchor_and_no_monkeypatch(self):
        sender = Transport(self.ctx,self.event,self.route)
        before = self.http.request
        guard = AsyncMock()
        receipt = await sender.send('synthetic reply', guard)
        self.assertEqual('native-receipt',receipt)
        self.assertEqual(before,self.http.request)
        self.assertEqual(1,len(self.calls))
        call = self.calls[0]
        self.assertEqual('https://api.sgroup.qq.com/v2/groups/GROUP_A/messages',call['url'])
        self.assertEqual('m1',call['json']['msg_id'])
        self.assertGreater(call['json']['msg_seq'],10000)
        guard.assert_awaited_once()

    async def test_normal_reply_uses_real_transport_and_shared_context(self):
        from astrbot.core.agent.message import TextPart
        from astrbot.core.provider.entities import ProviderRequest, LLMResponse
        from astrbot.core.message.message_event_result import MessageChain
        self.event.is_at_or_wake_command = True
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, self.ctx, {'groups':[{'umo': self.route.umo, 'capture_only':False}]})
            try:
                req = ProviderRequest(prompt='测试', contexts=[{'role':'assistant','content':'旧的扣1规则'}])
                await rt.inject(self.event, req, TextPart)
                await rt.response(self.event, LLMResponse(role='assistant', completion_text='新话题的回复'))
                result = await self.event.send(message=MessageChain().message('新话题的回复'))
                self.assertEqual({'id':'native-receipt'}, result)
                self.assertEqual(1, len(self.calls))
                self.assertEqual('m1', self.calls[0]['json']['msg_id'])
                row = list(rt.windows[self.route.key].rows)[-1]
                self.assertEqual(('reply', True, '新话题的回复'), (row['source'], row['self'], row['text']))
                self.assertEqual([], req.contexts)
                # Native duplicate decoration cannot cause a second API call.
                await self.event.send(MessageChain().message('新话题的回复'))
                self.assertEqual(1, len(self.calls))
            finally:
                await rt.close()

    async def test_bare_at_crosses_actual_host_empty_message_gate(self):
        import astrbot
        from astrbot.core.message.components import At, Image, File, Record, Reply, Video
        from astrbot.core.provider.entities import ProviderRequest
        from astrbot.core.agent.message import TextPart
        self.event.is_at_or_wake_command = True
        self.event.message_str = ''
        self.event.message_obj.message = [At(qq='qq_official')]
        # Execute the native gate, so changes in host component types or its
        # provider_request contract cannot be hidden by a synthetic predicate.
        path = Path(astrbot.__file__).parent/'core/pipeline/process_stage/method/agent_sub_stages/internal.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'process')
        block = next(n for n in ast.walk(method) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'has_provider_request' for t in n.targets))
        parent = next(n for n in ast.walk(method) if hasattr(n, 'body') and isinstance(n.body, list) and block in n.body)
        start = parent.body.index(block)
        statements = parent.body[start:start+5]
        self.assertIsInstance(statements[-1], ast.If)
        function = ast.parse('def gate(event):\n    pass').body[0]
        function.body = statements + [ast.Return(value=ast.Constant(True))]
        env = dict(Image=Image, File=File, Record=Record, Reply=Reply, Video=Video, logger=Mock())
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), '<native-empty-gate>', 'exec'), env)
        self.assertIsNone(env['gate'](self.event))
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, self.ctx, {'groups':[{'umo':self.route.umo, 'capture_only':False}]})
            try:
                await rt.observe(self.event, request_type=ProviderRequest)
                self.assertTrue(env['gate'](self.event))
                req = self.event.get_extra('provider_request')
                self.assertIsInstance(req, ProviderRequest)
                await rt.inject(self.event, req, TextPart)
                self.assertTrue(json.loads(req.extra_user_content_parts[-1].text)['ambient_data']['current_message']['attention_only'])
            finally:
                await rt.close()

    async def test_native_split_receipts_sequences_and_stream_replay(self):
        from astrbot.core.agent.message import TextPart
        from astrbot.core.provider.entities import ProviderRequest, LLMResponse
        from astrbot.core.message.message_event_result import MessageChain
        self.event.is_at_or_wake_command = True
        self.response = lambda: {'id': 'native-'+str(len(self.calls))}
        text = '第一段可以相同。\n\n第一段可以相同。\n\n第三段继续说。'
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, self.ctx, {'groups':[{'umo':self.route.umo,'capture_only':False}],
                                       'limits':{'reply_segment_interval_ms':0}})
            try:
                await rt.inject(self.event, ProviderRequest(prompt='详细解释'), TextPart)
                await rt.response(self.event, LLMResponse(role='assistant', completion_text=text))
                async def chunks():
                    for chunk in (text[:11], text[11:25], text[25:]):
                        yield MessageChain().message(chunk)
                await self.event.send_streaming(chunks())
                self.assertEqual(3, len(self.calls))
                self.assertEqual([10001, 10002, 10003], [r['json']['msg_seq'] for r in self.calls])
                self.assertTrue(all(r['json']['msg_id']=='m1' for r in self.calls))
                rows = [r for r in rt.windows[self.route.key].rows if r['self']]
                self.assertEqual(['native-1', 'native-2', 'native-3'], [r['id'] for r in rows])
                self.assertEqual(text, '\n\n'.join(r['text'] for r in rows))
                await self.event.send(MessageChain().message(text))
                self.assertEqual(3, len(self.calls))
            finally:
                await rt.close()

    async def test_native_second_part_failure_does_not_replay_first_or_send_third(self):
        from astrbot.core.agent.message import TextPart
        from astrbot.core.provider.entities import ProviderRequest, LLMResponse
        from astrbot.core.message.message_event_result import MessageChain
        self.event.is_at_or_wake_command = True
        self.response = lambda: {'id':'native-first'} if len(self.calls)==1 else {'code':22009}
        text = '前面已发。\n\n中间失败。\n\n后面不发。'
        with tempfile.TemporaryDirectory() as tmp:
            config = {'groups':[{'umo':self.route.umo,'capture_only':False}],
                      'limits':{'reply_segment_interval_ms':0}}
            rt = Runtime(tmp, self.ctx, config)
            try:
                await rt.inject(self.event, ProviderRequest(prompt='说几句'), TextPart)
                await rt.response(self.event, LLMResponse(role='assistant', completion_text=text))
                self.assertIsNone(await self.event.send(MessageChain().message(text)))
                self.assertEqual(2, len(self.calls))
                self.assertEqual(['前面已发。'], [r['text'] for r in rt.windows[self.route.key].rows if r['self']])
                self.assertEqual(3, rt.stores[self.route.key].remaining_qq_parts('m1'))
                await self.event.send(MessageChain().message(text))
                self.assertEqual(2, len(self.calls))
            finally:
                await rt.close()

    async def test_repaired_native_response_and_stream_have_one_confirmed_receipt(self):
        from astrbot.core.agent.message import TextPart
        from astrbot.core.provider.entities import ProviderRequest, LLMResponse
        from astrbot.core.message.message_event_result import MessageChain
        self.event.is_at_or_wake_command = True
        self.ctx.llm_generate.return_value = LLMResponse(role='assistant', completion_text='在，刚才走神了')
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, self.ctx, {'groups':[{'umo':self.route.umo, 'capture_only':False}]})
            try:
                await rt.inject(self.event, ProviderRequest(prompt='还活着吗'), TextPart)
                response = LLMResponse(role='assistant', result_chain=MessageChain().message('u1还活着'))
                await rt.response(self.event, response)
                self.assertEqual('在，刚才走神了', response.result_chain.get_plain_text())
                # Streaming chunks were emitted before the final response hook.
                async def chunks():
                    yield MessageChain().message('u1')
                    yield MessageChain().message('还活着')
                await self.event.send_streaming(chunks())
                await self.event.send(response.result_chain)
                self.assertEqual(1, len(self.calls))
                self.assertEqual('在，刚才走神了', self.calls[0]['json']['content'])
                self.ctx.llm_generate.assert_awaited_once()
            finally:
                await rt.close()

    async def test_host_followups_do_not_bypass_context_gate(self):
        from astrbot.core.pipeline.process_stage.follow_up import _ACTIVE_AGENT_RUNNERS, try_capture_follow_up
        runner = NS(follow_up=Mock(), run_context=NS(context=NS(event=self.event)))
        _ACTIVE_AGENT_RUNNERS[self.event.unified_msg_origin] = runner
        try:
            Runtime.isolate_followups(self.event)
            self.assertIsNone(try_capture_follow_up(self.event))
        finally:
            _ACTIVE_AGENT_RUNNERS.pop(self.event.unified_msg_origin, None)

    async def test_stream_fragments_cannot_bypass_alias_guard(self):
        from astrbot.core.agent.message import TextPart
        from astrbot.core.provider.entities import ProviderRequest
        from astrbot.core.message.message_event_result import MessageChain
        self.event.is_at_or_wake_command = True
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, self.ctx, {'groups':[{'umo': self.route.umo, 'capture_only':False}]})
            try:
                await rt.inject(self.event, ProviderRequest(prompt='测试'), TextPart)
                async def fragments(*texts):
                    for text in texts: yield MessageChain().message(text)
                await self.event.send_streaming(fragments('u', '1别闹了'))
                self.assertEqual([], self.calls)
                await self.event.send_streaming(fragments('正常', '回答'))
                self.assertEqual('正常回答', self.calls[0]['json']['content'])
            finally:
                await rt.close()

    async def test_builtin_message_tool_uses_event_transport_without_global_change(self):
        from astrbot.core.agent.tool import ToolSet
        from astrbot.core.provider.entities import ProviderRequest
        from astrbot.core.tools.message_tools import SendMessageToUserTool
        original = SendMessageToUserTool()
        tools = ToolSet(tools=[original])
        req = ProviderRequest(func_tool=tools)
        self.event.send = AsyncMock(return_value={'id':'synthetic-receipt'})
        host = NS(send_message=AsyncMock())
        ctx = NS(context=NS(event=self.event, context=host))
        route_send_tool(req, self.event)
        result = await req.func_tool.get_tool(original.name).call(ctx, messages=[{'type':'plain','text':'测试'}])
        self.assertIn('Message sent', result)
        self.event.send.assert_awaited_once()
        host.send_message.assert_not_awaited()
        self.assertIs(tools.get_tool(original.name), original)
        self.assertIs(ctx.context.context, host)
        self.event.send.return_value = None
        result = await req.func_tool.get_tool(original.name).call(ctx, messages=[{'type':'plain','text':'测试'}])
        self.assertIn('not confirmed', result)

    async def test_late_token_preemption_blocks_native_io(self):
        sender = Transport(self.ctx,self.event,self.route)
        async def change(): self.platform.appid='changed-account'
        self.token_gate=change
        with self.assertRaises(ValueError): await sender.send('reply',AsyncMock())
        self.assertEqual([],self.calls)

    async def test_unknown_errors_and_cancellation_never_retry(self):
        sender = Transport(self.ctx,self.event,self.route)
        for error in (ConnectionResetError(),TimeoutError(),asyncio.CancelledError()):
            self.calls.clear(); self.failure=error
            with self.assertRaises(type(error)): await sender.send('reply',AsyncMock())
            self.assertEqual(1,len(self.calls))

    async def test_invalid_receipts_and_expired_permission_fail_closed(self):
        for response in (True,None,{'id':''},{'id':'fake','code':1},{'message_id':'wrong-api'}):
            self.response=response
            with self.assertRaises(ValueError): await Transport(self.ctx,self.event,self.route).send('reply',AsyncMock())
        self.calls.clear()
        sender=Transport(self.ctx,self.event,self.route)
        sender.expires=0
        with self.assertRaises(ValueError): await sender.send('reply',AsyncMock())
        self.assertEqual([],self.calls)


if __name__ == '__main__':
    unittest.main()
