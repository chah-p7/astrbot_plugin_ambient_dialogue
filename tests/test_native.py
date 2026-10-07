"""Real AstrBot/qq-botpy contracts, intercepted at the native HTTP boundary."""
from __future__ import annotations

import ast
import asyncio
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
    async def test_real_request_roundtrip_preserves_persona_and_does_not_save_ambient(self):
        import astrbot
        from astrbot.core.agent.message import Message, TextPart, dump_messages_with_checkpoints, CheckpointMessageSegment, CheckpointData
        from astrbot.core.provider.entities import ProviderRequest, LLMResponse
        ctx, event = context(), Event()
        event.is_at_or_wake_command = True
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, ctx, {'groups':[{'umo':event.unified_msg_origin,'capture_only':False}]})
            try:
                await rt.observe(Event('刚刚输了三把', mid='old'))
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
            async def json(self): return owner.response
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
