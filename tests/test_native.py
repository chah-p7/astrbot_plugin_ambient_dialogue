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
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ambient.runtime import Runtime
from ambient.transport import Transport
from ambient.context import route_for
from test_ambient import Event, context


@unittest.skipUnless(os.environ.get('AMBIENT_NATIVE_TEST'), 'requires installed AstrBot 4.26.7 and native SDK')
class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_request_roundtrip_preserves_persona_and_does_not_save_ambient(self):
        import astrbot
        from astrbot.core.agent.message import Message, TextPart, dump_messages_with_checkpoints, CheckpointMessageSegment, CheckpointData
        from astrbot.core.provider.entities import ProviderRequest, LLMResponse
        ctx, event = context(), Event()
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
                history = manager.update_conversation.call_args.kwargs['history']
                self.assertNotIn('ambient_data', json.dumps(history))
                self.assertEqual('conversation', manager.update_conversation.call_args.args[1])
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
