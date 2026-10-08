"""Search execution, group isolation and interruption contracts."""
import asyncio
import json
import os
import tempfile
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

from ambient.search import SEARCH_TOOLS, _guarded_tool, _search_tools, generate
from test_ambient import Event, Part, context
import test_unified as unified


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_search_keeps_single_generation_and_checks_before_and_after(self):
        ctx, event, guard = context(), Event(), AsyncMock()
        await generate(ctx, event, chat_provider_id='ustc/free', system_prompt='persona',
                       prompt='current', guard=guard, max_tokens=600)
        self.assertEqual(2, guard.await_count)
        kwargs = ctx.llm_generate.call_args.kwargs
        self.assertIsNone(kwargs['tools'])
        self.assertEqual([], kwargs['contexts'])
        self.assertEqual([], kwargs['fallback_chat_provider_ids'])
        self.assertEqual(0, kwargs['request_max_retries'])
        self.assertEqual(600, kwargs['max_tokens'])

    async def test_tools_are_copied_bounded_and_errors_redacted(self):
        original = NS(active=True, call=AsyncMock(return_value='x'*20000))
        guard = AsyncMock()
        tool = _guarded_tool(original, guard, set())
        self.assertIsNot(tool, original)
        result = await tool.call('scope', query='public query')
        self.assertLess(len(result), 16100)
        self.assertIn('truncated', result)
        self.assertEqual(2, guard.await_count)
        self.assertIsInstance(original.call, AsyncMock)
        original.call.side_effect = RuntimeError('SECRET key and private response')
        self.assertEqual('Search unavailable (RuntimeError); no verified result.', await tool.call('scope'))
        original.call.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await tool.call('scope')

    async def test_changed_group_config_blocks_tool_before_io(self):
        ctx, event = context(), Event()
        settings = {'web_search': True, 'websearch_provider': 'tavily'}
        ctx.get_config = lambda _: {'provider_settings': settings}
        original = NS(active=True, call=AsyncMock(return_value='result'))
        async def run(**kwargs):
            settings['web_search'] = False
            await kwargs['tools'].call(None, query='query')
            self.fail('preempted tool must abort the runner')
        ctx.tool_loop_agent = AsyncMock(side_effect=run)
        with patch('ambient.search._search_tools', side_effect=lambda _ctx, _cfg, check, pending: _guarded_tool(original, check, pending)):
            with self.assertRaisesRegex(ValueError, 'search_context_changed'):
                await generate(ctx, event, chat_provider_id='ustc/free', system_prompt='persona',
                               prompt='current', guard=AsyncMock())
        original.call.assert_not_awaited()
        ctx.llm_generate.assert_not_awaited()

    async def test_no_final_answer_is_not_treated_as_a_completed_search(self):
        ctx = context()
        ctx.tool_loop_agent = AsyncMock(return_value=NS(role='tool', completion_text='', tools_call_name=['web_search_tavily']))
        with patch('ambient.search._search_tools', return_value=object()):
            with self.assertRaisesRegex(ValueError, 'search_incomplete'):
                await generate(ctx, Event(), chat_provider_id='ustc/free', system_prompt='persona',
                               prompt='current', guard=AsyncMock())


class SearchFlowTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = unified.UnifiedTests.asyncSetUp
    asyncTearDown = unified.UnifiedTests.asyncTearDown
    answer = unified.UnifiedTests.answer

    async def test_interjection_and_repair_use_search_and_shared_context(self):
        tool = NS(active=True, call=AsyncMock(return_value='PUBLIC_WEB_RESULT'))
        calls = []
        async def run(**kwargs):
            calls.append(kwargs)
            self.assertEqual('PUBLIC_WEB_RESULT', await kwargs['tools'].call(None, query='公开版本信息'))
            return NS(completion_text=(json.dumps({'reply': True, 'text': '现在可以查了'})
                      if len(calls) == 1 else '查到了，版本以官网为准'))
        self.ctx.tool_loop_agent = AsyncMock(side_effect=run)
        with patch('ambient.search._search_tools', side_effect=lambda _ctx, _cfg, check, pending: _guarded_tool(tool, check, pending)):
            event = Event('不知道现在是什么版本', mid='auto-search')
            await self.rt.observe(event)
            state = self.rt.interjections.states[self.route.key]
            await self.rt.interjections.run(self.route, state)
            direct = unified.directed('查一下版本', mid='repair-search')
            await self.answer(direct, 'u9查到了')
            await direct.send(NS(chain=[unified.Plain('u9查到了')]))
        self.assertEqual(['现在可以查了', '查到了，版本以官网为准'], self.sent)
        self.assertEqual(2, tool.call.await_count)
        self.assertEqual(2, self.ctx.tool_loop_agent.await_count)
        self.ctx.llm_generate.assert_not_awaited()
        self.assertIn('现在可以查了', calls[1]['prompt'])
        self.assertIn('查一下版本', calls[1]['prompt'])
        self.assertEqual([], calls[0]['contexts'])
        self.assertEqual('ustc/free', calls[0]['chat_provider_id'])
        self.assertEqual([], calls[0]['fallback_providers'])
        self.assertEqual(0, calls[0]['request_max_retries'])
        self.assertNotIn('PUBLIC_WEB_RESULT', str(list(self.rt.windows[self.route.key].rows)))
        self.assertNotIn('PUBLIC_WEB_RESULT', str(self.rt.stores[self.route.key].snapshot()))

    async def test_new_directed_message_during_search_preempts_auto_output(self):
        async def search(*args, **kwargs):
            await self.rt.observe(unified.directed('先回答我', mid='priority'))
            return 'outdated result'
        tool = NS(active=True, call=search)
        async def run(**kwargs):
            await kwargs['tools'].call(None, query='public query')
            self.fail('stale search must not continue generating a reply')
        self.ctx.tool_loop_agent = AsyncMock(side_effect=run)
        with patch('ambient.search._search_tools', side_effect=lambda _ctx, _cfg, check, pending: _guarded_tool(tool, check, pending)):
            await self.rt.observe(Event('现在出了新版本吗', mid='search'))
            await self.rt.interjections.run(self.route, self.rt.interjections.states[self.route.key])
        self.assertEqual([], self.sent)

    async def test_search_does_not_swallow_outer_deadline_or_restart_repair(self):
        tool = NS(active=True, call=AsyncMock(side_effect=TimeoutError('SECRET search response')))
        self.ctx.tool_loop_agent = AsyncMock(side_effect=TimeoutError('SECRET model response'))
        with patch('ambient.search._search_tools', return_value=tool):
            await self.answer(unified.directed(), 'u8')
        self.assertEqual([unified.REPLY_FAILURE_NOTICE], self.sent)
        self.ctx.tool_loop_agent.assert_awaited_once()


@unittest.skipUnless(os.environ.get('AMBIENT_NATIVE_TEST'), 'requires AstrBot 4.26.7')
class NativeSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from astrbot.core.star.context import Context
        from astrbot.core.platform.astr_message_event import AstrMessageEvent
        from astrbot.core.provider.provider import Provider
        from astrbot.core.provider.func_tool_manager import FunctionToolManager
        self.ctx, self.event = Mock(spec=Context), Mock(spec=AstrMessageEvent)
        self.event.unified_msg_origin = 'platform:GroupMessage:search-group'
        self.event.send = AsyncMock()
        self.settings = {'web_search': True, 'websearch_provider': 'tavily', 'websearch_tavily_key': ['synthetic']}
        self.ctx.get_config = lambda umo: {'provider_settings': self.settings} if umo == self.event.unified_msg_origin else {'provider_settings': {}}
        self.manager = FunctionToolManager()
        self.ctx.get_llm_tool_manager = lambda: self.manager
        self.provider = Mock(spec=Provider)
        self.provider.provider_config = {'id': 'native-search'}
        self.provider.text_chat = AsyncMock()
        self.ctx.provider_manager = NS(get_provider_by_id=AsyncMock(return_value=self.provider))
        self.ctx.tool_loop_agent = MethodType(Context.tool_loop_agent, self.ctx)

    async def test_native_provider_selection_active_flags_and_request_local_copies(self):
        for provider, names in SEARCH_TOOLS.items():
            self.settings['websearch_provider'] = provider
            tools = _search_tools(self.ctx, self.settings, AsyncMock(), set())
            self.assertEqual(set(names), {tool.name for tool in tools.tools})
            for tool in tools.tools:
                self.assertIsNot(tool, self.manager.get_builtin_tool(tool.name))
        self.settings['websearch_provider'] = 'tavily'
        native = self.manager.get_builtin_tool('web_search_tavily')
        native.active = False
        tools = _search_tools(self.ctx, self.settings, AsyncMock(), set())
        self.assertEqual({'tavily_extract_web_page'}, {tool.name for tool in tools.tools})
        native.active = True
        self.assertIsNone(_search_tools(self.ctx, {}, AsyncMock(), set()))
        self.assertIsNone(_search_tools(self.ctx, {'web_search': True, 'websearch_provider': 'default'}, AsyncMock(), set()))

    async def test_real_native_loop_executes_tavily_and_feeds_results_to_model(self):
        from astrbot.core.provider.entities import LLMResponse
        from astrbot.core.tools.web_search_tools import SearchResult
        self.provider.text_chat.side_effect = [
            LLMResponse(role='tool', tools_call_name=['web_search_tavily'],
                        tools_call_args=[{'query': 'AstrBot release'}], tools_call_ids=['call-search']),
            LLMResponse(role='assistant', completion_text='查到了 https://example.org/release'),
        ]
        with patch('astrbot.core.tools.web_search_tools._tavily_search', new_callable=AsyncMock) as http:
            http.return_value = [SearchResult(title='Release', url='https://example.org/release', snippet='VERIFIED_RELEASE')]
            response = await generate(self.ctx, self.event, chat_provider_id='native-search',
                                      system_prompt='RR persona', prompt='查一下版本', guard=AsyncMock())
        self.assertIn('example.org/release', response.completion_text)
        http.assert_awaited_once()
        self.assertEqual('AstrBot release', http.call_args.args[1]['query'])
        self.assertIs(self.settings, http.call_args.args[0])
        self.assertEqual(2, self.provider.text_chat.await_count)
        followup = self.provider.text_chat.call_args.kwargs
        self.assertIn('VERIFIED_RELEASE', str(followup['contexts']))
        self.assertIn('call-search', str(followup['contexts']))
        self.assertNotIn('synthetic', str(followup['contexts']))
        self.assertEqual({'web_search_tavily', 'tavily_extract_web_page'}, {t.name for t in followup['func_tool'].tools})
        self.event.send.assert_not_awaited()

    async def test_native_loop_cancellation_stops_search_before_network(self):
        from astrbot.core.provider.entities import LLMResponse
        async def model(**kwargs):
            self.settings['web_search'] = False
            return LLMResponse(role='tool', tools_call_name=['web_search_tavily'],
                               tools_call_args=[{'query': 'should not leave the process'}], tools_call_ids=['stale'])
        self.provider.text_chat.side_effect = model
        with patch('astrbot.core.tools.web_search_tools._tavily_search', new_callable=AsyncMock) as http:
            with self.assertRaisesRegex(ValueError, 'search_context_changed'):
                await generate(self.ctx, self.event, chat_provider_id='native-search',
                               system_prompt='persona', prompt='current', guard=AsyncMock())
        http.assert_not_awaited()
        self.provider.text_chat.assert_awaited_once()
        self.event.send.assert_not_awaited()

    async def test_cancelled_native_loop_cleans_up_inflight_search(self):
        from astrbot.core.provider.entities import LLMResponse
        started, stopped = asyncio.Event(), asyncio.Event()
        async def http(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        self.provider.text_chat.return_value = LLMResponse(role='tool', tools_call_name=['web_search_tavily'],
            tools_call_args=[{'query': 'public query'}], tools_call_ids=['cancel-search'])
        with patch('astrbot.core.tools.web_search_tools._tavily_search', side_effect=http):
            task = asyncio.create_task(generate(self.ctx, self.event, chat_provider_id='native-search',
                system_prompt='persona', prompt='current', guard=AsyncMock()))
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stopped.is_set())
        self.provider.text_chat.assert_awaited_once()
        self.event.send.assert_not_awaited()

    async def test_native_loop_cannot_execute_unlisted_send_tool(self):
        from astrbot.core.provider.entities import LLMResponse
        self.provider.text_chat.side_effect = [
            LLMResponse(role='tool', tools_call_name=['send_message_to_user'],
                        tools_call_args=[{'message': 'do not send'}], tools_call_ids=['unlisted']),
            LLMResponse(role='assistant', completion_text='只返回草稿'),
        ]
        response = await generate(self.ctx, self.event, chat_provider_id='native-search',
            system_prompt='persona', prompt='current', guard=AsyncMock())
        self.assertEqual('只返回草稿', response.completion_text)
        self.event.send.assert_not_awaited()

    async def test_ordinary_request_keeps_native_search_and_other_host_tools(self):
        from astrbot.core.agent.tool import FunctionTool, ToolSet
        from astrbot.core.provider.entities import ProviderRequest
        from ambient.runtime import Runtime
        ctx, event = context(), unified.directed('查一下版本')
        native = self.manager.get_builtin_tool('web_search_tavily')
        other = FunctionTool(name='existing_host_tool', description='test',
                             parameters={'type': 'object', 'properties': {}}, handler=AsyncMock())
        tools = ToolSet()
        tools.add_tool(native)
        tools.add_tool(other)
        req = ProviderRequest(prompt=event.message_str, system_prompt='RR persona', func_tool=tools)
        with tempfile.TemporaryDirectory() as tmp:
            rt = Runtime(tmp, ctx, {'groups': [{'umo': event.unified_msg_origin, 'capture_only': False}]})
            try:
                self.assertTrue(await rt.inject(event, req, Part))
                self.assertIs(tools, req.func_tool)
                self.assertIs(native, req.func_tool.get_tool('web_search_tavily'))
                self.assertIs(other, req.func_tool.get_tool('existing_host_tool'))
                self.assertIn('先搜索核实', req.system_prompt)
                self.assertIn('current_message', req.extra_user_content_parts[-1].text)
                ctx.llm_generate.assert_not_awaited()
            finally:
                await rt.close()


if __name__ == '__main__':
    unittest.main()
