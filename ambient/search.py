"""Request-local access to the group's native, read-only web tools."""
from __future__ import annotations

import asyncio
from copy import copy

from .policy import stable_id


SEARCH_TOOLS = {
    'tavily': ('web_search_tavily', 'tavily_extract_web_page'),
    'bocha': ('web_search_bocha',),
    'brave': ('web_search_brave',),
    'firecrawl': ('web_search_firecrawl', 'firecrawl_extract_web_page'),
    'baidu_ai_search': ('web_search_baidu',),
    'exa': ('web_search_exa', 'exa_get_contents'),
}


class _Preempted(asyncio.CancelledError):
    # Native runners turn ordinary tool exceptions into model-visible text.
    # Cancellation must escape that handler, without cancelling the caller.
    pass


def settings_id(context, event):
    return stable_id(context.get_config(event.unified_msg_origin).get('provider_settings', {}))


def _guarded_tool(original, check, pending):
    tool = copy(original)

    async def call(*args, **kwargs):
        task = asyncio.current_task()
        pending.add(task)
        try:
            await check()
            if not original.active:
                raise _Preempted()
            try:
                result = await original.call(*args, **kwargs)
            except Exception as exc:
                # Do not forward raw HTTP errors, credentials or response bodies.
                result = f'Search unavailable ({type(exc).__name__}); no verified result.'
            await check()
            text = str(result or 'Search returned no result.')
            return text if len(text) <= 16000 else text[:16000]+'\n[Search result truncated]'
        finally:
            pending.discard(task)

    tool.call = call
    return tool


def _search_tools(context, settings, check, pending):
    names = SEARCH_TOOLS.get(settings.get('websearch_provider', 'tavily'), ())
    if not settings.get('web_search', False) or not names:
        return None
    from astrbot.core.agent.tool import ToolSet

    tools = ToolSet()
    manager = context.get_llm_tool_manager()
    for name in names:
        original = manager.get_builtin_tool(name)
        if original.active:
            tools.add_tool(_guarded_tool(original, check, pending))
    return None if tools.empty() else tools


async def generate(context, event, *, chat_provider_id, system_prompt, prompt,
                   guard, expected_settings=None, max_tokens=None):
    """Return a draft; the caller owns its deadline, validation and delivery."""
    settings = context.get_config(event.unified_msg_origin).get('provider_settings', {})
    expected_settings = expected_settings or stable_id(settings)
    pending = set()

    async def check():
        if settings_id(context, event) != expected_settings:
            raise _Preempted()
        try:
            await guard()
        except ValueError as exc:
            raise _Preempted() from exc

    try:
        await check()
        tools = _search_tools(context, settings, check, pending)
        kwargs = dict(chat_provider_id=chat_provider_id, system_prompt=system_prompt,
                      prompt=prompt, contexts=[], tools=tools, request_max_retries=0)
        if tools is None:
            if max_tokens is not None:
                kwargs['max_tokens'] = max_tokens
            response = await context.llm_generate(**kwargs, fallback_chat_provider_ids=[])
        else:
            # The host executes calls and feeds results back to the same model.
            # Its step setting is per request, not a group/message quota.
            response = await context.tool_loop_agent(
                event=event, **kwargs, stream=False, fallback_providers=[],
                max_steps=max(1, int(settings.get('max_agent_step', 30))),
                tool_call_timeout=max(1, int(settings.get('tool_call_timeout', 120))))
        await check()
    except _Preempted:
        raise ValueError('search_context_changed') from None
    finally:
        # AstrBot 4.26.7 may leave a tool's child task alive when its runner is
        # cancelled. Own only this request's tool tasks, never global tasks.
        tasks = list(pending)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if (getattr(response, 'role', 'assistant') != 'assistant'
            or getattr(response, 'tools_call_name', None)):
        raise ValueError('search_incomplete')
    return response
