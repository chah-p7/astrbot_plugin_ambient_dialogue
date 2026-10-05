from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time

from .context import build_pack, compact
from .policy import SYSTEM_RULES, noise, normalized, reply_rejection, stable_id
from .transport import Transport


@dataclass
class Participation:
    generation: int = 0
    pending: bool = False
    last_input: float = 0
    next_check: float = 0
    suppressed_until: float = 0
    event: object = None
    current: dict = None
    observed: int = 0
    checks: int = 0
    sent: int = 0
    outcome: str = 'waiting_for_messages'


class Interjections:
    def __init__(self, runtime):
        self.runtime = runtime
        self.states = {}
        self.tasks = {}
        self.closed = False

    def observe(self, route, event, row, *, now):
        state = self.states.setdefault(route.key, Participation())
        state.generation += 1
        state.event, state.current = event, row
        state.last_input = row['at']
        state.observed += 1
        state.pending = not (row['command'] or row['self'] or row['attachment'] or row['directed'] or noise(row['text']))
        if row['directed'] or row['self']:
            self.suppress(route, now=now)

    def suppress(self, route, *, now):
        state = self.states.setdefault(route.key, Participation())
        state.generation += 1
        state.pending = False
        state.suppressed_until = now + max(30, self.runtime.policy(route).interject_cooldown_seconds)
        state.outcome = 'normal_reply_priority'

    async def identity(self, route, event):
        ctx = self.runtime.context
        provider = await ctx.get_current_chat_provider_id(route.umo)
        cid = await ctx.conversation_manager.get_curr_conversation_id(route.umo)
        conv = await ctx.conversation_manager.get_conversation(route.umo, cid) if cid else None
        _, persona, _, _ = await ctx.persona_manager.resolve_selected_persona(
            umo=route.umo, conversation_persona_id=getattr(conv, 'persona_id', None),
            platform_name=route.adapter, provider_settings=ctx.get_config(route.umo).get('provider_settings', {}))
        prompt = (persona or {}).get('prompt', '')
        return provider, prompt, stable_id(provider, persona, cid)

    async def tick(self, now=None):
        now = time.time() if now is None else now
        for key, state in list(self.states.items()):
            route = self.runtime.routes.get(key)
            if not route or not self.runtime.enabled(route, interject=True) or key in self.tasks or self.runtime.busy(route):
                continue
            p = self.runtime.policy(route)
            if (not state.pending or now < max(state.next_check, state.suppressed_until)
                    or now-state.last_input < p.interject_quiet_seconds):
                continue
            if now-state.last_input > p.interject_fresh_seconds:
                state.pending, state.outcome = False, 'stale'
                continue
            if len(self.tasks) >= 2:
                break
            state.pending = False
            state.next_check = now+p.interject_interval_seconds
            task = asyncio.create_task(self.run(route, state))
            self.tasks[key] = task
            task.add_done_callback(lambda _task, key=key: self.tasks.pop(key, None))

    async def run(self, route, state):
        generation, event, current = state.generation, state.event, state.current
        rt, p = self.runtime, self.runtime.policy(route)
        settings = stable_id(rt.config)
        try:
            transport = Transport(rt.context, event, route)
            provider, persona, identity = await self.identity(route, event)
            store, window = await rt.ensure(route)
            snap = await asyncio.to_thread(store.snapshot)
            gate = await asyncio.to_thread(store.interjection_gate)
            anchor = stable_id(route.key, current['id'])
            if time.time()-gate['last_at'] < p.interject_cooldown_seconds or anchor == gate['anchor']:
                state.outcome = 'cooldown_or_consumed'
                return
            if rt.busy(route) or time.time()-current['at'] > p.interject_fresh_seconds:
                state.outcome = 'normal_reply_pending_or_stale'
                return
            pack = build_pack(window, snap, current, now=time.time())
            state.checks += 1
            response = await asyncio.wait_for(rt.context.llm_generate(
                chat_provider_id=provider, system_prompt=persona+'\n'+SYSTEM_RULES+
                '\n你正在判断是否自然接入群聊。无需每次发言；若有明确可接的话再说。'
                '仅返回 JSON {"reply":true或false,"text":"一句自然回复"}，不提及判断过程。',
                prompt=compact({'ambient_data': pack}), contexts=[], tools=None,
                fallback_chat_provider_ids=[], max_tokens=600, request_max_retries=0), p.interject_timeout_seconds)
            raw = str(getattr(response, 'completion_text', '')).strip()
            if raw.startswith('```') and raw.endswith('```'):
                raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
            decision = json.loads(raw)
            if not isinstance(decision, dict) or decision.get('reply') is not True:
                state.outcome = 'silent'
                return
            text = normalized(decision.get('text', '')) if isinstance(decision.get('text'), str) else ''
            rejection = reply_rejection(text, current, window, interject=True)
            if rejection or len(text) > p.interject_max_chars or text.startswith('/'):
                state.outcome = rejection or 'invalid_draft'
                return

            async def before_send():
                _, _, current_identity = await self.identity(route, event)
                if (self.closed or not rt.enabled(route, interject=True) or settings != stable_id(rt.config)
                        or current_identity != identity or state.generation != generation
                        or rt.busy(route)
                        or time.time()-state.last_input > p.interject_fresh_seconds
                        or time.time() < state.suppressed_until):
                    raise ValueError('draft_preempted')
                transport.check()
                # Synchronous, short transaction: no await between final check and claim.
                if not store.claim_interjection(anchor, stable_id(text), snap['revision']):
                    raise ValueError('draft_consumed_or_memory_changed')
                rejection = reply_rejection(text, current, window, interject=True)
                if rejection or not store.claim_delivery(current['id'], stable_id(text)):
                    raise ValueError(rejection or 'duplicate_delivery')
                state.outcome = 'sending'

            state.outcome = 'preparing_send'
            async with rt.send_locks.setdefault(route.key, asyncio.Lock()):
                receipt = await asyncio.wait_for(transport.send(text, before_send), 20)
                await rt.confirmed(route, current, text, receipt, 'interjection')
            state.sent += 1
            state.outcome = 'confirmed'
            state.suppressed_until = time.time()+p.interject_cooldown_seconds
        except asyncio.CancelledError:
            state.outcome = 'cancelled_no_retry'
            raise
        except Exception as exc:
            # Never log raw chats, drafts, tokens or adapter exception bodies.
            state.outcome = 'uncertain_no_retry' if state.outcome == 'sending' else type(exc).__name__
            rt.errors += 1

    async def close(self):
        self.closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
