from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import time

from .context import build_pack, compact
from .delivery import send_reply_parts
from .policy import SYSTEM_RULES, noise, reply_rejection, stable_id
from .transport import Transport
from .stickers import STICKER_RULE, parse_sticker
from .learning import mentions
from .search import generate, settings_id


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
        if mentions(row) and 'bot' not in mentions(row):
            state.pending = False
            state.outcome = 'addressed_to_other_member'
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
        confirmed_parts = 0
        try:
            search_settings = settings_id(rt.context, event)
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
            offered_stickers = rt.stickers.add_choices(route, pack)

            async def draft_guard():
                _, _, current_identity = await self.identity(route, event)
                if (self.closed or not rt.enabled(route, interject=True) or settings != stable_id(rt.config)
                        or current_identity != identity or state.generation != generation
                        or rt.busy(route) or time.time()-state.last_input > p.interject_fresh_seconds
                        or time.time() < state.suppressed_until or store.memory_revision() != snap['revision']
                        or search_settings != settings_id(rt.context, event)):
                    raise ValueError('draft_preempted')
                transport.check()

            state.checks += 1
            response = await asyncio.wait_for(generate(rt.context, event,
                chat_provider_id=provider, system_prompt=persona+'\n'+SYSTEM_RULES+(STICKER_RULE if offered_stickers else '')+
                '\n你正在判断是否自然接入群聊。结合接话对象和近期反馈，只有读懂原意且有自然的接法才说；普通确认、办事问答和对别人的邀约可以安静旁听。'
                '收到明确拒绝插话，先退出这段对话，等明显换题或有人重新向你搭话。不要代被点名的人回答，不把每句话都加工成比喻或段子。'
                '仅返回 JSON {"reply":true或false,"text":"一句自然回复"}，不提及判断过程。',
                prompt=compact({'ambient_data': pack}), guard=draft_guard,
                expected_settings=search_settings, max_tokens=600), p.interject_timeout_seconds)
            raw = str(getattr(response, 'completion_text', '')).strip()
            if raw.startswith('```') and raw.endswith('```'):
                raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
            decision = json.loads(raw)
            if not isinstance(decision, dict) or decision.get('reply') is not True:
                state.outcome = 'silent'
                return
            text = decision.get('text', '').strip() if isinstance(decision.get('text'), str) else ''
            text, selected = parse_sticker(text, offered_stickers)
            sticker = rt.stickers.selected(route, selected, offered_stickers)
            rejection = reply_rejection(text, current, window, interject=True) if text or not sticker else ''
            if rejection or len(text) > p.interject_max_chars or text.startswith('/'):
                state.outcome = rejection or 'invalid_draft'
                return

            claimed = False

            async def before_send():
                nonlocal claimed
                await draft_guard()
                # Synchronous, short transaction: no await between final check and claim.
                if not claimed and not store.claim_interjection(anchor, stable_id(text, selected), snap['revision']):
                    raise ValueError('draft_consumed_or_memory_changed')
                rejection = reply_rejection(text, current, window, interject=True) if text or not sticker else ''
                if sticker and not rt.stickers.selected(route, selected, offered_stickers):
                    raise ValueError('sticker_no_longer_available')
                if rejection or (not claimed and not store.claim_delivery(current['id'], stable_id(text, selected))):
                    raise ValueError(rejection or 'duplicate_delivery')
                claimed = True
                sequence = store.claim_qq_part(current['id']) if route.adapter == 'qq_official' else None
                state.outcome = 'sending'
                return sequence

            async def record_part(part, receipt):
                nonlocal confirmed_parts
                await rt.confirmed(route, current, part, receipt, 'interjection')
                confirmed_parts += 1
                state.sent += 1

            state.outcome = 'preparing_send'
            async with rt.send_locks.setdefault(route.key, asyncio.Lock()):
                await send_reply_parts(transport, text, p, before_send, record_part,
                    sticker=sticker,
                    max_parts=store.remaining_qq_parts(current['id']) if route.adapter == 'qq_official' else None)
            state.outcome = 'confirmed'
            rt.delivery_audit(route, current, 'interjection', confirmed_parts)
            state.suppressed_until = time.time()+p.interject_cooldown_seconds
        except asyncio.CancelledError:
            state.outcome = 'cancelled_no_retry'
            raise
        except Exception as exc:
            # Never log raw chats, drafts, tokens or adapter exception bodies.
            state.outcome = ('partial_no_retry' if confirmed_parts else 'uncertain_no_retry') if state.outcome == 'sending' else type(exc).__name__
            rt.errors += 1

    async def close(self):
        self.closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
