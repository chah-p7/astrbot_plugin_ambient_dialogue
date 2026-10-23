"""Scheduled group-wide shares, independent of passive reply windows."""
import asyncio
import json
import time

from .context import build_pack, compact
from .delivery import send_reply_parts
from .policy import SYSTEM_RULES, stable_id
from .search import generate, settings_id
from .transport import Transport
from .videos import AUTO_VIDEO_RULE, AUTO_VIDEO_TASK, VIDEO_RULE, bounded, parse_video, share_text


def task_event(route):
    """Native tool context for a background task; never injected as a user message."""
    from astrbot.core.platform.astr_message_event import AstrMessageEvent
    from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember, MessageType
    from astrbot.core.platform.platform_metadata import PlatformMetadata
    message = AstrBotMessage()
    message.type, message.self_id, message.group_id = MessageType.GROUP_MESSAGE, route.account, route.group
    message.message_id, message.message_str, message.message = '', '', []
    message.sender = MessageMember(route.account, 'bot')
    message.raw_message = None
    event = AstrMessageEvent('', message, PlatformMetadata(route.adapter, 'Ambient video task', route.platform), route.group)
    event.set_extra('ambient_background_task', 'video_share')
    return event


class AutoShares:
    def __init__(self, videos):
        self.videos, self.rt = videos, videos.rt
        self.states, self.tasks = {}, {}

    def enabled(self, route):
        return self.videos.enabled(route) and self.rt.settings(route).get('video_auto_share_enabled') is True

    def interval(self, route):
        return bounded(self.rt.settings(route), 'video_auto_share_interval_seconds', 1800, 0, 86400)

    def result(self, route, outcome, **extra):
        state = self.states.setdefault(route.key, {'checks':0, 'selected':0, 'confirmed':0, 'failed':0})
        state.pop('reason', None)
        state.update(outcome=outcome, at=time.time(), **extra)
        if outcome in {'selected', 'confirmed', 'failed'}:
            state[outcome] += 1
        elif outcome in {'partial_no_retry','uncertain_no_retry'}:
            state['failed'] += 1
        return state

    def tick(self, now):
        if self.rt.closed:
            return
        if self.videos.task and not self.videos.task.done():
            return
        for route in self.rt.routes.values():
            if not self.enabled(route) or route.key in self.tasks:
                continue
            state = self.states.setdefault(route.key, {'checks':0, 'selected':0, 'confirmed':0, 'failed':0})
            due = max(state.get('next_check', 0), self.rt.stores[route.key].last_video_at()+self.interval(route))
            state['next_check'] = due
            if now < due or self.rt.busy(route) or route.key in self.rt.interjections.tasks:
                continue
            rows = self.rt.windows[route.key].rows
            if rows and now-rows[-1]['at'] < max(self.rt.policy(route).interject_quiet_seconds,
                                               30 if rows[-1]['self'] or rows[-1].get('directed') else 0):
                continue
            if not self.videos.pool(route) or now-self.videos.fetched_at > bounded(self.rt.settings(route),'video_candidate_minutes',60,5,1440)*60:
                state['outcome'] = 'waiting_for_candidates'
                continue
            if len(self.tasks) >= 2:
                break
            state['next_check'] = now+300
            task = asyncio.create_task(self.run(route))
            self.tasks[route.key] = task
            task.add_done_callback(lambda _task, key=route.key: self.tasks.pop(key, None))

    async def run(self, route):
        rt, videos = self.rt, self.videos
        parts = 0
        sending = False
        try:
            if not self.enabled(route) or rt.busy(route):
                return
            event = task_event(route)
            settings, search_settings = stable_id(rt.config), settings_id(rt.context, event)
            provider, persona, identity = await rt.interjections.identity(route, event)
            store, window = await rt.ensure(route)
            policy = rt.policy(route)
            snap = await asyncio.to_thread(store.snapshot)
            incoming = tuple(r['id'] for r in window.rows if not r['self'])
            pack = build_pack(window, snap, None, now=time.time())
            offered = videos.add_choices(route, pack)
            if not offered:
                self.result(route, 'no_candidates')
                return
            transport = Transport(rt.context, None, route, proactive=True)
            state = self.result(route, 'deciding')
            state['checks'] += 1
            state['next_check'] = time.time()+300

            async def guard():
                _, _, current_identity = await rt.interjections.identity(route, event)
                if (rt.closed or not self.enabled(route) or rt.busy(route) or stable_id(rt.config)!=settings
                        or current_identity!=identity or settings_id(rt.context,event)!=search_settings
                        or store.memory_revision()!=snap['revision']
                        or tuple(r['id'] for r in window.rows if not r['self'])!=incoming):
                    raise ValueError('draft_preempted')
                transport.check()

            response = await asyncio.wait_for(generate(rt.context,event,chat_provider_id=provider,
                system_prompt=persona+'\n'+SYSTEM_RULES+VIDEO_RULE+AUTO_VIDEO_RULE,
                prompt=compact({'ambient_data':pack,'task':AUTO_VIDEO_TASK}),
                guard=guard,expected_settings=search_settings,max_tokens=600),policy.interject_timeout_seconds)
            raw = str(getattr(response,'completion_text','')).strip()
            if raw.startswith('```') and raw.endswith('```'):
                raw = raw.split('\n',1)[1].rsplit('```',1)[0]
            decision = json.loads(raw)
            if not isinstance(decision,dict) or decision.get('reply') is not True:
                reason = decision.get('reason') if isinstance(decision,dict) else ''
                self.result(route,reason if reason in {'busy_topic','declined','no_fit'} else 'model_deferred')
                return
            video = videos.selected(route,decision.get('video_id'),offered)
            if not video:
                self.result(route,'no_video_selected')
                return
            self.result(route,'selected')
            text = decision.get('text','').strip() if isinstance(decision.get('text'),str) else ''
            text, _ = parse_video(text,offered)
            await videos.verify(video,route)
            card = await videos.card(route,video)
            if not card or 'data' in card:
                text = share_text(text or video['title'],video)
            claimed = False

            async def before_send():
                nonlocal claimed,sending
                await guard()
                if not videos.selected(route,video['id'],offered):
                    raise ValueError('video_no_longer_available')
                if not claimed:
                    videos.claim(route,video)
                    claimed = True
                sending = True

            async def confirmed(part,receipt):
                nonlocal parts
                await rt.confirmed(route,None,part,receipt,'video_share')
                parts += 1

            async with rt.send_locks.setdefault(route.key,asyncio.Lock()):
                await send_reply_parts(transport,text,policy,before_send,confirmed,card=card)
            self.result(route,'confirmed',confirmed_parts=parts,
                        next_check=max(time.time(),store.last_video_at()+self.interval(route)))
            rt.interjections.suppress(route,now=time.time())
        except asyncio.CancelledError:
            self.result(route,'cancelled_no_retry',confirmed_parts=parts)
            raise
        except Exception as exc:
            from .runtime import error_code
            reason = error_code(exc)
            outcome = ('partial_no_retry' if parts else 'uncertain_no_retry') if sending else 'failed'
            self.result(route,outcome,reason=reason,confirmed_parts=parts)
            if reason in {'qq_proactive_permission_required','qq_proactive_rate_limited'}:
                self.states[route.key]['next_check'] = time.time()+3600

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        self.tasks.clear()
