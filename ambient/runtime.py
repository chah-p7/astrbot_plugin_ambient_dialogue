from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import time

from .context import Route, Window, build_pack, compact, event_message, route_for
from .interjection import Interjections
from .policy import Policy, SYSTEM_RULES, candidate
from .store import Store


class Runtime:
    def __init__(self, root, context, config):
        self.root, self.context, self.config = Path(root), context, config
        self.stores, self.windows, self.routes = {}, {}, {}
        self.locks, self.next_maintenance = {}, {}
        self.errors = 0
        self.injected = 0
        self.captured = 0
        self.interjections = Interjections(self)
        self.closed = False

    def settings(self, route):
        return next((r for r in self.config.get('groups', []) if r.get('umo') == route.umo), {})

    def enabled(self, route, interject=False):
        row = self.settings(route)
        if not row or row.get('enabled', True) is not True:
            return False
        expected = str(row.get('account', '') or '')
        if expected and expected != route.account:
            return False
        return not interject or (row.get('capture_only', True) is False and row.get('interject_enabled', False) is True)

    def policy(self, route):
        row = self.settings(route)
        overrides = dict(row.get('limits', {}))
        if row.get('quota_mb', 0):
            overrides['quota_mb'] = row['quota_mb']
        return Policy.from_config(self.config, {'limits': overrides})

    async def ensure(self, route):
        async with self.locks.setdefault(route.key, asyncio.Lock()):
            if route.key not in self.stores:
                policy = self.policy(route)
                store = await asyncio.to_thread(Store, self.root/'groups', route.key, policy)
                rows = await asyncio.to_thread(store.load_raw, now=time.time())
                self.stores[route.key], self.windows[route.key] = store, Window(policy, rows)
                self.routes[route.key] = route
            p = self.policy(route)
            self.stores[route.key].policy = p
            self.windows[route.key].policy = p
            return self.stores[route.key], self.windows[route.key]

    async def initialize(self):
        # Reopen our own databases so TTL cleanup continues across a quiet restart.
        for path in (self.root/'groups').glob('*/state.sqlite3'):
            try:
                def read_key():
                    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
                    try:
                        return json.loads(db.execute('SELECT group_id FROM meta WHERE id=1').fetchone()[0])
                    finally:
                        db.close()
                platform, account, group = await asyncio.to_thread(read_key)
                inst = self.context.get_platform_inst(platform)
                adapter = inst.meta().name if inst else 'qq_official'
                route = Route(platform, account, group, adapter)
                await self.ensure(route)
            except Exception:
                self.errors += 1

    async def record(self, route, row):
        store, window = await self.ensure(route)
        now = time.time()
        if not window.add(row, now):
            return False
        self.captured += 1
        try:
            await asyncio.to_thread(store.append_raw, row, now=now)
        except Exception:
            self.errors += 1
        return True

    async def observe(self, event):
        if not any(r.get('umo') == str(event.unified_msg_origin) for r in self.config.get('groups', [])):
            return
        route = route_for(self.context, event)
        if not self.enabled(route):
            return
        row = event_message(event, route)
        if row and await self.record(route, row):
            self.interjections.observe(route, event, row, now=time.time())

    async def prepare(self, event):
        if not any(r.get('umo') == str(event.unified_msg_origin) for r in self.config.get('groups', [])):
            return None
        route = route_for(self.context, event)
        if not self.enabled(route) or self.settings(route).get('capture_only', True):
            return None
        row = event_message(event, route)
        if row is None or row['self'] or row['command']:
            return None
        self.interjections.suppress(route, now=time.time())
        await self.record(route, row)
        store, window = await self.ensure(route)
        status = 'none'
        item = candidate(row['text'], store.policy.fact_chars)
        if item:
            try:
                member = await asyncio.to_thread(store.identify, route.platform, row['sender'], row['name'])
                status = await asyncio.to_thread(store.enqueue, member, item, row['id'])
            except ValueError:
                status = 'not_saved_storage_full_or_invalid'
        snap = await asyncio.to_thread(store.snapshot)
        return build_pack(window, snap, row, now=time.time(), memory_status=status)

    async def inject(self, event, request, part_type):
        pack = await self.prepare(event)
        if pack is None:
            return False
        # Materialize before mutating the host request: fail-open is all-or-nothing.
        part = part_type(text=compact({'ambient_data': pack})).mark_as_temp()
        previous = event.get_extra('ambient_dialogue_part')
        parts = [p for p in request.extra_user_content_parts if p is not previous]
        parts.append(part)
        system = request.system_prompt or ''
        if SYSTEM_RULES not in system:
            system = system+'\n'+SYSTEM_RULES
        request.extra_user_content_parts, request.system_prompt = parts, system
        event.set_extra('ambient_dialogue_part', part)
        self.injected += 1
        return True

    async def command(self, event):
        if not any(r.get('umo') == str(event.unified_msg_origin) for r in self.config.get('groups', [])):
            return None
        route = route_for(self.context, event)
        if not self.enabled(route):
            return None
        if self.settings(route).get('capture_only', True):
            return 'Ambient 正在采集验证，尚未接管记忆。'
        store, window = await self.ensure(route)
        self.interjections.suppress(route, now=time.time())
        account, name = str(event.get_sender_id()), event.get_sender_name()
        member = route.member(account)
        text = str(event.message_str).strip().removeprefix('/').removeprefix('轻聊').strip()
        if not text or text == '状态':
            snap = await asyncio.to_thread(store.snapshot, member)
            facts = snap['facts']+snap['summary']
            lines = ['Ambient 已启用；'+('普通记忆写入暂停' if snap['paused'] else '记忆可写')+
                     f"；本群 {snap['used_bytes']//1024}/{snap['quota_bytes']//1024} KiB。"]
            lines.extend(f"{r['kind']} {r['key']}：{r['text']} [{r['id'][:12]}]" for r in facts)
            lines.extend(f"待整理 {r['kind']}：{r['text']} [{r['id'][:12]}]" for r in snap['candidates'])
            lines.append('记住 类型 键=内容；删除 编号；完成 编号；忘记我。相同类型和键会纠正旧值。')
            return '\n'.join(lines)
        if text == '忘记我':
            await asyncio.to_thread(store.edit, 'forget_member', {}, member=member)
            window.forget(member)
            return '已删除你在本群的记忆和 Ambient 原话缓存。'
        if text.startswith(('删除 ', '完成 ')):
            operation, prefix = text.split(maxsplit=1)
            snap = await asyncio.to_thread(store.snapshot, member)
            ids = {r['id'] for r in snap['facts']+snap['summary']+snap['candidates'] if r['id'].startswith(prefix)}
            if len(ids) != 1 or len(prefix) < 8:
                return '编号不存在或不唯一，请先查看 /轻聊 状态。'
            await asyncio.to_thread(store.edit, 'complete' if operation == '完成' else 'delete',
                                    {'id': ids.pop()}, member=member)
            return '已清理该条记录及其摘要副本。'
        if text.startswith('记住 '):
            pieces = text.split(maxsplit=2)
            if len(pieces) == 3 and '=' in pieces[2]:
                kind = {'称呼': 'address', '偏好': 'preference', '事实': 'fact', '边界': 'boundary',
                        '承诺': 'promise', '待办': 'todo'}.get(pieces[1], pieces[1])
                key, value = pieces[2].split('=', 1)
                await asyncio.to_thread(store.identify, route.platform, account, name)
                await asyncio.to_thread(store.edit, 'set', {'kind': kind, 'key': key, 'text': value,
                    'quote': text, 'source': str(event.message_obj.message_id)}, member=member)
                return '已保存；再次使用相同类型和键可纠正。'
        return '用法：/轻聊 状态｜记住 偏好 饮料=无糖｜删除 编号｜完成 编号｜忘记我'

    async def maintain(self):
        now = time.time()
        for key, store in list(self.stores.items()):
            if now < self.next_maintenance.get(key, 0):
                continue
            try:
                await asyncio.to_thread(store.maintain)
                self.windows[key].trim(now)
            except Exception:
                self.errors += 1
            self.next_maintenance[key] = now+store.policy.batch_seconds
        await self.interjections.tick(now)

    def status(self):
        return {'captured': self.captured, 'injected': self.injected, 'errors': self.errors,
                'groups': {key: {'raw_in_memory': len(self.windows[key].rows),
                    'used_bytes': self.stores[key].size(),
                    'interjection': {k: getattr(self.interjections.states.get(key), k, 0)
                        for k in ('observed', 'checks', 'sent', 'outcome')}} for key in self.routes}}

    async def close(self):
        self.closed = True
        await self.interjections.close()
