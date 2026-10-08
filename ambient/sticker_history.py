"""Read only the selected group's retained host history; never scrape other chats."""
import asyncio
from datetime import datetime, timezone
import json
import time

from .policy import stable_id
from .stickers import image_sources


def timestamp(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except (ValueError, TypeError):
        return 0


async def import_history(stickers, route):
    bank, context = stickers.bank(route), stickers.rt.context
    report = {'state': 'running', 'started': int(time.time()), 'messages': 0, 'images': 0,
              'downloaded': 0, 'unavailable': 0, 'conversations': 0, 'approximate_time': 0, 'failures': {}}
    bank.metadata('history', report)

    async def consume(identity, at, parts):
        if not stickers.enabled(route) or stickers.rt.closed:
            raise ValueError('group_disabled')
        if report['messages'] >= 20000:
            report['truncated'] = True
            return
        report['messages'] += 1
        for source in set(image_sources(parts, archived=True)):
            report['images'] += 1
            try:
                result = await stickers.ingest(route, identity, at, 'history', source)
                report['downloaded' if result else 'unavailable'] += 1
                if not result:
                    report['failures']['outside_retention_or_disabled'] = report['failures'].get('outside_retention_or_disabled',0)+1
            except Exception as exc:
                report['unavailable'] += 1
                # Only our own fixed reason codes; never expose URLs or exception bodies.
                reason = str(exc) if isinstance(exc, ValueError) and str(exc).startswith('image_') and len(str(exc)) < 64 else type(exc).__name__
                report['failures'][reason] = report['failures'].get(reason, 0)+1
            bank.metadata('history', report)

    try:
        excluded = set(stickers.rt.settings(route).get('style_excluded_senders', [])) | {route.account}
        manager = getattr(context, 'message_history_manager', None)
        if manager:
            for scope in (route.group, route.umo):
                for page in range(1, 51):
                    if not stickers.enabled(route) or stickers.rt.closed:
                        raise ValueError('group_disabled')
                    rows = await manager.get(platform_id=route.platform, user_id=scope, page=page, page_size=200)
                    for row in rows:
                        if (row.platform_id != route.platform or row.user_id != scope
                                or not row.sender_id or row.sender_id in excluded):
                            continue
                        content = row.content
                        parts = content if isinstance(content, list) else content.get('message', content.get('content', []))
                        if not isinstance(parts, list):
                            continue
                        mid = content.get('message_id') if isinstance(content, dict) else None
                        await consume(str(mid) if mid else 'host:'+str(row.id), timestamp(row.created_at), parts)
                    if len(rows) < 200:
                        break
        manager = getattr(context, 'conversation_manager', None)
        if manager:
            conversations = await manager.get_conversations(unified_msg_origin=route.umo)
            for conv in conversations[:100]:
                if conv.user_id != route.umo:
                    continue
                report['conversations'] += 1
                messages = json.loads(conv.history) if isinstance(conv.history, str) else conv.history
                for row in (messages or [])[:10000]:
                    if not isinstance(row, dict) or row.get('role') != 'user':
                        continue
                    if row.get('sender_id') in excluded:
                        continue
                    parts = row.get('content')
                    if not isinstance(parts, list):
                        continue
                    at = timestamp(row.get('timestamp') or row.get('created_at'))
                    if not at:
                        at = timestamp(conv.created_at)
                        report['approximate_time'] += 1
                    # Missing native IDs cannot prove two identical archived images
                    # were separate events. Conservative dedup also handles forks.
                    identity = row.get('message_id') or 'archive:'+stable_id(parts, row.get('timestamp'))
                    await consume(str(identity), at, parts)
        report['state'] = 'complete'
    except asyncio.CancelledError:
        report['state'] = 'cancelled'
        raise
    except Exception as exc:
        report['state'], report['error'] = 'failed', type(exc).__name__
    finally:
        report['finished'] = int(time.time())
        bank.metadata('history', report)
    return report
