"""Single native receipt, no retry. Adapted from BotLife (AGPL-3.0)."""
from __future__ import annotations

from datetime import datetime
import base64
from functools import lru_cache
import hashlib
import inspect
import time
from types import SimpleNamespace

from .context import route_for
from .policy import stable_id


def native_id(value):
    return isinstance(value, str) and 0 < len(value) <= 256 and all(ord(c) > 32 and ord(c) != 127 for c in value)


@lru_cache(maxsize=4)
def sdk_verified(post_group, check_session):
    try:
        hashes = tuple(hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                       for fn in (post_group, check_session))
    except (TypeError, OSError):
        return False
    return hashes == (
        '2036d7c42f3402ff57837fc38d8b4fc8ee99b4c615d58df19a7c9558b325f516',
        '330c911d023da65cfc0130ff1a50796b6708907966addc66db1a36937e954e81')


def official_client(platform):
    from botpy.api import BotAPI
    from botpy.http import BotHttp
    client = platform.get_client()
    api, http = client.api, client.http
    if (type(api) is not BotAPI or type(http) is not BotHttp or api._http is not http
            or getattr(api.post_group_message, '__func__', None) is not BotAPI.post_group_message
            or getattr(http.check_session, '__func__', None) is not BotHttp.check_session
            or not sdk_verified(BotAPI.post_group_message, BotHttp.check_session)
            or str(getattr(getattr(http, '_token', None), 'app_id', '')) != str(platform.appid)
            or client.is_closed()):
        raise ValueError('qq_native_contract_unavailable')
    return client


async def upload_qq_image(platform, group, png, check=lambda: None):
    """Upload only: srv_send_msg=False never posts an unsolicited QQ message."""
    from aiohttp import ClientTimeout
    from botpy.http import Route as QQRoute
    client = official_client(platform)
    await client.http.check_session()
    check()
    if official_client(platform) is not client:
        raise ValueError('qq_client_changed')
    route = QQRoute('POST', '/v2/groups/{group_openid}/files', group_openid=group)
    route.is_sandbox = client.http.is_sandbox
    if not png or len(png) > 2*1024*1024:
        raise ValueError('sticker_upload_size')
    async with client.http._session.request(method=route.method, url=route.url,
            headers=client.http._headers, timeout=ClientTimeout(total=15), allow_redirects=False,
            json={'file_type': 1, 'file_data': base64.b64encode(png).decode(),
                  'srv_send_msg': False}) as response:
        if response.status != 200:
            raise ValueError('sticker_upload_unconfirmed')
        data = await response.json()
        if (not isinstance(data, dict) or any(k in data for k in ('code', 'error', 'retcode'))
                or not isinstance(data.get('file_info'), str) or not 0 < len(data['file_info']) <= 16384):
            raise ValueError('sticker_upload_unconfirmed')
        return {'file_info': data['file_info']}


class Transport:
    def __init__(self, context, event, route, *, now=None):
        now = time.time() if now is None else now
        self.context, self.event, self.route = context, event, route
        self.platform = context.get_platform_inst(route.platform)
        self.expires = now + 180
        self.anchor = str(getattr(event.message_obj, 'message_id', '') or '')
        if route.adapter == 'qq_official':
            raw = event.message_obj.raw_message
            stamp = getattr(raw, 'timestamp', None)
            at = stamp.timestamp() if isinstance(stamp, datetime) else datetime.fromisoformat(str(stamp).replace('Z', '+00:00')).timestamp()
            self.expires = min(self.expires, min(now, at)+270)
            if not native_id(self.anchor):
                raise ValueError('qq_reply_anchor_unavailable')
        self.check(now)

    def check(self, now=None):
        now = time.time() if now is None else now
        if (now >= self.expires or self.context.get_platform_inst(self.route.platform) is not self.platform
                or route_for(self.context, self.event) != self.route):
            raise ValueError('transport_route_expired_or_changed')

    async def prepare_image(self, png):
        self.check()
        if self.route.adapter == 'qq_official':
            return await upload_qq_image(self.platform, self.route.group, png, self.check)
        return {'type': 'image', 'data': {'file': 'base64://'+base64.b64encode(png).decode()}}

    async def send(self, text, before_send, *, media=None):
        self.check()
        if self.route.adapter == 'aiocqhttp':
            from aiocqhttp import CQHttp
            client = self.platform.get_client()
            if type(client) is not CQHttp:
                raise ValueError('onebot_native_client_unavailable')
            login = await client.call_action(action='get_login_info', self_id=int(self.route.account))
            if not isinstance(login, dict) or str(login.get('user_id')) != self.route.account:
                raise ValueError('onebot_account_mismatch')
            await before_send()
            self.check()
            message = ([{'type': 'text', 'data': {'text': text}}] if text else [])
            if media:
                message.append(media)
            data = await client.call_action(action='send_group_msg', self_id=int(self.route.account),
                group_id=int(self.route.group), message=message)
            mid = data.get('message_id') if isinstance(data, dict) else None
            if isinstance(mid, bool) or not isinstance(mid, (str, int)) or not str(mid) or any(k in data for k in ('status', 'retcode')):
                raise ValueError('onebot_receipt_unconfirmed')
            return str(mid)
        from aiohttp import ClientTimeout
        from botpy.api import BotAPI
        client = official_client(self.platform)

        async def request(route, **kwargs):
            await client.http.check_session()
            if official_client(self.platform) is not client:
                raise ValueError('qq_client_changed')
            sequence = await before_send()
            self.check()
            route.is_sandbox = client.http.is_sandbox
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                # Allocated durably at the final send gate. Identical parts
                # still need distinct msg_id + msg_seq pairs in QQ.
                kwargs['json']['msg_seq'] = sequence
            # Local facade retains authentication/pooling while avoiding SDK retry.
            async with client.http._session.request(method=route.method, url=route.url,
                    headers=client.http._headers, timeout=ClientTimeout(total=15),
                    allow_redirects=False, **kwargs) as response:
                if response.status != 200:
                    raise ValueError('qq_receipt_unconfirmed')
                data = await response.json()
                if not isinstance(data, dict) or any(k in data for k in ('code', 'retcode', 'error')) or not native_id(data.get('id')):
                    raise ValueError('qq_receipt_unconfirmed')
                return data

        data = await BotAPI(SimpleNamespace(request=request)).post_group_message(
            group_openid=self.route.group, content=text, msg_type=7 if media else 0, msg_id=self.anchor,
            **({'media': media} if media else {}),
            msg_seq=10001+int(stable_id(self.route.key, self.anchor, text)[:8], 16) % 2147470000)
        return data['id']
