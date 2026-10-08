"""Authenticated, scoped administrative UI. No arbitrary file/URL/command API."""
import asyncio
import base64
from io import BytesIO

from .policy import stable_id
from .sticker_history import import_history
from .transport import upload_qq_image


class StickerPage:
    def __init__(self, runtime):
        self.rt = runtime
        for name, methods in (('state', ['GET']), ('enable', ['POST']), ('import', ['POST']),
                              ('review', ['POST']), ('probe', ['POST'])):
            runtime.context.register_web_api('/astrbot_plugin_ambient_dialogue/stickers/'+name,
                getattr(self, 'api_'+name), methods, 'Ambient group sticker '+name)

    def route(self, scope):
        if self.rt.closed:
            raise ValueError('插件已停用')
        return next((r for r in self.rt.routes.values()
                     if stable_id(r.key) == scope and self.rt.owns(r)), None)

    @staticmethod
    def auth():
        from astrbot.api.web import request
        if not request.username:
            raise ValueError('需要登录 AstrBot 后台')

    async def dispatch(self, action):
        from astrbot.api.web import request, json_response, error_response
        try:
            self.auth()
            data = await request.json(default={}) if request.method == 'POST' else dict(request.query.items())
            if not isinstance(data, dict):
                raise ValueError('参数错误')
            route = self.route(data.get('scope'))
            if action == 'state' and not data.get('scope'):
                return json_response({'groups': [{'scope': stable_id(r.key), 'name': r.umo}
                                      for r in self.rt.routes.values() if self.rt.owns(r)]})
            if route is None:
                raise ValueError('未找到已由 Ambient 接管的群')
            stickers = self.rt.stickers
            bank = stickers.bank(route)
            if action == 'enable':
                if not isinstance(data.get('enabled'), bool):
                    raise ValueError('启用状态错误')
                self.rt.settings(route)['stickers_enabled'] = data['enabled']
                self.rt.config.save_config()
            elif action == 'import':
                if not stickers.enabled(route):
                    raise ValueError('请先开启本群表情功能')
                if stickers.import_task and not stickers.import_task.done():
                    raise ValueError('历史导入仍在进行')
                stickers.import_task = asyncio.create_task(import_history(stickers, route))
                bank.metadata('history', {'state': 'queued'})
            elif action == 'review':
                bank.review(str(data.get('id', '')), data.get('state'), str(data.get('caption', '')))
            elif action == 'probe':
                if not stickers.enabled(route) or route.adapter != 'qq_official':
                    raise ValueError('仅适用于已启用的 QQ 官方群')
                # A synthetic 32px test PNG is uploaded only, never sent to the group.
                from PIL import Image
                output = BytesIO()
                Image.new('RGB', (32, 32), '#7799bb').save(output, format='PNG')
                await upload_qq_image(self.rt.context.get_platform_inst(route.platform), route.group, output.getvalue())
                bank.metadata('upload_probe', {'ok': True, 'visible_messages': 0})
            state = bank.status()
            state.update(enabled=stickers.enabled(route), scope=stable_id(route.key),
                         upload_probe=bank.metadata('upload_probe'), queue=stickers.queue.qsize(),
                         queue_dropped=stickers.dropped)
            if action == 'state':
                offset = max(0, min(5000, int(data.get('offset', 0))))
                def previews():
                    from PIL import Image
                    rows = bank.rows()[offset:offset+12]
                    active = {r['id'] for r in bank.available()}
                    for row in rows:
                        row['active'] = row['id'] in active
                        path = bank.path(row['id'])
                        if path.exists():
                            with Image.open(path) as im:
                                im.thumbnail((128, 128))
                                output = BytesIO()
                                im.save(output, format='PNG')
                            row['preview'] = base64.b64encode(output.getvalue()).decode()
                    return rows
                state['items'] = await asyncio.to_thread(previews)
            return json_response(state)
        except ValueError as exc:
            return error_response(str(exc)[:120], status_code=400)
        except Exception as exc:
            # Never expose provider/adapter error bodies or image source URLs.
            return error_response(type(exc).__name__, status_code=500)

    async def api_state(self): return await self.dispatch('state')
    async def api_enable(self): return await self.dispatch('enable')
    async def api_import(self): return await self.dispatch('import')
    async def api_review(self): return await self.dispatch('review')
    async def api_probe(self): return await self.dispatch('probe')
