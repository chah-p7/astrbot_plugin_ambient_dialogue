from __future__ import annotations

import asyncio
import json
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart

from .ambient.runtime import Runtime


class AmbientDialogue(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.runtime = Runtime(StarTools.get_data_dir('astrbot_plugin_ambient_dialogue'), context, config)
        self.worker = None

    async def initialize(self):
        await self.runtime.initialize()
        self.worker = asyncio.create_task(self._maintenance())
        logger.info('Ambient Dialogue ready; groups=%s', len(self.runtime.config.get('groups', [])))

    async def _maintenance(self):
        while True:
            try:
                await self.runtime.maintain()
                # Bounded operational counters only, never chat text or credentials.
                path = self.runtime.root/'status.json'
                data = json.dumps(self.runtime.status(), ensure_ascii=False)
                await asyncio.to_thread(path.write_text, data, encoding='utf-8')
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.runtime.errors += 1
                logger.warning('Ambient maintenance: %s', type(exc).__name__)
            await asyncio.sleep(5)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=1000)
    async def capture(self, event: AstrMessageEvent):
        try:
            await self.runtime.observe(event)
        except Exception as exc:
            self.runtime.errors += 1
            logger.warning('Ambient capture skipped: %s', type(exc).__name__)

    @filter.on_llm_request(priority=-1000)
    async def inject_context(self, event: AstrMessageEvent, req):
        try:
            await self.runtime.inject(event, req, TextPart)
        except Exception as exc:
            self.runtime.errors += 1
            logger.warning('Ambient context skipped: %s', type(exc).__name__)

    @filter.command('轻聊', priority=10000)
    async def memory_command(self, event: AstrMessageEvent):
        try:
            text = await self.runtime.command(event)
        except ValueError as exc:
            text = str(exc) if str(exc).startswith(('本群', '记忆', 'kind', '缺少')) else '操作未保存，请检查参数。'
        except Exception:
            self.runtime.errors += 1
            text = '记忆暂不可用，操作未保存。'
        if text is not None:
            event.stop_event()
            yield event.plain_result(text)

    async def terminate(self):
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None
        await self.runtime.close()
