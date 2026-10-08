import asyncio
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image as PillowImage
from test_ambient import Event, Plain, Part, context, request
from ambient.context import route_for
from ambient.delivery import send_reply_parts
from ambient.policy import Policy
from ambient.runtime import Runtime
from ambient.stickers import Bank, image_sources, normalize_image, parse_sticker, fetch_image
from ambient.sticker_history import import_history


def png(color='red', format='PNG'):
    out = BytesIO()
    PillowImage.new('RGB', (48, 48), color).save(out, format=format)
    return out.getvalue()


class BankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.route = route_for(context(), Event())
        self.bank = Bank(self.tmp.name, self.route, {'stickers_top': 2})

    def tearDown(self): self.tmp.cleanup()

    def insert(self, mid, color='red', at=None):
        return self.bank.ingest(mid, at or time.time(), 'live', png(color))

    def test_content_dedup_message_idempotence_and_restart(self):
        identity = self.insert('m1')
        self.bank.ingest('m1', time.time(), 'history', png(format='BMP'))
        self.insert('m2')
        self.bank = Bank(self.tmp.name, self.route, {})
        self.assertEqual([(identity, 2)], [(r['id'], r['count']) for r in self.bank.rows()])

    def test_ranked_pool_frequency_threshold_and_manual_block(self):
        red = self.insert('r1'); self.insert('r2')
        blue = self.insert('b1', 'blue'); self.insert('b2', 'blue'); self.insert('b3', 'blue')
        green = self.insert('g1', 'green')
        for identity in (red, blue, green): self.bank.review(identity, 'ready', '无语')
        self.assertEqual([blue, red], [r['id'] for r in self.bank.available()])
        self.bank.review(blue, 'blocked', '不合适')
        self.assertEqual([red], [r['id'] for r in self.bank.available()])
        self.assertTrue(self.bank.path(blue).exists())
        self.assertIsNone(self.bank.selected(blue[:16], [blue[:16]]))

    def test_group_isolation_and_only_offered_ids(self):
        identity = self.insert('m1'); self.insert('m2')
        self.bank.review(identity, 'ready', '笑哭')
        other = Bank(self.tmp.name, replace(self.route, group='other'), {})
        self.assertEqual([], other.available())
        self.assertIsNone(self.bank.selected(identity[:16], []))
        self.assertIsNotNone(self.bank.selected(identity[:16], [identity[:16]]))
        with self.assertRaises(ValueError): self.bank.path('../credentials')

    def test_bounded_candidate_files_and_expired_counts(self):
        for i in range(9): self.insert(str(i), (i*20, 5, 10))
        self.assertLessEqual(len(list(self.bank.root.glob('*.png'))), 6)
        with self.bank.db() as db: db.execute('UPDATE occurrences SET at=1')
        self.bank.trim()
        self.assertEqual(0, self.bank.status()['observations'])
        self.assertEqual([], list(self.bank.root.glob('*.png')))

    def test_reject_nonimages_and_no_metadata_retained_in_png(self):
        with self.assertRaises(Exception): normalize_image(b'not an image')
        _, data, animated = normalize_image(png())
        self.assertFalse(animated)
        self.assertEqual({}, PillowImage.open(BytesIO(data)).info)

    def test_extracts_only_top_level_images(self):
        Image = type('Image', (), {})
        Reply = type('Reply', (), {})
        image, quote = Image(), Reply()
        image.url='https://gchat.qpic.cn/a'; quote.message=[image]
        self.assertEqual([image.url], list(image_sources([quote, image, Plain('hello')])) )

    def test_markers_never_select_an_unoffered_or_arbitrary_file(self):
        self.assertEqual(('笑死', 'abc'), parse_sticker('笑死[[sticker:abc]][[sticker:../../x]]', ['abc']))
        self.assertEqual(('', ''), parse_sticker('[[sticker:unknown]]', []))


class StickerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = context()
        self.route = route_for(self.ctx, Event())
        self.cfg = {'groups': [{'umo': self.route.umo, 'capture_only': False, 'interject_enabled': True,
                              'stickers_enabled': True}], 'limits': {'reply_segment_interval_ms': 0}}
        self.rt = Runtime(self.tmp.name, self.ctx, self.cfg)
        await self.rt.ensure(self.route)
        self.bank = self.rt.stickers.bank(self.route)
        self.identity = self.bank.ingest('img1', time.time(), 'live', png())
        self.bank.ingest('img2', time.time(), 'live', png())
        self.bank.review(self.identity, 'ready', '笑哭猫猫')
        self.short = self.identity[:16]

    async def asyncTearDown(self):
        await self.rt.close()
        self.tmp.cleanup()

    async def test_both_reply_paths_share_catalog_and_native_image_records(self):
        sent, uploads = [], []
        class Sender:
            def __init__(self, *args): pass
            def check(self): pass
            async def prepare_image(self, data): uploads.append(data); return {'file_info': 'test'}
            async def send(self, text, guard, *, media=None):
                await guard(); sent.append((text, media)); return 'receipt'+str(len(sent))
        with patch('ambient.runtime.Transport', Sender), patch('ambient.interjection.Transport', Sender):
            event = Event('给个表情', mid='asked'); event.is_at_or_wake_command=True
            req = request()
            await self.rt.inject(event, req, Part)
            pack = json.loads(req.extra_user_content_parts[-1].text)['ambient_data']
            self.assertEqual(self.short, pack['sticker_choices'][0]['id'])
            response = NS(completion_text='[[sticker:'+self.short+']]')
            await self.rt.response(event, response)
            await event.send(NS(chain=[Plain(response.completion_text)]))
            self.assertEqual('', sent[0][0]); self.assertIsNotNone(sent[0][1])
            self.assertIn('表情包：笑哭猫猫', self.rt.windows[self.route.key].rows[-1]['text'])
            await event.send(NS(chain=[Plain(response.completion_text)]))
            self.assertEqual(1, len(sent))
            incoming=Event('笑死我了', mid='ambient')
            await self.rt.observe(incoming)
            state=self.rt.interjections.states[self.route.key]; state.suppressed_until=0
            self.ctx.llm_generate.return_value=NS(completion_text=json.dumps({'reply':True,'text':'[[sticker:'+self.short+']]'}))
            await self.rt.interjections.run(self.route,state)
            self.assertEqual(2,len(sent))
            self.assertIn('sticker_choices', self.ctx.llm_generate.call_args.kwargs['prompt'])

    async def test_attachment_uses_last_text_slot_and_uncertain_send_does_not_retry(self):
        sent, recorded = [], []
        async def send(text, guard, **kwargs):
            await guard(); sent.append((text, kwargs));
            if len(sent)==2: raise TimeoutError()
            return 'ok'
        transport=NS(prepare_image=AsyncMock(return_value={'file_info':'x'}), send=send)
        with self.assertRaises(TimeoutError):
            await send_reply_parts(transport, '一。\n\n二。\n\n三。', Policy(reply_segment_interval_ms=0),
                AsyncMock(), AsyncMock(side_effect=lambda text,mid:recorded.append(text)), max_parts=2,
                sticker={'data':png(),'caption':'笑哭'})
        self.assertEqual(['一。'],recorded)
        self.assertEqual({'media':{'file_info':'x'}},sent[-1][1])
        self.assertEqual('二。\n\n三。',sent[-1][0])

    async def test_upload_failure_falls_back_only_before_any_message_send(self):
        transport=NS(prepare_image=AsyncMock(side_effect=TimeoutError()), send=AsyncMock(return_value='ok'))
        recorded=AsyncMock()
        await send_reply_parts(transport,'正文',Policy(),AsyncMock(),recorded,sticker={'data':png(),'caption':'笑'})
        self.assertEqual(('正文','ok'),recorded.call_args.args)
        transport.send.reset_mock()
        with self.assertRaises(TimeoutError):
            await send_reply_parts(transport,'',Policy(),AsyncMock(),recorded,sticker={'data':png(),'caption':'笑'})
        transport.send.assert_not_awaited()

    async def test_history_empty_and_scope_fail_closed(self):
        self.ctx.message_history_manager=NS(get=AsyncMock(return_value=[]))
        self.ctx.conversation_manager=NS(get_conversations=AsyncMock(return_value=[NS(user_id='other')]))
        report=await import_history(self.rt.stickers,self.route)
        self.assertEqual('complete',report['state']); self.assertEqual(0,report['images'])
        self.ctx.conversation_manager.get_conversations.assert_awaited_once_with(unified_msg_origin=self.route.umo)
        for call in self.ctx.message_history_manager.get.await_args_list:
            self.assertEqual(self.route.platform,call.kwargs['platform_id'])
            self.assertIn(call.kwargs['user_id'],(self.route.group,self.route.umo))

    async def test_history_forks_and_reruns_cannot_inflate_frequency(self):
        parts=[{'type':'image_url','image_url':{'url':'https://gchat.qpic.cn/a'}}]
        conv=NS(user_id=self.route.umo,history=json.dumps([{'role':'user','content':parts}]),created_at=time.time())
        self.ctx.conversation_manager=NS(get_conversations=AsyncMock(return_value=[conv,conv]))
        with patch('ambient.stickers.fetch_image',AsyncMock(return_value=png('blue'))):
            await import_history(self.rt.stickers,self.route); await import_history(self.rt.stickers,self.route)
        blue=normalize_image(png('blue'))[0]
        self.assertEqual(1,next(r['count'] for r in self.bank.rows() if r['id']==blue))

    async def test_photo_classifier_and_failure_do_not_auto_accept(self):
        self.bank.review(self.identity,'pending')
        self.cfg['groups'][0]['stickers_caption_provider']='vision'
        self.ctx.llm_generate.return_value=NS(completion_text='{"meme":false,"caption":"聊天截图"}')
        await self.rt.stickers.classify(self.route,self.bank.rows()[0])
        self.assertEqual('blocked',self.bank.rows()[0]['state'])
        self.bank.review(self.identity,'pending')
        self.ctx.llm_generate.return_value=NS(completion_text='broken')
        await self.rt.stickers.classify(self.route,self.bank.rows()[0])
        self.assertEqual('pending',self.bank.rows()[0]['state'])
        self.assertGreater(self.bank.rows()[0]['retry'],time.time())

    async def test_collector_excludes_self_quoted_and_other_bot_images(self):
        Image=type('Image',(),{}); img=Image(); img.url='https://gchat.qpic.cn/a'
        event=Event(mid='photo'); event.message_obj.message=[img]
        self.cfg['groups'][0]['style_excluded_senders']=[event.get_sender_id()]
        await self.rt.observe(event)
        self.assertEqual(0,self.rt.stickers.queue.qsize())
        self.cfg['groups'][0]['style_excluded_senders']=[]
        event.message_obj.message_id='photo2'
        await self.rt.observe(event)
        self.assertEqual(1,self.rt.stickers.queue.qsize())

    async def test_network_and_local_sources_are_restricted(self):
        for source in ('http://gchat.qpic.cn/x','https://127.0.0.1/x','https://qq.com/x',
                       'https://user:pass@gchat.qpic.cn/x','file:///etc/passwd'):
            with self.assertRaises(ValueError): await fetch_image(source)
        path=Path(self.tmp.name)/'test.png'; path.write_bytes(png())
        self.assertEqual(png(),await fetch_image(str(path),(self.tmp.name,)))

    async def test_archived_embedded_images_and_redacted_failure_counts(self):
        source='data:image/png;base64,'+base64.b64encode(png('blue')).decode()
        self.assertEqual(png('blue'),await fetch_image(source))
        with self.assertRaises(ValueError): await fetch_image('data:image/svg+xml;base64,abcd')
        parts=[{'type':'image_url','image_url':{'url':source}},
               {'type':'image_url','image_url':{'url':'https://example.com/private?secret=never-report'}}]
        self.ctx.conversation_manager=NS(get_conversations=AsyncMock(return_value=[
            NS(user_id=self.route.umo,history=json.dumps([{'role':'user','content':parts}]),created_at=time.time())]))
        report=await import_history(self.rt.stickers,self.route)
        self.assertEqual(1,report['downloaded'])
        self.assertEqual({'image_host_not_allowed':1},report['failures'])
        self.assertNotIn('secret',json.dumps(report))


if __name__ == '__main__': unittest.main()
