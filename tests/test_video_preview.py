"""Ordinary image previews keep captions, links and native receipts separate."""
from io import BytesIO
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image, ImageFont
from ambient.video_preview import render_preview, fetch_cover
import test_videos as videos_tests
import test_unified as unified
from test_ambient import Event


class PreviewRenderTests(unittest.TestCase):
    def test_long_titles_portrait_covers_and_upload_size(self):
        data = BytesIO()
        Image.new('RGB', (300, 800), '#b34467').save(data, 'PNG')
        video = {'title': 'A very long title '*30, 'author': 'Author '*20, 'duration_seconds': 125}
        with patch('ambient.video_preview.preview_font', side_effect=lambda size, bold=False: ImageFont.load_default(size=size)):
            result = render_preview(video, data.getvalue())
        self.assertLess(len(result), 2*1024*1024)
        with Image.open(BytesIO(result)) as image:
            self.assertEqual('JPEG', image.format)
            self.assertEqual((960, 756), image.size)
            self.assertTrue(all(abs(a-b) <= 3 for a,b in zip((179,68,103), image.getpixel((480,250)))))
            self.assertGreater(image.getpixel((20, 250))[0], 220)

    def test_oversized_and_invalid_images_are_rejected(self):
        with self.assertRaises(ValueError):
            render_preview({}, b'x'*(4*1024*1024+1))
        data = BytesIO()
        Image.new('RGB', (3000, 3000)).save(data, 'PNG')
        with self.assertRaisesRegex(ValueError, 'video_cover_dimensions'):
            render_preview({}, data.getvalue())


class PreviewFlowTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = unified.UnifiedTests.asyncTearDown
    answer = unified.UnifiedTests.answer
    interject = unified.UnifiedTests.interject

    async def asyncSetUp(self):
        await videos_tests.VideoFlowTests.asyncSetUp(self)
        self.config['groups'][0]['video_share_format'] = 'image'
        self.wires, self.uploads = [], []
        self.upload_error = None
        self.after_upload = None
        owner = self

        class Sender:
            def __init__(self, *args, **kwargs): pass
            def check(self): pass
            async def prepare_image(self, data):
                owner.uploads.append(data)
                if owner.upload_error:
                    raise owner.upload_error
                if owner.after_upload:
                    owner.after_upload()
                return {'file_info': 'uploaded-preview'}
            async def send(self, text, guard, *, media=None, ark=None):
                if owner.late_change:
                    owner.late_change()
                await guard()
                owner.sent.append(text)
                owner.wires.append({'text': text, 'media': media, 'ark': ark})
                if owner.error:
                    raise owner.error
                return 'receipt-'+str(len(owner.wires))

        for item in (patch('ambient.runtime.Transport', Sender), patch('ambient.interjection.Transport', Sender),
                     patch('ambient.video_preview.fetch_cover', AsyncMock(return_value=b'cover')),
                     patch('ambient.video_preview.render_preview', return_value=b'preview-jpeg')):
            self.patches.append(item)
            item.start()

    async def test_reply_and_interjection_send_caption_link_then_independent_image(self):
        first, second = videos_tests.FIRST, videos_tests.SECOND
        self.config['limits']['reply_segment_chars'] = 8
        await self.answer(unified.directed('发个视频', 'image-reply'), '企鹅上班了 [[bilibili:'+first+']]')
        await self.interject(Event('再来一个', mid='image-interject'), '猫也上班了 [[bilibili:'+second+']]')
        self.assertEqual(['企鹅上班了\nhttps://www.bilibili.com/video/'+first+'/', '',
                          '猫也上班了\nhttps://www.bilibili.com/video/'+second+'/', ''], self.sent)
        self.assertEqual([None, {'file_info':'uploaded-preview'}]*2, [w['media'] for w in self.wires])
        self.assertTrue(all(w['ark'] is None for w in self.wires))
        rows = [r for r in self.rt.windows[self.route.key].rows if r['self']]
        self.assertEqual(['reply', 'reply', 'interjection', 'interjection'], [r['source'] for r in rows])
        self.assertIn(first, rows[1]['text'])
        self.assertNotIn('表情包', rows[1]['text'])

    async def test_failed_image_is_not_recorded_but_confirmed_link_is_preserved(self):
        self.late_change = lambda: setattr(self, 'error', TimeoutError() if self.sent else None)
        await self.answer(unified.directed('发个视频'), '企鹅上班了 [[bilibili:'+videos_tests.FIRST+']]')
        self.assertEqual(2, len(self.wires))
        rows = [r for r in self.rt.windows[self.route.key].rows if r['self']]
        self.assertEqual([self.sent[0]], [r['text'].replace(' https://', '\nhttps://') for r in rows])
        self.assertEqual('partial_no_retry', self.rt.replies[self.route.key]['outcome'])

    async def test_upload_failure_and_settings_change_send_no_caption(self):
        self.upload_error = ValueError('upload_failed')
        await self.answer(unified.directed('发个视频', 'failed-upload'), '企鹅上班了 [[bilibili:'+videos_tests.FIRST+']]')
        self.assertEqual([], self.sent)
        self.assertEqual(set(), self.rt.stores[self.route.key].video_history())
        self.upload_error = None
        self.after_upload = lambda: self.config['groups'][0].update(video_share_format='link')
        await self.answer(unified.directed('发个视频', 'changed-config'), '企鹅上班了 [[bilibili:'+videos_tests.FIRST+']]')
        self.assertEqual([], self.sent)

    async def test_one_slot_does_not_upload_or_leave_orphan_caption(self):
        store = self.rt.stores[self.route.key]
        for _ in range(4):
            store.claim_qq_part('last-slot')
        await self.answer(unified.directed('发个视频', 'last-slot'), '企鹅上班了 [[bilibili:'+videos_tests.FIRST+']]')
        self.assertEqual([], self.sent)
        self.assertEqual([], self.uploads)

    async def test_preview_fetch_rejects_non_cdn_urls_before_network(self):
        for url in ('http://i0.hdslb.com/bfs/a.jpg', 'https://evil.test/a.jpg',
                    'https://i0.hdslb.com/bfs/a.jpg?redirect=secret'):
            with self.assertRaises(ValueError):
                await fetch_cover(url)
