"""A native event may clean its files before background image decoding starts."""
import os
import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

import test_vision as vision_tests


@unittest.skipUnless(os.environ.get('AMBIENT_NATIVE_TEST'),'requires native event cleanup')
class NativeImageLifetimeTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = vision_tests.VisionTests.asyncSetUp
    asyncTearDown = vision_tests.VisionTests.asyncTearDown
    event = vision_tests.VisionTests.event
    complete = vision_tests.VisionTests.complete

    async def test_native_cleanup_cannot_delete_image_before_background_read(self):
        import astrbot.api
        from astrbot.core.platform.astr_message_event import AstrMessageEvent
        self.fetch.stop()
        path = Path(self.tmp.name)/'native-owned.png'
        path.write_bytes(vision_tests.png())
        event = self.event(sources=(str(path),))
        event._temporary_local_files = [str(path)]
        with patch.object(self.rt.stickers,'local_roots',return_value=(Path(self.tmp.name),)):
            await self.rt.observe(event)
            AstrMessageEvent.cleanup_temporary_local_files(event)
            self.assertFalse(path.exists())
            await self.complete()
        notes = self.rt.windows[self.route.key].image_notes['photo']
        self.assertEqual('ready',notes[0]['state'],notes)
        self.ctx.llm_generate.assert_awaited_once()
        self.assertEqual(0,self.rt.vision.buffered_bytes)

    async def test_capture_does_not_broaden_local_file_access(self):
        self.fetch.stop()
        path=Path(self.tmp.name)/'outside.png';path.write_bytes(vision_tests.png())
        with patch.object(self.rt.stickers,'local_roots',return_value=(Path(self.tmp.name)/'cache',)):
            await self.rt.observe(self.event(sources=(str(path),)))
            await self.complete()
        note=self.rt.windows[self.route.key].image_notes['photo'][0]
        self.assertEqual('image_local_source_unavailable',note['reason'])
        self.ctx.llm_generate.assert_not_awaited()
        self.assertEqual(0,self.rt.vision.buffered_bytes)

    async def test_buffers_are_bounded_and_released_on_cancel(self):
        self.fetch.stop()
        path=Path(self.tmp.name)/'image.png';data=vision_tests.png();path.write_bytes(data)
        gate=asyncio.Event()
        async def pending(**kwargs): await gate.wait()
        self.ctx.llm_generate.side_effect=pending
        with patch.object(self.rt.stickers,'local_roots',return_value=(Path(self.tmp.name),)), \
                patch('ambient.vision.MAX_BUFFERED_IMAGES',len(data)):
            await self.rt.observe(self.event(mid='one',sources=(str(path),)))
            self.assertEqual(len(data),self.rt.vision.buffered_bytes)
            await self.rt.observe(self.event(mid='two',sources=(str(path),)))
            note=self.rt.windows[self.route.key].image_notes['two'][0]
            self.assertEqual('image_buffer_full',note['reason'])
            await self.rt.vision.close()
        self.assertEqual(0,self.rt.vision.buffered_bytes)

    async def test_same_event_does_not_capture_twice_and_remote_download_stays_background(self):
        event=self.event()
        await self.rt.observe(event)
        row=self.rt.windows[self.route.key].rows[-1]
        await self.rt.vision.observe(self.route,event,row)
        self.assertEqual(1,len(self.rt.vision.tasks))
        self.assertEqual(0,self.rt.vision.buffered_bytes)
        await self.complete()
        self.ctx.llm_generate.assert_awaited_once()


if __name__=='__main__': unittest.main()
