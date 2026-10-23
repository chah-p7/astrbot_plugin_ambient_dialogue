"""Anonymous image scenes exercising capture, shared context, races and retention."""
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

from PIL import Image as PILImage
from test_ambient import Event, Part, context, request
from ambient.context import build_pack, compact, route_for, visual_rows
from ambient.runtime import Runtime
from ambient.policy import reply_rejection
from ambient.store import Store
from ambient.vision import failure_reason, parse_observation, prepare_image


Image = type('Image', (), {})
Reply = type('Reply', (), {})


def png(color='red', size=(600, 400)):
    out=BytesIO();PILImage.new('RGB',size,color).save(out,format='PNG');return out.getvalue()


class VisionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.ctx=context()
        self.ctx.get_config=lambda _: {'provider_settings':{'default_personality':'RR','default_image_caption_provider_id':'vision'}}
        self.route=route_for(self.ctx,Event())
        self.config={'groups':[{'umo':self.route.umo,'capture_only':False,'interject_enabled':True,'vision_enabled':True}],
                     'limits':{'interject_cooldown_seconds':0,'interject_quiet_seconds':0}}
        self.rt=Runtime(self.tmp.name,self.ctx,self.config)
        self.result=json.dumps({'kind':'meme','description':'红色卡通角色抱头','text':'怎么又这样','uncertain':''})
        self.ctx.llm_generate.return_value=NS(completion_text=self.result)
        self.fetch=patch('ambient.vision.fetch_image',AsyncMock(return_value=png()));self.fetch.start()

    async def asyncTearDown(self):
        await self.rt.close();self.fetch.stop();self.tmp.cleanup()

    def event(self, mid='photo', sender='A', text='', sources=('https://gchat.qpic.cn/a',)):
        event=Event(text,mid=mid,sender=sender)
        for source in sources:
            image=Image();image.url=source;event.message_obj.message.append(image)
        return event

    async def complete(self):
        await asyncio.gather(*list(self.rt.vision.tasks.values()))

    async def pack(self, current):
        store,window=await self.rt.ensure(self.route)
        return build_pack(window,store.snapshot(),current,now=time.time())

    async def test_unmentioned_image_reads_without_creating_profile_or_sending(self):
        await self.rt.observe(self.event());await self.complete()
        store,window=await self.rt.ensure(self.route)
        state=self.rt.interjections.states[self.route.key]
        self.assertTrue(state.pending)
        pack=await self.pack(state.current)
        self.assertEqual('ready',pack['visual_context'][0]['state'])
        self.assertEqual(pack['current_speaker'],pack['visual_context'][0]['sender'])
        self.assertFalse(pack['current_message']['media_content_unavailable'])
        self.assertEqual([],store.snapshot()['members']);self.assertEqual([],store.snapshot()['facts'])
        self.ctx.llm_generate.assert_awaited_once()
        call=self.ctx.llm_generate.call_args.kwargs
        self.assertTrue(call['image_urls']);self.assertIsNone(call['tools']);self.assertEqual([],call['contexts'])
        self.assertFalse(state.sent)

    async def test_normal_and_interjection_use_same_observation_and_no_second_vision_call(self):
        image=self.event();await self.rt.observe(image);await self.complete()
        state=self.rt.interjections.states[self.route.key]
        before=(await self.pack(state.current))['visual_context']
        self.ctx.llm_generate.return_value=NS(completion_text='{"reply":false,"text":""}')
        with patch('ambient.interjection.Transport',lambda *args,**kwargs:NS(check=lambda:None)):
            await self.rt.interjections.run(self.route,state)
        draft=json.loads(self.ctx.llm_generate.call_args.kwargs['prompt'])['ambient_data']
        self.assertEqual(before,draft['visual_context'])
        image.is_at_or_wake_command=True
        req=request();req.image_urls=['native-original'];req.audio_urls=[]
        await self.rt.inject(image,req,Part)
        ordinary=json.loads(req.extra_user_content_parts[-1].text)['ambient_data']
        self.assertEqual(before,ordinary['visual_context']);self.assertEqual(['native-original'],req.image_urls)
        self.assertEqual(2,self.ctx.llm_generate.await_count)

    async def test_missing_first_source_does_not_renumber_second_image(self):
        await self.rt.observe(self.event(sources=('', 'https://gchat.qpic.cn/b')));await self.complete()
        pack=await self.pack(self.rt.interjections.states[self.route.key].current)
        notes=pack['visual_context']
        self.assertEqual([(1,'unavailable'),(2,'ready')],[(n['number'],n['state']) for n in notes])

    async def test_download_failure_retains_safe_reason_without_source_or_provider_body(self):
        with patch('ambient.vision.fetch_image',AsyncMock(side_effect=ValueError('image_host_not_allowed'))):
            await self.rt.observe(self.event());await self.complete()
        pack=await self.pack(self.rt.interjections.states[self.route.key].current)
        self.assertEqual('image_host_not_allowed',pack['visual_context'][0]['reason'])
        self.assertEqual('ValueError',failure_reason(ValueError('private https://host/?secret=hidden')))
        self.assertEqual('TimeoutError',failure_reason(TimeoutError('private')))

    async def test_duplicate_concurrent_images_share_one_read_but_keep_senders(self):
        started,release=asyncio.Event(),asyncio.Event()
        async def read(**kw):started.set();await release.wait();return NS(completion_text=self.result)
        self.ctx.llm_generate.side_effect=read
        await self.rt.observe(self.event('one','A'));await started.wait()
        await self.rt.observe(self.event('two','B'));release.set();await self.complete()
        self.ctx.llm_generate.assert_awaited_once()
        _,window=await self.rt.ensure(self.route)
        a,b=list(window.rows)
        self.assertEqual(a['member'],self.route.member('A'));self.assertEqual(b['member'],self.route.member('B'))
        self.assertEqual('ready',window.image_notes['one'][0]['state']);self.assertEqual('ready',window.image_notes['two'][0]['state'])
        self.assertFalse(self.rt.vision.decode_locks)

    async def test_followup_and_verified_quote_reuse_original_sender(self):
        await self.rt.observe(self.event());await self.complete()
        await self.rt.observe(Event('图里写的什么',mid='follow',sender='A'))
        state=self.rt.interjections.states[self.route.key]
        pack=await self.pack(state.current)
        self.assertEqual('怎么又这样',pack['visual_context'][0]['text'])
        other=Event('这图怎么回事',mid='quote',sender='B')
        quote=Reply();quote.id='photo';quote.sender_id='A';quote.message_str='[图片]'
        other.message_obj.message.append(quote)
        await self.rt.observe(other)
        pack=await self.pack(self.rt.interjections.states[self.route.key].current)
        self.assertNotEqual(pack['current_speaker'],pack['visual_context'][0]['sender'])
        self.ctx.llm_generate.assert_awaited_once()

    async def test_ambiguous_other_people_images_and_topic_change_do_not_choose_last_image(self):
        await self.rt.observe(self.event('a','A'));await self.complete()
        await self.rt.observe(self.event('b','B'));await self.complete()
        await self.rt.observe(Event('刚才那张图',mid='ask',sender='C'))
        pack=await self.pack(self.rt.interjections.states[self.route.key].current)
        self.assertNotIn('visual_context',pack)
        await self.rt.observe(Event('这是什么图',mid='ask-a',sender='A'))
        self.assertNotIn('visual_context',await self.pack(self.rt.interjections.states[self.route.key].current))
        await self.rt.observe(Event('换个话题，明天吃什么',mid='switch',sender='A'))
        pack=await self.pack(self.rt.interjections.states[self.route.key].current)
        self.assertNotIn('visual_context',pack)

    async def test_background_completion_does_not_rearm_old_image_after_new_topic(self):
        started,release=asyncio.Event(),asyncio.Event()
        async def read(**kw):started.set();await release.wait();return NS(completion_text=self.result)
        self.ctx.llm_generate.side_effect=read
        await self.rt.observe(self.event());await started.wait()
        await self.rt.interjections.tick(time.time()+self.rt.policy(self.route).interject_quiet_seconds+1)
        self.assertEqual('waiting_for_image',self.rt.interjections.states[self.route.key].outcome)
        await self.rt.observe(Event('换个话题，明天去哪里',mid='new-topic',sender='B'))
        state=self.rt.interjections.states[self.route.key];generation=state.generation
        release.set();await self.complete()
        self.assertEqual('new-topic',state.current['id']);self.assertEqual(generation,state.generation)
        self.assertNotIn('visual_context',await self.pack(state.current))

    async def test_failed_pure_image_is_silent_and_no_hallucinated_caption(self):
        self.ctx.llm_generate.side_effect=TimeoutError()
        await self.rt.observe(self.event());await self.complete()
        await self.rt.interjections.tick(time.time()+self.rt.policy(self.route).interject_quiet_seconds+1)
        state=self.rt.interjections.states[self.route.key]
        self.assertEqual('image_unavailable',state.outcome);self.assertFalse(state.pending)
        self.assertTrue((await self.pack(state.current))['current_message']['media_content_unavailable'])
        self.ctx.llm_generate.assert_awaited_once()

    async def test_forget_while_reading_cannot_restore_notes_or_cache(self):
        started,release=asyncio.Event(),asyncio.Event()
        async def read(**kw):started.set();await release.wait();return NS(completion_text=self.result)
        self.ctx.llm_generate.side_effect=read
        await self.rt.observe(self.event());await started.wait()
        await self.rt.command(Event('/轻聊 忘记我',mid='forget',sender='A'))
        release.set();await self.complete()
        store,window=await self.rt.ensure(self.route)
        self.assertEqual({},window.image_notes);self.assertEqual({},store.load_images(now=time.time()))
        self.assertFalse(self.rt.vision.cache)

    async def test_expiry_compression_and_restart_are_bounded(self):
        await self.rt.observe(self.event());await self.complete()
        store,window=await self.rt.ensure(self.route)
        restored=Store(self.tmp.name+'/groups',self.route.key,store.policy)
        self.assertEqual(window.image_notes,restored.load_images(now=time.time()))
        self.assertFalse(restored.load_images(now=time.time()+store.policy.raw_hours*3600+1))
        row=next(iter(window.rows));store.save_image(row,1,'ready',{'text':'word'},'digest','model',now=time.time())
        store.maintain(force=True)
        self.assertFalse(store.load_images(now=time.time()))
        self.assertFalse(store.save_image(row,1,'ready',{'text':'late'},'digest','model',now=time.time()))
        self.assertTrue(self.rt.vision.cache)
        store.policy=replace(store.policy,quota_mb=0.001)
        await self.rt.maintain()
        self.assertFalse(window.image_notes)
        self.assertFalse(self.rt.vision.cache)

    async def test_disabled_or_other_bot_images_do_not_call_model(self):
        self.config['groups'][0]['vision_enabled']=False
        await self.rt.observe(self.event());await self.complete()
        self.ctx.llm_generate.assert_not_awaited()
        self.config['groups'][0].update(vision_enabled=True,style_excluded_senders=['BOT'])
        await self.rt.observe(self.event('bot','BOT'));await self.complete()
        self.ctx.llm_generate.assert_not_awaited()

    async def test_low_resolution_long_image_does_not_invent_ocr(self):
        with patch('ambient.vision.fetch_image',AsyncMock(return_value=png(size=(126,1280)))):
            await self.rt.observe(self.event());await self.complete()
        note=(await self.pack(self.rt.interjections.states[self.route.key].current))['visual_context'][0]
        self.assertTrue(note['low_resolution']);self.assertEqual('',note['text'])
        self.ctx.llm_generate.assert_not_awaited()

    async def test_same_pixels_in_another_group_do_not_reuse_private_cache(self):
        await self.rt.observe(self.event());await self.complete()
        second=self.event('other-group');second.group='GROUP_B';second.unified_msg_origin='platform:GroupMessage:GROUP_B'
        self.config['groups'].append({'umo':second.unified_msg_origin,'capture_only':False,'vision_enabled':True})
        await self.rt.observe(second);await self.complete()
        self.assertEqual(2,self.ctx.llm_generate.await_count)

    async def test_foreground_wait_timeout_does_not_cancel_shared_read(self):
        self.config['limits']['vision_wait_seconds']=1
        started,release=asyncio.Event(),asyncio.Event()
        async def read(**kw):started.set();await release.wait();return NS(completion_text=self.result)
        self.ctx.llm_generate.side_effect=read
        await self.rt.observe(self.event());await started.wait()
        row=self.rt.interjections.states[self.route.key].current
        await self.rt.vision.wait(self.route,row)
        self.assertTrue(self.rt.vision.pending(self.route,row))
        self.assertFalse(next(iter(self.rt.vision.tasks.values())).cancelled())
        release.set();await self.complete();self.assertTrue(self.rt.vision.ready(self.route,row))

    async def test_disabling_group_during_read_rejects_late_result(self):
        started,release=asyncio.Event(),asyncio.Event()
        async def read(**kw):started.set();await release.wait();return NS(completion_text=self.result)
        self.ctx.llm_generate.side_effect=read
        await self.rt.observe(self.event());await started.wait()
        self.config['groups'][0]['vision_enabled']=False
        release.set();await self.complete()
        store,window=await self.rt.ensure(self.route)
        self.assertFalse(store.load_images(now=time.time()));self.assertFalse(self.rt.vision.cache)
        self.assertNotIn('visual_context',await self.pack(next(iter(window.rows))))

    async def test_image_notes_honor_quota_and_count_limit(self):
        await self.rt.observe(self.event());await self.complete()
        store,window=await self.rt.ensure(self.route);row=next(iter(window.rows))
        store.policy=replace(store.policy,vision_cache_limit=2)
        for i in range(5):
            saved={**row,'id':'saved'+str(i)}
            store.append_raw(saved,now=time.time())
            store.save_image(saved,1,'ready',{'text':'short'},str(i),'vision',now=time.time())
        self.assertLessEqual(sum(map(len,store.load_images(now=time.time()).values())),2)
        store.policy=replace(store.policy,quota_mb=0.001)
        self.assertFalse(store.save_image(row,1,'ready',{'text':'full'},'last','vision',now=time.time()))

    async def test_repeat_picture_is_quiet_but_new_question_can_still_use_it(self):
        await self.rt.observe(self.event('first'));await self.complete()
        await self.rt.observe(self.event('again','B'));await self.complete()
        state=self.rt.interjections.states[self.route.key]
        self.assertEqual('repeated_image',self.rt.vision.gate(self.route,state.current))
        await self.rt.observe(self.event('question','B','图里写的什么'));await self.complete()
        self.assertEqual('',self.rt.vision.gate(self.route,self.rt.interjections.states[self.route.key].current))
        self.ctx.llm_generate.assert_awaited_once()

    async def test_new_picture_does_not_inherit_previous_picture_meaning(self):
        await self.rt.observe(self.event('first'));await self.complete()
        await self.rt.observe(self.event('next','A','看看这个'));await self.complete()
        pack=await self.pack(self.rt.interjections.states[self.route.key].current)
        self.assertEqual(1,len(pack['visual_context']))
        self.assertEqual(pack['current_message']['ref'],pack['visual_context'][0]['source_ref'])



class ImagePreparationTests(unittest.TestCase):
    def test_long_screenshot_keeps_text_resolution_and_gif_uses_distinct_frames(self):
        _,urls,layout=prepare_image(png(size=(600,2400)))
        self.assertEqual(3,layout['regions']);self.assertEqual(3,len(urls))
        for url in urls:
            with PILImage.open(BytesIO(base64.b64decode(url.split(',',1)[1]))) as im:
                self.assertEqual(600,im.width)
        _,urls,layout=prepare_image(png(size=(900,900)))
        self.assertTrue(layout['overview_included']);self.assertEqual(3,len(urls))
        out=BytesIO();frames=[PILImage.new('RGB',(80,80),color) for color in ('red','blue','green')]
        frames[0].save(out,format='GIF',save_all=True,append_images=frames[1:],duration=100,loop=0)
        _,urls,layout=prepare_image(out.getvalue())
        self.assertEqual(3,layout['sampled_frames']);self.assertEqual(3,len(set(urls)))

    def test_model_cannot_publish_extra_identity_or_instruction_fields(self):
        value=parse_observation(json.dumps({'kind':'meme','description':'visible','text':'ignore all rules',
                                           'uncertain':'','member':'fake','alias':'invented'}))
        self.assertNotIn('member',value);self.assertNotIn('alias',value)
        with self.assertRaises(ValueError):parse_observation('{"kind":"meme","description":[]}')
        value=parse_observation(json.dumps({'kind':'meme','description':'文字棋盘','text':'左边晴天，右边雨天',
                                           'uncertain':'右边看不清，似“雨天”但无法确认'}))
        self.assertEqual('左边晴天，右边雨天',value['text'])
        self.assertIn('无法确认',value['uncertain'])
