"""Proactive video jobs use shared history without borrowing a user reply anchor."""
import asyncio
import json
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from test_ambient import Event
import test_unified as unified
import test_videos as video_tests
from ambient.videos import AUTO_VIDEO_RULE
from ambient.store import Store


class AutoVideoTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = unified.UnifiedTests.asyncTearDown

    async def asyncSetUp(self):
        await video_tests.VideoFlowTests.asyncSetUp(self)
        self.config['groups'][0].update(video_auto_share_enabled=True,video_auto_share_interval_seconds=1800)
        self.store=self.rt.stores[self.route.key]
        owner=self
        class Sender:
            def __init__(self,context,event,route,*,proactive=False):
                owner.assertTrue(proactive)
                owner.assertIsNone(event)
            def check(self): pass
            async def send(self,text,guard):
                await guard()
                owner.sent.append(text)
                if owner.error: raise owner.error
                return 'receipt-'+str(len(owner.sent))
        for item in [patch('ambient.video_auto.Transport',Sender),
                     patch('ambient.video_auto.task_event',side_effect=lambda route:Event())]:
            item.start();self.patches.append(item)

    async def run_auto(self,decision=None):
        self.ctx.llm_generate.return_value=NS(completion_text=json.dumps(decision or {
            'reply':True,'video_id':video_tests.FIRST,'text':'企鹅也开始上班了','reason':'share'}))
        await self.rt.videos.auto.run(self.route)
        return self.rt.videos.auto.states.get(self.route.key,{})

    async def test_no_incoming_event_or_passive_anchor_needed_and_no_member_created(self):
        await self.run_auto()
        call=self.ctx.llm_generate.call_args.kwargs
        self.assertIn(AUTO_VIDEO_RULE,call['system_prompt'])
        pack=json.loads(call['prompt'])['ambient_data']
        self.assertIsNone(pack['current_message'])
        self.assertIn('bilibili.com/video/'+video_tests.FIRST,self.sent[0])
        state=self.rt.videos.auto.states[self.route.key]
        self.assertEqual((1,1,1),(state['checks'],state['selected'],state['confirmed']))
        self.assertEqual([],self.store.snapshot()['members'])
        self.assertTrue(all(r['quote'] is None and r['source']=='video_share' for r in self.rt.windows[self.route.key].rows))

    async def test_last_unread_image_does_not_block_group_wide_share(self):
        await self.rt.observe(Event('今晚来点乐子',mid='old'))
        row=self.rt.windows[self.route.key].rows[-1]
        row['text']='[图片]';row['image_count']=1;row['attachment']=True
        self.rt.vision.gate=lambda *args:'image_unavailable'
        await self.run_auto()
        pack=json.loads(self.ctx.llm_generate.call_args.kwargs['prompt'])['ambient_data']
        self.assertIn('[图片]',str(pack['recent_context']))
        self.assertEqual(1,len(self.sent))

    async def test_scheduled_tick_works_without_new_input_or_interjection_enabled(self):
        self.config['groups'][0]['interject_enabled']=False
        self.ctx.llm_generate.return_value=NS(completion_text=json.dumps({'reply':True,'video_id':video_tests.FIRST,'text':'企鹅上班','reason':'share'}))
        self.rt.videos.auto.tick(time.time())
        tasks=list(self.rt.videos.auto.tasks.values())
        self.assertEqual(1,len(tasks))
        await asyncio.gather(*tasks)
        self.assertEqual(1,len(self.sent))

    async def test_cooldown_survives_restart_and_leaves_normal_chat_available(self):
        await self.run_auto()
        self.rt.videos.auto.states.clear()
        self.rt.stores[self.route.key]=Store(self.store.directory.parent,self.route.key,self.store.policy)
        self.rt.videos.auto.tick(time.time()+31)
        self.assertEqual({},self.rt.videos.auto.tasks)
        await unified.UnifiedTests.interject(self,Event('企鹅也是打工人',mid='later'),'是啊，一天没闲着')
        self.assertIn('一天没闲着',self.sent[-1])
        self.assertEqual(2,self.ctx.llm_generate.await_count)

    async def test_busy_topic_defers_with_reason_and_bounded_retry(self):
        state=await self.run_auto({'reply':False,'text':'','video_id':'','reason':'busy_topic'})
        self.assertEqual([],self.sent)
        self.assertEqual('busy_topic',state['outcome'])
        self.assertGreater(state['next_check'],time.time()+290)

    async def test_disable_and_normal_reply_priority_prevent_job(self):
        self.config['groups'][0]['video_auto_share_enabled']=False
        self.rt.videos.auto.tick(time.time())
        self.assertFalse(self.rt.videos.auto.tasks)
        self.config['groups'][0]['video_auto_share_enabled']=True
        await self.rt.observe(unified.directed('先回答我','priority'))
        self.rt.videos.auto.tick(time.time()+20)
        self.assertFalse(self.rt.videos.auto.tasks)
        self.ctx.llm_generate.assert_not_awaited()

    async def test_new_direct_message_preempts_before_claim(self):
        async def interrupt():
            await self.rt.observe(unified.directed('先别分享','interrupt'))
            return {'list':list(self.raw.values())}
        self.api.side_effect=interrupt
        state=await self.run_auto()
        self.assertEqual([],self.sent)
        self.assertEqual('draft_preempted',state['reason'])
        self.assertEqual(set(),self.store.video_history())

    async def test_uncertain_send_not_retried_after_restart(self):
        self.error=TimeoutError('private error')
        state=await self.run_auto()
        self.assertEqual('uncertain_no_retry',state['outcome'])
        self.assertEqual(0,state['confirmed_parts'])
        self.rt.videos.auto.states.clear()
        self.rt.videos.auto.tick(time.time()+31)
        self.assertFalse(self.rt.videos.auto.tasks)
        self.assertEqual(1,len(self.sent))


if __name__=='__main__': unittest.main()
