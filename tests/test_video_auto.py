"""Explicit proactive video decisions use the existing context and delivery path."""
import json
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

from test_ambient import Event
import test_unified as unified
import test_videos as video_tests
from ambient.videos import AUTO_VIDEO_RULE
from ambient.context import route_for
from ambient.store import Store


class AutoVideoTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = unified.UnifiedTests.asyncTearDown

    async def asyncSetUp(self):
        await video_tests.VideoFlowTests.asyncSetUp(self)
        self.config['groups'][0].update(video_auto_share_enabled=True,video_auto_share_interval_seconds=1800)
        self.store=self.rt.stores[self.route.key]

    async def run_auto(self, event=None, decision=None):
        event=event or Event('刚睡醒，下午又没了',mid='ordinary')
        await self.rt.observe(event)
        self.ctx.llm_generate.return_value=NS(completion_text=json.dumps(decision or {
            'reply':True,'video_id':video_tests.FIRST,'text':'企鹅也开始上班了','reason':'share'}))
        state=self.rt.interjections.states[self.route.key]
        state.suppressed_until=0
        await self.rt.interjections.run(self.route,state)
        return state

    async def test_ordinary_chat_gets_explicit_share_task_and_real_link_without_requesting_video(self):
        await self.run_auto()
        self.ctx.llm_generate.assert_awaited_once()
        call=self.ctx.llm_generate.call_args.kwargs
        self.assertIn(AUTO_VIDEO_RULE,call['system_prompt'])
        pack=json.loads(call['prompt'])['ambient_data']
        self.assertEqual('刚睡醒，下午又没了',pack['current_message']['text'])
        self.assertIn(video_tests.FIRST,pack['video_candidates'][0]['id'])
        self.assertIn('bilibili.com/video/'+video_tests.FIRST,self.sent[0])
        outcome=self.rt.videos.auto_states[self.route.key]
        self.assertEqual((1,1,1),(outcome['checks'],outcome['selected'],outcome['confirmed']))
        own=[r for r in self.rt.windows[self.route.key].rows if r['self']]
        self.assertTrue(all(r['source']=='video_share' and r['quote'] is None for r in own))
        self.assertEqual([],self.store.snapshot()['members'])

    async def test_share_interval_survives_restart_and_never_limits_normal_chat(self):
        await self.run_auto()
        self.rt.videos.auto_states.clear()
        self.rt.stores[self.route.key]=Store(self.store.directory.parent,self.route.key,self.store.policy)
        await self.run_auto(Event('企鹅也是打工人',mid='later'),{'reply':True,'text':'是啊，一天没闲着'})
        self.assertNotIn(AUTO_VIDEO_RULE,self.ctx.llm_generate.call_args.kwargs['system_prompt'])
        self.assertIn('一天没闲着',self.sent[-1])
        self.assertEqual(2,self.ctx.llm_generate.await_count)

    async def test_deferred_topic_has_bounded_reason_and_next_opportunity(self):
        await self.run_auto(Event('服务器崩了先看看日志'),{'reply':False,'text':'','reason':'busy_topic'})
        self.assertEqual([],self.sent)
        state=self.rt.videos.auto_states[self.route.key]
        self.assertEqual('busy_topic',state['outcome'])
        row=list(self.rt.windows[self.route.key].rows)[-1]
        self.assertFalse(self.rt.videos.auto_opportunity(self.route,row,self.rt.videos.candidates,now=state['at']+2))
        self.assertTrue(self.rt.videos.auto_opportunity(self.route,row,self.rt.videos.candidates,now=state['at']+301))

    async def test_other_group_disabled_bot_and_no_candidates_cannot_start_auto_task(self):
        row={'sender':'otherbot','self':False,'bot_sender':True}
        self.assertFalse(self.rt.videos.auto_opportunity(self.route,row,self.rt.videos.candidates))
        row={'sender':'person','self':False}
        other=route_for(self.ctx,Event(group='OTHER'))
        self.assertFalse(self.rt.videos.auto_opportunity(other,row,self.rt.videos.candidates))
        self.assertFalse(self.rt.videos.auto_opportunity(self.route,row,{}))
        self.config['groups'][0]['video_auto_share_enabled']=False
        self.assertFalse(self.rt.videos.auto_opportunity(self.route,row,self.rt.videos.candidates))

    async def test_actual_tick_does_not_steal_a_directed_reply_or_other_members_conversation(self):
        e=Event('<@other> 今晚去哪吃',mid='other')
        await self.rt.observe(e)
        state=self.rt.interjections.states[self.route.key]
        self.assertFalse(state.pending)
        await self.rt.interjections.tick(time.time()+20)
        self.assertEqual({},self.rt.interjections.tasks)
        await self.rt.observe(unified.directed('先回答我的问题','directed'))
        await self.rt.interjections.tick(time.time()+20)
        self.ctx.llm_generate.assert_not_awaited()

    async def test_native_failure_keeps_diagnostic_reason_and_no_false_success(self):
        self.api.side_effect=ValueError('video_source_changed')
        state=await self.run_auto()
        self.assertEqual('video_source_changed',state.outcome)
        record=self.rt.videos.auto_states[self.route.key]
        self.assertEqual(('failed','video_source_changed',0),(record['outcome'],record['reason'],record['confirmed']))
        self.assertEqual([],self.sent)

    async def test_new_direct_message_preempts_auto_send_without_reusing_old_reply(self):
        async def interrupt():
            await self.rt.observe(unified.directed('先别分享','interrupt'))
            return {'list':list(self.raw.values())}
        self.api.side_effect=interrupt
        await self.run_auto()
        self.assertEqual([],self.sent)
        self.assertEqual('failed',self.rt.videos.auto_states[self.route.key]['outcome'])
        self.assertEqual(set(),self.store.video_history())


if __name__=='__main__':unittest.main()
