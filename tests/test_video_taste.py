"""Evidence ownership, source boundaries and selection before any QQ side effect."""
import json
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import unittest

from test_ambient import Event
import test_unified as unified
from test_videos import raw_video, FIRST, SECOND
from ambient.context import event_message, route_for
from ambient.policy import stable_id
from ambient.video_sources import references, resolve_reference
from ambient.video_selection import parse_selection, SELECTION_RULES
from ambient.video_taste import TasteProfile
from ambient.videos import video_row


class VideoTasteTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = unified.UnifiedTests.asyncSetUp
    asyncTearDown = unified.UnifiedTests.asyncTearDown

    async def setup_video(self):
        self.config['groups'][0].update(video_sharing_enabled=True,video_cover_selection_enabled=False)
        await self.rt.ensure(self.route)
        self.store = self.rt.stores[self.route.key]
        self.taste = self.store.video_taste
        self.now = time.time()

    def row(self, ref=FIRST, sender='human', mid='share', title='企鹅鬼畜音MAD'):
        return {'id':mid,'sender':sender,'member':self.route.member(sender),'self':False,'command':False,
                'at':self.now,'text':'【'+title+'-哔哩哔哩】 https://www.bilibili.com/video/'+ref+'/'}

    async def test_only_human_shares_learn_and_do_not_create_member_profiles(self):
        await self.setup_video()
        self.assertEqual(1,self.taste.observe(self.row(),now=self.now))
        self.assertEqual(0,self.taste.observe(self.row(mid='duplicate'),now=self.now))
        for r in [{**self.row(SECOND),'self':True},{**self.row(SECOND),'command':True},self.row(SECOND,sender='other_bot')]:
            self.assertEqual(0,self.taste.observe(r,{'other_bot'},now=self.now))
        self.assertEqual(1,len(self.taste.examples(now=self.now)))
        self.assertEqual([],self.store.snapshot()['members'])

    async def test_card_only_event_and_short_links_keep_exact_source(self):
        e=Event('')
        Json=type('Json',(),{})
        card=Json();card.data={'app':'com.tencent.miniapp_01','meta':{'detail_1':{'desc':'企鹅跳舞','qqdocurl':'https://b23.tv/AbCd123'}}}
        e.message_obj.message=[card]
        r=event_message(e,self.route)
        self.assertEqual('https://b23.tv/AbCd123',r['video_refs'][0]['ref'])
        self.assertEqual('企鹅跳舞',r['video_refs'][0]['title'])
        self.assertIsNotNone(r)
        self.assertEqual([],references('prefix'+FIRST+'suffix'))
        self.assertEqual([],references('https://b23.tv.evil.test/AbCd123'))
        for bad in ('http://127.0.0.1/a','https://b23.tv@localhost/a','file:///secret'):
            with self.assertRaisesRegex(ValueError,'video_reference_invalid'):
                await resolve_reference(bad)

    async def test_forget_clears_evidence_and_inflight_resolution_cannot_restore_it(self):
        await self.setup_video()
        r=self.row()
        self.taste.observe(r,now=self.now)
        pending=self.taste.pending(self.now)[0]
        self.store.edit('forget_member',{},member=r['member'],now=self.now+1)
        self.taste.resolve(pending,video_row(raw_video()),now=self.now+2)
        self.assertEqual([],self.taste.examples(now=self.now+2))
        self.assertEqual(0,self.taste.observe(r,now=self.now+2))

    async def test_expiry_quota_and_compression_bound_memory(self):
        await self.setup_video()
        self.taste.configure({'video_learning_samples':16,'video_learning_days':7})
        for i in range(30):
            self.taste.observe(self.row('BV'+str(i).zfill(10),mid=str(i)),now=self.now)
        self.assertEqual(16,len(self.taste.examples(now=self.now)))
        self.assertEqual([],self.taste.examples(now=self.now+8*86400))
        from dataclasses import replace
        self.store.policy=replace(self.store.policy,quota_mb=.001)
        self.assertEqual(0,self.taste.observe(self.row(),now=self.now))

    async def test_public_unavailable_uses_shared_title_and_scoped_related_source(self):
        await self.setup_video()
        self.taste.observe(self.row(),now=self.now)
        with patch('ambient.video_selection.public_json',AsyncMock(side_effect=ValueError('video_source_unavailable'))), \
             patch('ambient.video_selection.related_feed',AsyncMock(return_value=[raw_video(SECOND)])) as related:
            await self.rt.videos.selection.refresh(self.route)
        related.assert_awaited_once_with(FIRST)
        profile=self.rt.videos.selection.profile(self.route,self.now)
        self.assertEqual(1,profile.size)
        self.assertEqual('shared_title',profile.examples[0]['video']['source'])
        self.assertIn(SECOND,self.rt.videos.pool(self.route))
        other=route_for(self.ctx,Event(group='OTHER'))
        self.assertNotIn(SECOND,self.rt.videos.pool(other))
        with patch('ambient.videos.related_feed',AsyncMock(return_value=[raw_video(SECOND)])) as feed:
            await self.rt.videos.verify(video_row(raw_video(SECOND)),self.route)
            feed.assert_awaited_once_with(FIRST)

    async def test_profile_dedup_decay_and_multi_member_support(self):
        now=time.time()
        one={'ref':FIRST,'member':'a','at':now,'video':{'id':FIRST,'title':'音MAD二创','category':'鬼畜'}}
        two={**one,'member':'b','ref':SECOND,'video':{**one['video'],'id':SECOND}}
        three={**one,'member':'c','ref':'BV0000000003','video':{**one['video'],'id':'BV0000000003'}}
        target={'title':'音MAD二创合集','category':'鬼畜'}
        single=TasteProfile([one],{},now)
        profile=TasteProfile([one,two,three],{},now)
        repeated=TasteProfile([one]*20,{},now)
        old=TasteProfile([{**r,'at':now-90*86400} for r in [one,two,three]],{},now)
        self.assertGreater(profile.affinity(target),single.affinity(target))
        self.assertEqual(1,repeated.size)
        self.assertLessEqual(repeated.affinity(target),single.affinity(target))
        self.assertLess(old.affinity(target),profile.affinity(target))

    async def test_resolved_short_link_is_seen_even_when_its_title_cannot_be_fetched(self):
        await self.setup_video()
        r={**self.row(),'text':'https://b23.tv/AbCd123'}
        self.taste.observe(r,now=self.now)
        with patch('ambient.video_selection.resolve_reference',AsyncMock(return_value=FIRST)), \
             patch('ambient.video_selection.public_json',AsyncMock(side_effect=ValueError('video_source_unavailable'))):
            await self.rt.videos.selection.refresh(self.route)
        self.assertIn(FIRST,self.store.video_history())
        self.assertEqual(0,self.rt.videos.selection.profile(self.route,self.now).size)

    async def test_current_topic_wins_over_taste_and_candidates_fit_budget(self):
        await self.setup_video()
        self.rt.videos.candidates={FIRST:video_row(raw_video(FIRST,'企鹅动画混剪')),SECOND:video_row(raw_video(SECOND,'服务器游戏配置教程'))}
        self.rt.videos.fetched_at=self.now
        for i in range(3):self.taste.observe(self.row('BV'+str(i).zfill(10),sender=str(i),title='企鹅动画混剪'),now=self.now)
        self.rt.videos.selection.notes[self.route.key]={FIRST:{'score':3,'traits':['动画混剪'],'evidence':'可见反差'}}
        pack={'current_message':{'text':'服务器配置'},'recent_context':[]}
        offered=self.rt.videos.add_choices(self.route,pack)
        self.assertEqual(SECOND,next(iter(offered)))
        self.assertLessEqual(len(json.dumps(pack,ensure_ascii=False,separators=(',',':'))),self.rt.policy(self.route).pack_chars)

    async def test_native_backfill_excludes_bot_and_unrelated_scope(self):
        await self.setup_video()
        def native(sender,scope):
            return NS(id=sender,platform_id=self.route.platform,user_id=scope,sender_id=sender,created_at=self.now,
                      content=[{'type':'Plain','text':'【音MAD二创】 https://www.bilibili.com/video/'+FIRST+'/'}])
        manager=NS(get=AsyncMock(side_effect=[[native('person',self.route.group),native(self.route.account,self.route.group),native('other','elsewhere')],[]]))
        self.ctx.message_history_manager=manager
        await self.rt.videos.selection.backfill(self.route)
        self.assertEqual(1,len(self.taste.examples(now=self.now)))
        await self.rt.videos.selection.backfill(self.route)
        self.assertEqual(2,manager.get.await_count)

    async def test_selection_is_background_cached_and_disabled_change_discards_result(self):
        await self.setup_video()
        self.config['groups'][0].update(video_cover_selection_enabled=True,video_selection_provider='vision')
        self.rt.videos.candidates={FIRST:video_row(raw_video())}
        reply=NS(completion_text=json.dumps({'items':[{'id':FIRST,'score':3,'traits':['动画混剪'],'evidence':'标题封面反差'}]}))
        self.ctx.llm_generate.return_value=reply
        with patch('ambient.video_preview.fetch_cover',AsyncMock(return_value=b'cover')),patch('ambient.video_selection.cover_data',return_value='data:image/jpeg;base64,YQ=='):
            await self.rt.videos.selection.refresh(self.route)
            await self.rt.videos.selection.refresh(self.route)
        self.ctx.llm_generate.assert_awaited_once()
        self.assertEqual(1,self.rt.videos.selection.states[self.route.key]['reviewed'])
        self.assertIsNone(self.ctx.llm_generate.call_args.kwargs['tools'])
        self.assertEqual([],self.ctx.llm_generate.call_args.kwargs['contexts'])
        self.assertEqual(SELECTION_RULES,self.ctx.llm_generate.call_args.kwargs['system_prompt'])
        self.assertEqual([],self.sent)
        self.rt.videos.candidates[SECOND]=video_row(raw_video(SECOND))
        async def change(**kwargs):
            self.config['groups'][0]['video_sharing_enabled']=False
            return NS(completion_text=json.dumps({'items':[{'id':SECOND,'score':3,'traits':['动画混剪'],'evidence':'标题封面反差'}]}))
        self.ctx.llm_generate.side_effect=change
        with patch('ambient.video_preview.fetch_cover',AsyncMock(return_value=b'cover')),patch('ambient.video_selection.cover_data',return_value='data:image/jpeg;base64,YQ=='):
            await self.rt.videos.selection.refresh(self.route)
        self.assertNotIn(SECOND,self.taste.notes(self.rt.videos.candidates,now=self.now))

    async def test_model_can_only_annotate_offered_ids_and_cannot_change_candidates(self):
        parsed=parse_selection(json.dumps({'items':[{'id':FIRST,'score':3,'traits':['音MAD'],'evidence':'仅标题：跨作品二创'},
            {'id':'https://bad.invalid','score':3,'traits':[],'evidence':'x'},
            {'id':SECOND,'score':'excellent','traits':[],'evidence':'x'}]}),{FIRST,SECOND})
        self.assertEqual({FIRST},set(parsed))
        self.assertEqual('仅标题：跨作品二创',parsed[FIRST]['evidence'])


if __name__=='__main__':unittest.main()
