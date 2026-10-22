"""Anonymous scenes exercising the real extraction, storage and lookup paths."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

import test_social_memory as fixtures
from test_ambient import Event, context
from ambient.alias_context import AliasLearning, discovered_names, infer_batch, prepare_batch, proposals, structural_proposals
from ambient.alias_evidence import human_text, pack_size, prepare_review
from ambient.aliases import identity_lookups, known_names
from ambient.context import Window, build_pack, compact, route_for
from ambient.learning import quote_source
from ambient.runtime import Runtime
from ambient.policy import reply_rejection


class AliasContextTests(unittest.TestCase):
    setUp = fixtures.SocialMemoryTests.setUp
    row = fixtures.SocialMemoryTests.row
    member = fixtures.SocialMemoryTests.member
    learn = fixtures.SocialMemoryTests.learn
    def scene(self):
        self.member('A', '青禾'); self.member('B', '阿林'); self.member('C', '小明')
        rows = [self.row('这小月又把门修好了', 'name1', 'B', ago=130),
                self.row('刚修好就坏了', 'target1', 'A', ago=120),
                self.row('小月的手艺稳', 'name2', 'C', ago=30),
                self.row('还得再修一次', 'target2', 'A', ago=20)]
        snapshot = self.store.snapshot()
        batch = prepare_batch(rows, snapshot, self.policy)
        result = {'aliases':[{'name':'小月','target':batch['accounts']['A'],
                              'evidence':[batch['refs'][r['id']] for r in rows]}]}
        return rows, snapshot, batch, result

    def test_context_candidate_survives_restart_without_becoming_confirmed(self):
        rows, snap, batch, raw = self.scene()
        items = proposals(json.dumps(raw), batch, snap)
        self.assertEqual('learned',self.store.learning.apply_context(items,batch,now=self.now))
        snapshot=self.store.snapshot()
        self.assertEqual('inferred',snapshot['aliases'][0]['kind'])
        a=next(m for m in snapshot['members'] if m['account']=='A')
        self.assertNotIn('小月',known_names(snapshot,a))
        current=self.row('小月是谁','query','C')
        pack=build_pack(Window(self.policy,[current]),snapshot,current,now=self.now)
        self.assertEqual('probable',pack['identity_lookup']['status'])
        self.assertEqual('青禾',next(c['name'] for c in pack['member_cards']
                                   if c['speaker']==pack['identity_lookup']['speakers'][0]))
        self.assertLessEqual(len(compact(pack)),self.policy.pack_chars)
        self.assertEqual(batch['cursor'],self.store.learning.cursor())

    def test_multiple_names_and_bot_suffix_are_separate_entities(self):
        self.member('A','小月'); self.member('B','小月bot'); self.member('C','阿林')
        current=self.row('小月是谁，阿林是谁，小月bot是谁','q','D')
        window=Window(self.policy,[current],style_excluded_senders={'B'})
        pack=build_pack(window,self.store.snapshot(),current,now=self.now)
        self.assertEqual(['小月','阿林','小月bot'],[q['name'] for q in pack['identity_lookups']])
        self.assertEqual(3,len({q['speakers'][0] for q in pack['identity_lookups']}))
        card=next(c for c in pack['member_cards'] if c['name']=='小月bot')
        self.assertTrue(card['automated'])
        self.assertFalse(next(c for c in pack['member_cards'] if c['name']=='小月').get('automated'))
        self.assertEqual('unconfirmed',identity_lookups({'text':'月是谁'},self.store.snapshot())[0]['status'] if identity_lookups({'text':'月是谁'},self.store.snapshot()) else 'unconfirmed')

    def test_model_cannot_supply_unknown_or_bot_target_or_fabricated_sources(self):
        rows,snap,batch,raw=self.scene()
        for patch in ({'target':'u999'}, {'name':'男同'}, {'evidence':['m999','m1']},
                      {'evidence':['m1']}, {'name':'未曾出现'}, {'target':[]}):
            data={'aliases':[{**raw['aliases'][0],**patch}]}
            try:items=proposals(json.dumps(data),batch,snap)
            except TypeError:self.fail('untrusted JSON must not cause TypeError')
            self.assertEqual([],items)
        batch['excluded']=['A']
        self.assertEqual([],proposals(json.dumps(raw),batch,snap))

    def test_single_scene_and_repeated_batch_do_not_create_confirmation(self):
        rows,snap,batch,raw=self.scene()
        raw['aliases'][0]['evidence']=[batch['refs'][r['id']] for r in rows[:2]]
        items=proposals(json.dumps(raw),batch,snap)
        self.store.learning.apply_context(items,batch,now=self.now)
        self.assertEqual('candidate',self.store.snapshot()['aliases'][0]['kind'])
        revision=self.store.memory_revision()
        batch['revision']=revision
        self.store.learning.apply_context(items,batch,now=self.now)
        self.assertEqual(revision,self.store.memory_revision())
        self.assertEqual('candidate',self.store.snapshot()['aliases'][0]['kind'])

    def test_correction_while_model_runs_rejects_stale_result(self):
        rows,snap,batch,raw=self.scene()
        items=proposals(json.dumps(raw),batch,snap)
        self.learn([self.row('别再叫我小月了','denial','A')])
        self.assertEqual('memory_changed_retry',self.store.learning.apply_context(items,batch,now=self.now))
        self.assertEqual([0,''],self.store.learning.cursor())
        batch['revision']=self.store.memory_revision()
        self.store.learning.apply_context(items,batch,now=self.now)
        self.assertEqual('rejected',next(a['kind'] for a in self.store.snapshot()['aliases'] if a['name']=='小月'))

    def test_forget_supporting_speaker_retracts_inference_and_old_backfill(self):
        rows,snap,batch,raw=self.scene()
        items=proposals(json.dumps(raw),batch,snap)
        self.store.learning.apply_context(items,batch,now=self.now)
        self.store.edit('forget_member',{},member=self.route.member('B'),now=self.now)
        self.assertEqual([],self.store.snapshot()['aliases'])
        batch['revision']=self.store.memory_revision()
        self.store.learning.apply_context(items,batch,now=self.now+86400*2)
        self.assertEqual([],self.store.snapshot()['aliases'])

    def test_quote_wrapper_resolves_outer_message_not_nested_ancestor(self):
        a=self.row('信息有点慢','a','A',ago=50)
        b=self.row('你也是小月？','b','B',ago=20,quote={
            'text':'=== 消息 1 === [消息内容] 信息有点慢 [消息类型] 引用消息 [关联消息] --- 第1条 --- [消息内容] 怎么崩了'})
        self.assertEqual(a,quote_source(b,[a,b])[0])
        self.assertIsNone(quote_source({**b,'quote':{**b['quote'],'id':'missing'}},[a,b])[0])
        self.assertIsNone(quote_source(b,[a,{**a,'id':'duplicate'},b])[0])
        self.assertIsNone(quote_source({**b,'quote':{**b['quote'],'truncated':True}},[a,b])[0])
        self.learn([a,b,self.row('✓','yes','A',quote={'id':'b','text':b['text']})])
        self.assertEqual([],self.store.snapshot()['aliases'])

    def test_batch_cursor_overlap_and_size_do_not_rescan_old_window(self):
        rows,snap,batch,_=self.scene()
        self.assertIsNone(prepare_batch(rows,snap,self.policy,after=batch['cursor']))
        self.assertNotIn(self.route.key,json.dumps(batch['data']))
        self.assertTrue(all(m['id'].startswith('m') for m in batch['data']['messages']))
        tiny=replace(self.policy,alias_context_chars=100)
        self.assertIsNone(prepare_batch(rows,snap,tiny))

    def test_conflicting_inferences_stay_ambiguous(self):
        rows,snap,batch,raw=self.scene()
        items=proposals(json.dumps(raw),batch,snap)
        self.store.learning.apply_context(items,batch,now=self.now)
        batch['revision']=self.store.memory_revision()
        self.store.learning.apply_context([{**i,'member':self.route.member('C')} for i in items],batch,now=self.now)
        result=identity_lookups({'text':'小月是谁'},self.store.snapshot())[0]
        self.assertEqual('ambiguous',result['status'])

    def test_operator_verification_has_distinct_provenance_and_wins_over_inference(self):
        rows,snap,batch,raw=self.scene()
        self.store.learning.review_aliases([{'member':self.route.member('A'),'name':'小月'}],
                                          'operator-reviewed-test',now=self.now)
        self.assertEqual('reviewed',self.store.snapshot()['aliases'][0]['kind'])
        batch['revision']=self.store.memory_revision()
        self.store.learning.apply_context(proposals(json.dumps(raw),batch,snap),batch,now=self.now)
        self.assertEqual('unique',identity_lookups({'text':'小月是谁'},self.store.snapshot())[0]['status'])
        self.assertEqual('reviewed',self.store.snapshot()['aliases'][0]['kind'])
        alias=self.store.snapshot()['aliases'][0]
        self.store.edit('delete',{'id':alias['id']},member=self.route.member('A'),now=self.now+1)
        self.assertEqual('unconfirmed',identity_lookups({'text':'小月是谁'},self.store.snapshot())[0]['status'])

    def test_distant_named_mentions_cannot_borrow_an_unrelated_target(self):
        rows,snap,batch,raw=self.scene()
        for r in batch['rows']:
            if r['sender']=='A':r['at']-=86400
        self.assertEqual([],proposals(json.dumps(raw),batch,snap))

    def test_confirmed_lookup_cannot_end_with_disclaiming_the_same_identity(self):
        self.member('A','青禾')
        self.store.learning.review_aliases([{'member':self.route.member('A'),'name':'小月'}],'review',now=self.now)
        row=self.row('小月是谁','q','C');window=Window(self.policy,[row])
        window.identity_snapshot=self.store.snapshot()
        self.assertEqual('identity_reply_conflict',reply_rejection('小月是青禾，但这外号跟本人对不上号',row,window))
        self.assertEqual('',reply_rejection('小月是青禾',row,window))
        unknown={**row,'text':'从没听过是谁'}
        self.assertEqual('',reply_rejection('不认识这个称呼',unknown,window))

    def test_discovery_can_find_a_name_without_guessing_its_owner(self):
        rows,snap,batch,_=self.scene()
        raw=json.dumps({'names':['小月','编造的名字',None,'男同'],'aliases':[]})
        self.assertEqual(['小月'],discovered_names(raw,batch))
        self.assertEqual([],proposals(raw,batch,snap))

    def test_repeated_media_and_old_overlap_cannot_starve_new_messages(self):
        self.member('A');self.member('B')
        old=[self.row('旧段落'*180,str(i),'A',ago=100+i) for i in range(12)]
        media=self.row('[表情] [图片]','image','A',ago=30)
        new=self.row('小月你来一下','new','B')
        cursor=[media['at'],media['id']]
        self.assertFalse(human_text(media))
        batch=prepare_batch([*old,media,new],self.store.snapshot(),replace(self.policy,alias_context_chars=1500),after=cursor)
        self.assertGreater(batch['cursor'],cursor)
        self.assertIn('new',{r['id'] for r in batch['rows']})
        self.assertLessEqual(pack_size(batch),1500)
        batch=prepare_batch([media],self.store.snapshot(),self.policy)
        self.assertTrue(batch['skip_model'])

    def test_review_keeps_recent_reply_evidence_and_excludes_bot_candidates(self):
        self.member('A','青禾');self.member('B','阿林');self.member('BOT','小月bot')
        rows=[self.row('这小月说的话真长'+'随便聊聊'*50,'old'+str(i),'B',ago=90000+i*100)
              for i in range(20)]
        target=self.row('我把车修好了','target','A',ago=20)
        name=self.row('这是小月的手艺','name','B',ago=10,quote={'id':'target','text':target['text']})
        batch=prepare_review([*rows,target,name],self.store.snapshot(),replace(self.policy,alias_context_chars=3000),['小月'],excluded=['BOT'])
        self.assertTrue({'name','target'} <= {r['id'] for r in batch['rows']})
        self.assertLessEqual(pack_size(batch),3000)
        self.assertNotIn('小月bot',{m['name'] for m in batch['data']['members']})
        rendered=next(r for r in batch['data']['messages'] if r['id']==batch['refs']['name'])
        self.assertEqual(batch['accounts']['A'],rendered['quoted_speaker'])

    def link_scene(self):
        self.member('A','青禾');self.member('B','阿林');self.member('C','小明')
        a=self.row('我修的车','a','A',ago=30)
        b=self.row('小月真能折腾','b','B',ago=20,quote={'id':'a','text':a['text']})
        c=self.row('小月是谁','c','C')
        snap=self.store.snapshot()
        batch=prepare_review([a,b,c],snap,self.policy,['小月'])
        link={'name':'小月','mention':batch['refs']['b'],'target':batch['accounts']['A'],
              'relation':'quoted_person','evidence':[batch['refs']['b'],batch['refs']['a']]}
        return snap,batch,link

    def test_link_must_name_the_actual_quote_author_not_the_mentioner(self):
        snap,batch,link=self.link_scene()
        self.assertEqual(self.route.member('A'),proposals(json.dumps({'links':[link]}),batch,snap)[0]['member'])
        for patch in ({'target':batch['accounts']['B']},{'target':batch['accounts']['C']},
                      {'mention':'missing'},{'evidence':[batch['refs']['b']]},{'relation':'made_up'}):
            self.assertEqual([],proposals(json.dumps({'links':[{**link,**patch}]}),batch,snap))
        self.assertEqual([],proposals(json.dumps({'aliases':[],'links':None}),batch,snap))

    def test_same_speaker_followup_keeps_listener_and_ignores_imitation(self):
        self.member('A');self.member('B');self.member('C')
        rows=[self.row('不来了','a','A',ago=30),self.row('<@A> 你不来？','b','B',ago=20),
              self.row('你不是小月吗','c','B',ago=18),self.row('你也是小月？','d','C',ago=10)]
        snap=self.store.snapshot();batch=prepare_review(rows,snap,self.policy,['小月'])
        link={'name':'小月','mention':batch['refs']['c'],'target':batch['accounts']['A'],
              'relation':'addressed_person','evidence':[batch['refs'][r['id']] for r in rows[:3]]}
        self.assertEqual(self.route.member('A'),proposals(json.dumps({'links':[link]}),batch,snap)[0]['member'])
        link.update(mention=batch['refs']['d'],evidence=[batch['refs']['d'],batch['refs']['a']],relation='topic_response')
        self.assertEqual([],proposals(json.dumps({'links':[link]}),batch,snap))

    def test_source_snippets_survive_raw_expiry_and_are_removed_on_forget(self):
        rows,snap,batch,raw=self.scene()
        self.store.learning.apply_context(proposals(json.dumps(raw),batch,snap),batch,now=self.now)
        restored=type(self.store)(self.tmp.name,self.route.key,self.policy)
        sources=restored.learning.context_rows()
        self.assertTrue({'name1','target1'} <= {r['id'] for r in sources})
        self.assertTrue(all(len(r['text'])<=280 for r in sources))
        restored.edit('forget_member',{},member=self.route.member('A'),now=self.now)
        self.assertEqual([],restored.learning.context_rows())

    def test_recheck_retracts_only_revisited_automatic_wrong_binding(self):
        rows,snap,batch,raw=self.scene()
        items=proposals(json.dumps(raw),batch,snap)
        self.store.learning.apply_context([{**i,'member':self.route.member('C')} for i in items],batch,now=self.now)
        batch.update(revision=self.store.memory_revision(),review_names=['小月'])
        self.store.learning.apply_context(items,batch,now=self.now)
        lookup=identity_lookups({'text':'小月是谁'},self.store.snapshot())[0]
        self.assertEqual(['A'],lookup['accounts'])
        self.assertEqual('probable',lookup['status'])

    def test_target_message_resolves_stable_member_and_cannot_conflict_with_id(self):
        snap,batch,link=self.link_scene()
        link.pop('target');link['target_message']=batch['refs']['a']
        self.assertEqual(self.route.member('A'),proposals(json.dumps({'links':[link]}),batch,snap)[0]['member'])
        link['target']=batch['accounts']['B']
        self.assertEqual([],proposals(json.dumps({'links':[link]}),batch,snap))

    def test_unresolved_scenes_persist_without_becoming_identity_and_honor_forget(self):
        rows,snap,batch,_=self.scene()
        batch['review_names']=['小月']
        self.assertEqual('evidence_saved',self.store.learning.apply_context([],batch,now=self.now))
        self.assertEqual([],self.store.snapshot()['aliases'])
        self.assertTrue(self.store.learning.context_rows())
        self.store.edit('forget_member',{},member=self.route.member('A'),now=self.now+1)
        self.assertEqual([],self.store.learning.context_rows())
        batch['revision']=self.store.memory_revision()
        self.store.learning.apply_context([],batch,now=self.now+2)
        self.assertEqual([],self.store.learning.context_rows())

    def test_unresolved_scene_archive_is_count_bounded_and_compressible(self):
        self.member('A');self.member('B')
        rows=[]
        for i in range(8):
            rows += [self.row('这个小月又开始了','n'+str(i),'A',ago=1000-i*100),
                     self.row('等一下','r'+str(i),'B',ago=990-i*100)]
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        self.store.learning.apply_context([],batch,now=self.now)
        with self.store.connection() as db:
            self.assertLessEqual(db.execute('SELECT COUNT(*) FROM alias_scenes').fetchone()[0],3)
        self.store.maintain(force=True,now=self.now)
        self.assertEqual([],self.store.learning.context_rows())

    def test_response_chain_is_a_hint_and_cannot_use_a_bot_agreement(self):
        self.member('A');self.member('B');self.member('C')
        rows=[self.row('小月怎么开始修车了','a','A',ago=40),self.row('他爱拆东西','b','A',ago=35),
              self.row('对','c','B',ago=30),self.row('我发现拆东西挺快乐','d','B',ago=20)]
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        chain=batch['data']['response_chains'][0]
        self.assertEqual(batch['refs']['d'],chain['continuation'])
        self.assertEqual('unconfirmed',chain['status'])
        self.assertEqual([],self.store.snapshot()['aliases'])
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'],excluded=['B'])
        self.assertFalse(batch['data']['response_chains'])

    def test_structural_learning_uses_reply_author_and_followup_you(self):
        self.member('A','青禾');self.member('B','阿林')
        rows=[self.row('这机器还能修','a','A',ago=40),
              self.row('这是小月的强项','b','B',ago=30,quote={'id':'a','text':'这机器还能修'}),
              self.row('你又能发挥了','c','B',ago=25)]
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        items=structural_proposals(batch,self.store.snapshot())
        self.assertEqual({self.route.member('A')},{i['member'] for i in items})
        self.store.learning.apply_context(items,batch,now=self.now)
        self.assertEqual('candidate',self.store.snapshot()['aliases'][0]['kind'])

    def test_structural_learning_distinguishes_rhetorical_call_from_plain_denial(self):
        self.member('A');self.member('B');self.member('C')
        rows=[self.row('小月的车还在这里','old','C',ago=600),self.row('我先走了','a','A',ago=50),
              self.row('<@A> 这就走了？','b','B',ago=40),self.row('你不是小月','c','B',ago=35)]
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        self.assertEqual([],structural_proposals(batch,self.store.snapshot()))
        rows.append(self.row('你被什么东西夺舍了','d','B',ago=30))
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        items=structural_proposals(batch,self.store.snapshot())
        self.assertEqual({self.route.member('A')},{i['member'] for i in items})

    def test_first_person_recognition_stays_tentative_and_comparison_does_not_bind(self):
        self.member('A');self.member('B')
        rows=[self.row('小月的说话风格变了','a','A',ago=40),self.row('他爱拆机器','b','A',ago=35),
              self.row('对','c','B',ago=30),self.row('我发现修机器很快乐','d','B',ago=20)]
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        items=structural_proposals(batch,self.store.snapshot())
        self.assertEqual({self.route.member('B')},{i['member'] for i in items})
        self.store.learning.apply_context(items,batch,now=self.now)
        self.assertEqual('tentative',identity_lookups({'text':'小月是谁'},self.store.snapshot())[0]['status'])
        rows[-1]['text']='我也发现修机器很快乐'
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        self.assertEqual([],structural_proposals(batch,self.store.snapshot()))

    def test_first_person_chain_cannot_cross_an_explicit_topic_change(self):
        self.member('A');self.member('B')
        rows=[self.row('小月的说话风格变了','a','A',ago=40),self.row('他爱修车','b','A',ago=35),
              self.row('换个话题','switch','A',ago=32),self.row('对','c','B',ago=30),
              self.row('我发现新开的餐馆挺好吃','d','B',ago=20)]
        batch=prepare_review(rows,self.store.snapshot(),self.policy,['小月'])
        self.assertEqual([],structural_proposals(batch,self.store.snapshot()))

    def test_pipeline_uses_source_edges_without_saved_mappings_or_extra_reviews(self):
        self.member('A');self.member('B')
        rows=[self.row('这机器还能修','a','A',ago=40),
              self.row('这是小月的强项','b','B',ago=30,quote={'id':'a','text':'这机器还能修'}),
              self.row('你又能发挥了','c','B',ago=25)]
        snap=self.store.snapshot()
        snap['aliases']=[{'name':'禁止进入发现提示词','member':self.route.member('B')}]
        async def generate(system,data):
            self.assertNotIn('禁止进入发现提示词',json.dumps(data,ensure_ascii=False))
            self.assertNotIn('names_to_review',data)
            return json.dumps({'names':['小月']})
        mock=AsyncMock(side_effect=generate)
        items,batch=asyncio.run(infer_batch(rows,snap,self.policy,mock))
        self.assertEqual(1,mock.await_count)
        self.assertEqual({self.route.member('A')},{i['member'] for i in items})
        self.assertEqual([],batch['review_errors'])

    def test_review_failure_keeps_unresolved_sources_and_stops_extra_calls(self):
        rows,snap,_,_=self.scene()
        for r in rows:r['text']='小月和小云来过这里'
        mock=AsyncMock(side_effect=[json.dumps({'names':['小月','小云']}),TimeoutError()])
        items,batch=asyncio.run(infer_batch(rows,snap,self.policy,mock))
        self.assertEqual([],items)
        self.assertEqual(2,mock.await_count)
        self.assertEqual(['TimeoutError'],batch['review_errors'])
        self.assertEqual('evidence_saved',self.store.learning.apply_context(items,batch,now=self.now))
        self.assertEqual([],self.store.snapshot()['aliases'])
        self.assertTrue(self.store.learning.context_rows())

    def test_review_budget_and_cancellation_are_not_silently_retried(self):
        rows,snap,_,_=self.scene()
        for r in rows:r['text']='小月和小云来过这里'
        mock=AsyncMock(side_effect=[json.dumps({'names':['小月','小云']}),json.dumps({'links':[]})])
        items,batch=asyncio.run(infer_batch(rows,snap,replace(self.policy,alias_context_reviews=1),mock))
        self.assertEqual(2,mock.await_count)
        self.assertEqual(1,batch['review_count'])
        mock=AsyncMock(side_effect=[json.dumps({'names':['小月']}),asyncio.CancelledError()])
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(infer_batch(rows,snap,self.policy,mock))


class AliasWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_overlap_and_cancel_on_close(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            ctx=context();route=route_for(ctx,Event());rt=Runtime(root,ctx,{'groups':[{'umo':route.umo,'capture_only':False}]})
            store,window=await rt.ensure(route)
            for i in range(4):
                e=Event('普通聊天'+str(i),mid=str(i),sender='A' if i%2 else 'B')
                await rt.observe(e)
            store.identify(route.platform,'A','甲')
            store.identify(route.platform,'B','乙')
            started=asyncio.Event()
            async def pending(**kw):
                self.assertIsNone(kw['tools']); self.assertEqual([],kw['contexts'])
                started.set();await asyncio.Event().wait()
            ctx.llm_generate.side_effect=pending
            rt.alias_learning.tick(route);rt.alias_learning.tick(route)
            await asyncio.wait_for(started.wait(),2)
            self.assertEqual(1,ctx.llm_generate.call_count)
            await rt.close()
            self.assertFalse(rt.alias_learning.tasks)
