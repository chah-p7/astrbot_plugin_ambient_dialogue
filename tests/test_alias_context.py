"""Anonymous scenes exercising the real extraction, storage and lookup paths."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock

import test_social_memory as fixtures
from test_ambient import Event, context
from ambient.alias_context import AliasLearning, prepare_batch, proposals
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
