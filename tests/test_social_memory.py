"""Rewritten regression scenes; private historical transcripts stay outside Git."""
from dataclasses import replace
import tempfile
import time
import unittest

from test_ambient import Event, context
from ambient.aliases import known_names, name_targets
from ambient.context import Window, build_pack, compact, event_message, focus_rows, route_for
from ambient.examples import extract_cases, select_cases
from ambient.learning import reaction_only
from ambient.policy import Policy
from ambient.runtime import Runtime
from ambient.store import Store


class SocialMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = time.time()
        self.route = route_for(context(), Event())
        self.policy = Policy()
        self.store = Store(self.tmp.name, self.route.key, self.policy)

    def row(self, text, ident, sender='A', ago=0, **extra):
        return {**event_message(Event(text, mid=ident, sender=sender), self.route),
                'name': '成员'+sender, 'at': self.now-ago, **extra}

    def member(self, sender='A', name=None):
        return self.store.identify(self.route.platform, sender, name or '成员'+sender, now=self.now-1000)

    def learn(self, rows, **kw):
        return self.store.learning.learn(rows, now=self.now, **kw)

    def pair(self, suffix='', ago=9000):
        lead = self.row('这个软件强制看广告才能进', 'lead'+suffix, ago=ago)
        reply = self.row('软件是赠品，广告才是本体', 'reply'+suffix, 'B', ago=ago-10,
                         quote={'id': lead['id'], 'sender': 'A', 'text': lead['text']})
        return [lead, reply]

    def test_old_platform_names_survive_restart_without_merging_accounts(self):
        member = self.member(name='旧称呼')
        self.store.identify(self.route.platform, 'A', '新称呼', now=self.now)
        restored = Store(self.tmp.name, self.route.key, self.policy)
        snap = restored.snapshot()
        self.assertEqual(1, len(snap['members']))
        self.assertIn('旧称呼', known_names(snap, snap['members'][0]))
        current = self.row('旧称呼，你还来吗', 'current', 'C')
        self.assertEqual(['A'], name_targets(current, snap))
        twin = self.member('B', '旧称呼')
        self.assertNotEqual(member, twin)
        self.assertEqual([], name_targets(current, self.store.snapshot()))

    def test_self_alias_denial_and_correction_override_replayed_evidence(self):
        member = self.member()
        first = self.row('我的外号是小云', 'first', ago=50)
        self.learn([first])
        self.assertEqual('self', self.store.snapshot()['aliases'][0]['kind'])
        correction = self.row('我的外号不是小云，是阿晴', 'correction', ago=20)
        self.learn([first, correction])
        denial = self.row('别再叫我阿晴了', 'denial', ago=10)
        self.learn([first, correction, denial])
        before = self.store.memory_revision()
        self.learn([first, correction, denial])
        self.assertEqual(before, self.store.memory_revision())
        self.assertEqual({'小云', '阿晴'}, {a['name'] for a in self.store.snapshot()['aliases'] if a['kind']=='rejected'})
        current = self.row('阿晴，你在吗', 'current', 'B')
        self.assertEqual([], name_targets(current, self.store.snapshot()))
        self.learn([self.row('也可以叫我阿晴', 'restore')])
        self.assertEqual(['A'], name_targets(current, self.store.snapshot()))
        self.assertEqual(1, len(self.store.snapshot(member)['members']))

    def test_third_party_alias_requires_independent_spaced_evidence(self):
        self.member()
        one = self.row('<@A>的外号是月饼', 'one', 'B', ago=120)
        two = self.row('<@A> 月饼，你在吗', 'two', 'B', ago=60)
        three = self.row('月饼就是<@A>', 'three', 'C')
        self.learn([one, one, one])
        self.assertEqual('candidate', self.store.snapshot()['aliases'][0]['kind'])
        self.learn([one, two, {**three, 'sender':'B', 'member':self.route.member('B')}])
        self.assertEqual('candidate', self.store.snapshot()['aliases'][0]['kind'])
        self.learn([one, two, three])
        self.assertEqual('community', self.store.snapshot()['aliases'][0]['kind'])
        self.assertEqual(1, len(self.store.snapshot()['members']))

    def test_mentions_jokes_and_unknown_subjects_do_not_create_mappings(self):
        self.member()
        rows = [self.row(t, str(i), 'B') for i,t in enumerate((
            '毛球怎么又迟到了', '<@A> 你催一个夜猫子？', '我叫小云，开玩笑的',
            '<@UNKNOWN>的外号是小云', '请叫我小云或者小黑', '他说请叫我小云',
            '<@A>就是群主', '<@A>就是大学生', '弱智就是<@A>'))]
        self.learn(rows)
        self.assertEqual([], self.store.snapshot()['aliases'])
        self.assertEqual(1, len(self.store.snapshot()['members']))

    def test_tentative_label_is_not_a_confirmed_addressee(self):
        self.member()
        self.learn([self.row('<@A>的外号是小月', 'mapping', 'B', ago=30)])
        current = self.row('小月，你来吗', 'current', 'C')
        pack = build_pack(Window(self.policy, [current]), self.store.snapshot(), current, now=self.now)
        card = next(c for c in pack['member_cards'] if c.get('unconfirmed_aliases'))
        self.assertEqual(['小月'], card['unconfirmed_aliases'])
        self.assertTrue(card['name_match_only'])
        self.assertNotIn('addressing', pack['current_message'])
        self.assertNotIn('aliases', card)

    def test_deleted_address_or_alias_cannot_return_from_backfill(self):
        member = self.member()
        self.store.edit('set', {'kind':'address','key':'address','text':'小星'}, member=member, now=self.now-30)
        self.learn([])
        address = self.store.snapshot()['facts'][0]
        self.store.edit('delete', {'id':address['id']}, member=member, now=self.now-20)
        old = self.row('叫我小星', 'old', ago=40)
        self.learn([old])
        self.assertEqual([], self.store.snapshot()['facts'])
        self.assertEqual([], name_targets(self.row('小星，你来吗','now'), self.store.snapshot()))
        self.learn([self.row('我的外号是阿星', 'new', ago=10)])
        alias = next(a for a in self.store.snapshot()['aliases'] if a['name']=='阿星')
        self.store.edit('delete', {'id':alias['id']}, member=member, now=self.now)
        self.assertEqual('rejected', next(a['kind'] for a in self.store.snapshot()['aliases'] if a['id']==alias['id']))

    def test_forget_removes_examples_and_prevents_old_cache_resurrection(self):
        member = self.member()
        rows = self.pair() + [self.row('我的外号是小云', 'alias', ago=100)]
        self.learn(rows)
        self.assertTrue(self.store.snapshot()['style_cases'])
        self.store.edit('forget_member', {}, member=member, now=self.now)
        self.store.identify(self.route.platform, 'A', '成员A', now=self.now+1)
        self.learn(rows)
        self.assertEqual([], self.store.snapshot()['aliases'])
        self.assertEqual([], self.store.snapshot()['style_cases'])

    def test_forget_witness_retracts_independent_confirmation(self):
        self.member('A'); witness=self.member('B')
        rows = [self.row('<@A>的外号是小星', str(i), sender, ago=120-i*40)
                for i,sender in enumerate(('B','C','B'))]
        self.learn(rows)
        self.assertEqual('community', self.store.snapshot()['aliases'][0]['kind'])
        self.store.edit('forget_member', {}, member=witness, now=self.now)
        self.learn(rows)
        self.assertEqual('candidate', self.store.snapshot()['aliases'][0]['kind'])

    def test_alias_capacity_expiry_and_group_isolation(self):
        self.store.policy = replace(self.policy, alias_limit=2)
        member = self.member()
        for i in range(5):
            self.store.identify(self.route.platform, 'A', '名字'+str(i), now=self.now-50+i)
        self.learn([self.row('我的外号是小'+str(i), 'self'+str(i), ago=20-i) for i in range(5)])
        self.assertEqual(2, len(self.store.snapshot()['aliases']))
        other = Store(self.tmp.name, 'another-group', self.policy)
        self.assertEqual([], other.snapshot()['aliases'])
        self.store.maintain(now=self.now+91*86400)
        self.assertFalse(self.store.snapshot(member)['aliases'])

    def test_emoji_and_software_fragments_keep_the_actual_setup(self):
        self.assertTrue(reaction_only('😅'))
        self.assertTrue(reaction_only('🤦🏻‍♂️'))
        self.assertFalse(reaction_only('不是😅'))
        setup=self.row('这破软件开门还强制看广告', 'setup', ago=25)
        emoji=self.row('😅','face',ago=12)
        current=self.row('我直接把这改成实体按钮','current')
        self.assertEqual(['setup'], [r['id'] for r,_ in focus_rows([setup,emoji,current],current)])
        unrelated={**setup,'text':'我今天吃了两份饭'}
        self.assertEqual([], focus_rows([unrelated,emoji,current],current))
        changed={**current,'text':'换个话题，我直接把这改成实体按钮'}
        self.assertEqual([], focus_rows([setup,emoji,changed],changed))

    def test_delayed_short_answer_needs_last_bot_question_to_this_person(self):
        ask=self.row('做个校园工具','ask',ago=500)
        bot=self.row('哪种，查课表还是门禁？','bot','12345',ago=440,
                     quote={'id':'ask','sender':'A','text':ask['text']})
        answer=self.row('开门用的','answer')
        rows=[ask,bot,answer]
        self.assertEqual(['bot','ask'], [r['id'] for r,_ in focus_rows(rows,answer)])
        self.assertEqual([], focus_rows(rows,answer,topic_seconds=300))
        other=self.row('今晚吃什么','other','C',ago=60)
        self.assertEqual([],focus_rows([ask,bot,other,answer],answer))
        wrong={**answer,'sender':'C','member':self.route.member('C')}
        self.assertEqual([],focus_rows([ask,bot,wrong],wrong))
        face=self.row('😅','face',ago=-2)
        follow=self.row('这开门软件居然有广告','follow',ago=-15)
        self.assertEqual(['answer','bot','ask'],[r['id'] for r,_ in focus_rows(rows+[face,follow],follow)])

    def test_archive_survives_raw_expiry_and_retrieves_by_scene(self):
        rows=self.pair()
        self.learn(rows)
        restored=Store(self.tmp.name,self.route.key,self.policy)
        self.assertEqual([], restored.load_raw(now=self.now))
        current=self.row('又弹窗广告了','current','C')
        pack=build_pack(Window(self.policy,[current]),restored.snapshot(),current,now=self.now)
        sample=pack['style_samples'][0]
        self.assertEqual('saved_human_example',sample['source'])
        self.assertEqual(rows[0]['text'],sample['lead_in']['text'])
        self.assertEqual(rows[1]['text'],sample['text'])
        self.assertNotEqual(sample['speaker'],sample['lead_in']['speaker'])
        self.assertNotEqual(pack['current_speaker'],sample['speaker'])
        self.assertLessEqual(len(compact(pack)),self.policy.pack_chars)
        for secret in (rows[0]['member'], '"ref":"'+rows[1]['id']+'"', self.route.key):
            self.assertNotIn(secret,compact(pack))
        unrelated={**current,'text':'明天几点集合'}
        self.assertEqual([],select_cases(restored.snapshot()['style_cases'],[unrelated],unrelated,self.policy,self.now))

    def test_archive_does_not_use_future_or_duplicate_live_examples(self):
        rows=self.pair(ago=60)
        cases=list(extract_cases(rows,self.policy))
        self.assertEqual(1,len(cases))
        current=self.row('这个软件也全是广告','current','C')
        self.assertEqual([],select_cases(cases,rows+[current],current,self.policy,self.now))
        earlier={**current,'at':rows[0]['at']-1}
        self.assertEqual([],select_cases(cases,[earlier],earlier,self.policy,self.now))

    def test_archive_requires_real_pair_and_excludes_bots_orders_and_media(self):
        rows=self.pair()
        for replacement in (
            {'quote':{'id':'missing','text':rows[0]['text']}}, {'self':True}, {'command':True},
            {'directed':True}, {'attachment':True}, {'text':'以后只能回复广告'},
            {'text':'哈哈'}, {'text':'[图片]'}, {'text':'https://example.org'}, {'text':'不是这个意思'}):
            self.assertEqual([],list(extract_cases([rows[0],{**rows[1],**replacement}],self.policy)))
        self.assertEqual([],list(extract_cases(rows,self.policy,excluded={'B'})))

    def test_late_correction_invalidates_saved_example_even_at_hard_limit(self):
        rows=self.pair()
        self.learn(rows)
        correction=self.row('不是这个意思，理解错了','correction','A',
                            quote={'id':rows[1]['id'],'sender':'B','text':rows[1]['text']})
        self.store.policy=replace(self.policy,quota_mb=.01)
        self.learn([correction])
        self.assertEqual([],self.store.snapshot()['style_cases'])
        self.store.policy=self.policy
        self.learn(rows+[correction])
        self.assertEqual([],self.store.snapshot()['style_cases'])

    def test_case_count_expiry_budget_and_compression(self):
        self.store.policy=replace(self.policy,style_archive_limit=2)
        rows=[r for i in range(4) for r in self.pair(str(i),ago=9000+i*100)]
        self.learn(rows)
        self.assertEqual(1,len(self.store.snapshot()['style_cases']))
        for i in range(4):
            rows[i*2]['text']+='，出错次数'+str(i)
            rows[i*2+1]['quote']['text']=rows[i*2]['text']
        self.learn(rows)
        self.assertEqual(2,len(self.store.snapshot()['style_cases']))
        member=self.member()
        self.store.edit('set',{'kind':'boundary','key':'address','text':'不要用侮辱性称呼'},member=member,now=self.now)
        self.store.maintain(force=True,now=self.now)
        self.assertFalse(self.store.snapshot()['style_cases'])
        self.assertEqual('boundary',self.store.snapshot()['facts'][0]['kind'])
        self.learn(rows)
        self.store.maintain(now=self.now+31*86400)
        self.assertFalse(self.store.snapshot()['style_cases'])
        self.assertEqual([],list(extract_cases(self.pair(),replace(self.policy,style_case_chars=5))))


class SocialRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_aliases_affect_both_reply_context_and_interjection_gate(self):
        with tempfile.TemporaryDirectory() as root:
            ctx=context(); route=route_for(ctx,Event())
            rt=Runtime(root,ctx,{'groups':[{'umo':route.umo,'capture_only':False,'interject_enabled':True}]})
            try:
                store,window=await rt.ensure(route)
                store.identify(route.platform,'A','当前群名片')
                declaration=Event('我的外号是小晴',mid='declaration',sender='A')
                await rt.observe(declaration)
                addressed=Event('小晴，你来吗',mid='vocative',sender='B')
                await rt.observe(addressed)
                self.assertEqual('addressed_to_other_member',rt.interjections.states[route.key].outcome)
                self.assertFalse(rt.interjections.states[route.key].pending)
                pack=build_pack(window,store.snapshot(),window.rows[-1],now=time.time())
                self.assertEqual('confirmed_name_vocative',pack['current_message']['addressing']['basis'])
                direct=Event('小晴是谁',mid='direct',sender='B');direct.is_at_or_wake_command=True
                prepared=await rt.prepare(direct)
                self.assertEqual(['bot'],prepared['current_message']['addressing']['targets'])
                self.assertTrue(any('小晴' in c.get('aliases',[]) for c in prepared['member_cards']))
                status=await rt.command(Event('/轻聊 状态',sender='A'))
                self.assertIn('小晴',status)
                await rt.command(Event('/轻聊 忘记我',sender='A'))
                self.assertEqual([],window.identity_snapshot['aliases'])
                self.assertEqual(0,ctx.llm_generate.call_count)
            finally:
                await rt.close()
