"""Synthetic regressions for referents, quote chains and borrowed biographies."""
from dataclasses import replace
import json
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

from test_ambient import Event, Part, Plain, context, request
from ambient.context import Window, build_pack, compact, event_message, route_for
from ambient.learning import addressing, style_examples
from ambient.policy import Policy, reply_rejection
from ambient.runtime import Runtime


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.route = route_for(context(), Event())

    def row(self, text, ident, sender, ago=0, **values):
        row = event_message(Event(text, mid=ident, sender=sender), self.route)
        return {**row, 'at': self.now-ago, **values}

    def pack(self, rows, current, **limits):
        return build_pack(Window(replace(Policy(), **limits), rows), {}, current, now=self.now)

    def test_comparison_and_trailing_reaction_keep_the_speakers_complete_turn(self):
        first = self.row('模型甲今天挺聪明', 'first', 'A', 31)
        second = self.row('模型乙最近脑子不行', 'second', 'A', 2)
        current = self.row('[表情:太好笑]', 'reaction', 'A')
        pack = self.pack([first, second, current], current)
        self.assertEqual([first['text'], second['text'], current['text']], [r['text'] for r in pack['current_turn']])
        self.assertEqual([31,2,0], [r['seconds_ago'] for r in pack['current_turn']])
        self.assertTrue(pack['current_message']['reaction_only'])
        self.assertEqual(1, len({r['speaker'] for r in pack['current_turn']}))
        self.assertEqual([], pack['focus_context'])

    def test_turn_stops_at_another_speaker_quote_new_addressee_and_explicit_topic_change(self):
        first = self.row('模型甲今天挺聪明', 'first', 'A', 15)
        other = self.row('外卖到了', 'other', 'B', 5)
        for current in (self.row('[表情:笑]', 'r', 'A'),
                        self.row('换个话题，午饭吃什么', 'new', 'A'),
                        self.row('<@B> 午饭吃什么', 'target', 'A'),
                        self.row('这句说谁', 'quote', 'A', quote={'id':'other','sender':'B','text':other['text']})):
            rows = [first,other,current] if current['id']=='r' else [first,current]
            self.assertNotIn('current_turn', self.pack(rows,current))

    def test_reaction_retains_an_explicit_target_without_treating_it_as_a_new_question(self):
        first = self.row('<@B> 你这次跑通了吗', 'first', 'A', 2)
        current = self.row('[表情:笑]', 'reaction', 'A')
        self.assertEqual((['B'],'recent_same_sender_reaction'), addressing(current,[first,current]))
        pack = self.pack([first,current],current)
        self.assertNotEqual(['bot'],pack['current_message']['addressing']['targets'])

    def test_turn_is_bounded_and_never_reads_later_messages(self):
        rows = [self.row('这一条短句'+str(i), str(i), 'A', 50-i*10) for i in range(6)]
        current = rows[-1]
        later = self.row('后来的结果不该看到', 'later', 'A', -1)
        pack = self.pack([*rows,later],current,pack_chars=2000)
        self.assertLessEqual(len(compact(pack)),2000)
        self.assertLessEqual(len(pack.get('current_turn',[])),4)
        self.assertNotIn('后来的结果',compact(pack))
        self.assertNotIn(rows[0]['text'],[r['text'] for r in pack.get('current_turn',[])])

    def test_long_reaction_burst_does_not_recurse_or_inherit_an_old_target(self):
        first = self.row('<@B> 你跑通了吗', 'first', 'A', 1501)
        reactions = [self.row('[表情:笑]', str(i), 'A', 1500-i) for i in range(1501)]
        self.assertEqual(([], 'unspecified'), addressing(reactions[-1],[first,*reactions]))

    def test_inline_mention_keeps_its_subject_and_direct_reply_targets_bot(self):
        target = self.row('这课还得上吗', 'seed', 'PRIVATE_TARGET', 10)
        for native in (False, True):
            event = Event('<@12345> 你戳到 <@PRIVATE_TARGET> 的痛处了')
            event.is_at_or_wake_command = True
            if native:
                At = type('At', (), {'__init__': lambda self, qq: setattr(self, 'qq', qq)})
                event.message_obj.message = [At('12345'), Plain('你戳到'), At('PRIVATE_TARGET'), Plain('的痛处了')]
            current = event_message(event, self.route, now=self.now)
            pack = self.pack([target, current], current)
            alias = pack['recent_context'][0]['speaker']
            self.assertEqual('@bot 你戳到 @'+alias+' 的痛处了', pack['current_message']['text'])
            self.assertEqual(['bot'], pack['current_message']['addressing']['targets'])
            self.assertEqual(['bot', alias], pack['current_message']['mentions'])
            self.assertNotIn('PRIVATE_TARGET', compact(pack))

    def test_unseen_mentioned_people_remain_distinct_without_profiles(self):
        current = self.row('请比较 <@UNSEEN_A> 和 <@UNSEEN_B> 的说法', 'current', 'ASKER', directed=True)
        pack = self.pack([current], current)
        a, b = pack['current_message']['mentions']
        self.assertNotEqual(a, b)
        self.assertNotIn('UNSEEN_', compact(pack))
        self.assertEqual([], pack['memory_snapshot'])
        self.assertTrue(all(not c['known_in_group'] for c in pack['member_cards']))

    def test_missing_native_nickname_uses_same_accounts_last_valid_name(self):
        previous = self.row('刚刚到家', 'previous', 'A', 10, name='小雨')
        nameless = self.row('先歇一会', 'nameless', 'A', 5, name='')
        current = self.row('<@A> 到家了吗', 'current', 'B')
        pack = self.pack([previous, nameless, current], current)
        target = pack['current_message']['mentions'][0]
        self.assertEqual('小雨', next(c['name'] for c in pack['member_cards'] if c['speaker']==target))

    def test_inline_subject_is_not_an_addressee_but_leading_mention_is(self):
        subject = self.row('今天 <@PERSON> 也来了', 'subject', 'A')
        self.assertEqual(([], 'unspecified'), addressing(subject, [subject]))
        addressed = self.row('<@PERSON> 明天来吗', 'addressed', 'A')
        self.assertEqual((['PERSON'], 'explicit_mention'), addressing(addressed, [addressed]))
        old = {**subject, 'text': '今天 也来了'}
        pack = self.pack([old], old)
        self.assertTrue(pack['current_message']['mention_positions_unknown'])
        self.assertEqual('mention_positions_unknown', pack['current_message']['addressing']['basis'])

    def test_reply_ancestors_survive_busy_room_and_twenty_minute_cutoff(self):
        start = self.row('我的导师让交报告', 'start', 'A', 1800)
        ours = self.row('那你先交报告', 'ours', '12345', 1700,
                        quote={'id': 'start', 'sender': 'A', 'text': start['text']})
        unrelated = [self.row('别的话题'+str(i), 'noise'+str(i), 'B', 100-i) for i in range(40)]
        current = self.row('这句说的是谁', 'current', 'C', directed=True,
                           quote={'id': 'ours', 'sender': '12345', 'text': ours['text']})
        pack = self.pack([start, ours, *unrelated, current], current)
        focus = pack['focus_context']
        self.assertEqual([ours['text'], start['text']], [r['text'] for r in focus])
        self.assertEqual(['direct_quote', 'quote_ancestor'], [r['relevance'] for r in focus])
        self.assertEqual(focus[1]['ref'], focus[0]['quote']['ref'])
        self.assertEqual(focus[1]['speaker'], focus[0]['quote']['speaker'])
        self.assertNotEqual(pack['current_speaker'], focus[1]['speaker'])
        self.assertEqual(1700, focus[0]['seconds_ago'])
        self.assertLessEqual(len(compact(focus))+len(compact(pack['recent_context'])), 4500)

    def test_same_person_fragment_stays_a_hint_and_excludes_later_messages(self):
        first = self.row('[图片]', 'first', 'A', 8, attachment=True)
        other = self.row('晚饭吃什么', 'other', 'B', 5)
        current = self.row('评价一下', 'current', 'A', directed=True)
        later = self.row('后来的答案不能泄漏', 'later', 'B')
        pack = self.pack([first, other, current, later], current)
        self.assertEqual('same_speaker_recent', pack['focus_context'][0]['relevance'])
        self.assertEqual('[图片]', pack['focus_context'][0]['text'])
        self.assertNotIn('后来的答案', compact(pack))
        self.assertNotIn('addressing', pack['focus_context'][0])
        small = self.pack([first, other, current, later], current, pack_chars=2000)
        self.assertLessEqual(len(compact(small)), 2000)

    def test_bot_replies_to_other_members_do_not_become_my_last_exchange(self):
        other = self.row('我忘交报告了', 'other', 'B', 30)
        ours = self.row('那你赶紧补', 'ours', '12345', 20,
                        quote={'id': 'other', 'sender': 'B', 'text': other['text']})
        current = self.row('你说谁呢', 'current', 'A', directed=True)
        self.assertEqual([], self.pack([other, ours, current], current)['focus_context'])

    def test_unrelated_same_account_exchange_stays_in_background(self):
        start = self.row('测试接口启动不了', 'start', 'A', 160)
        ours = self.row('哪个服务启动不了', 'ours', '12345', 140,
                        quote={'id': 'start', 'sender': 'A', 'text': start['text']})
        meal = self.row('饭店套餐只要六块钱', 'meal', 'A', 17)
        current = self.row('这点钱够成本吗', 'current', 'A', directed=True)
        pack = self.pack([start, ours, meal, current], current)
        self.assertEqual([meal['text']], [r['text'] for r in pack['focus_context']])
        self.assertIn(ours['text'], [r['text'] for r in pack['recent_context']])

    def test_complete_new_topic_does_not_pin_even_an_immediate_same_sender_line(self):
        start = self.row('这个服务今天也启动不了', 'start', 'A', 5)
        current = self.row('今天的晚餐我也吃不了', 'current', 'A')
        self.assertEqual([], self.pack([start, current], current)['focus_context'])

    def test_same_topic_reply_and_explicit_recall_keep_verified_anchor(self):
        start = self.row('接口连接超时', 'start', 'A', 130)
        ours = self.row('先检查网络', 'ours', '12345', 120,
                        quote={'id': 'start', 'sender': 'A', 'text': start['text']})
        for text in ('接口连接还是超时', '你刚才说了什么'):
            current = self.row(text, 'current', 'A', directed=True)
            focus = self.pack([start, ours, current], current)['focus_context']
            self.assertEqual([ours['text'], start['text']], [r['text'] for r in focus])
            self.assertEqual(['last_reply_to_speaker', 'reply_anchor'], [r['relevance'] for r in focus])

    def test_vague_followup_cannot_jump_over_another_conversation(self):
        start = self.row('接口连接超时', 'start', 'A', 40)
        ours = self.row('先检查网络', 'ours', '12345', 20,
                        quote={'id': 'start', 'sender': 'A', 'text': start['text']})
        current = self.row('为什么', 'current', 'A', directed=True)
        self.assertIn(ours['text'], [r['text'] for r in self.pack([start, ours, current], current)['focus_context']])
        other = self.row('午饭店关门了', 'other', 'B', 5)
        self.assertEqual([], self.pack([start, ours, other, current], current)['focus_context'])

    def test_topic_change_stops_implicit_focus_but_not_an_explicit_quote(self):
        start = self.row('接口报错', 'start', 'A', 60)
        ours = self.row('接口日志贴一下', 'ours', '12345', 50,
                        quote={'id': 'start', 'sender': 'A', 'text': start['text']})
        changed = self.row('先不聊接口，晚饭吃什么', 'changed', 'A', 10)
        current = self.row('那吃什么', 'current', 'A')
        focus = self.pack([start, ours, changed, current], current)['focus_context']
        self.assertEqual([changed['text']], [r['text'] for r in focus])
        current['quote'] = {'id': 'ours', 'sender': '12345', 'text': ours['text']}
        self.assertEqual([ours['text'], start['text']], [r['text'] for r in self.pack([start, ours, changed, current], current)['focus_context']])

    def test_short_trial_followup_keeps_its_source_without_older_topic(self):
        old = self.row('晚饭要吃面条', 'old', 'A', 20)
        source = self.row('不知道这接口能不能连上', 'source', 'A', 3)
        current = self.row('我来试试', 'current', 'A')
        self.assertEqual([source['text']], [r['text'] for r in self.pack([old, source, current], current)['focus_context']])

    def test_style_selection_does_not_backfill_unrelated_topics(self):
        source = self.row('这门实验课有用吗', 'source', 'A', 90)
        reply = self.row('有用，治失眠', 'reply', 'B', 80,
                         quote={'id': 'source', 'sender': 'A', 'text': source['text']})
        unrelated = self.row('烤肉店明天中午营业', 'meal', 'C', 5)
        current = self.row('实验课到底有没有用', 'current', 'D', directed=True)
        selected = style_examples([source, reply, unrelated, current], Policy(), self.now, current)
        self.assertIn('reply', [s.row['id'] for s in selected])
        self.assertNotIn('meal', [s.row['id'] for s in selected])

    def test_real_world_first_person_claims_are_blocked_but_quotes_are_not(self):
        current = self.row('实验课有没有用', 'current', 'A', directed=True)
        window = Window(Policy(), [current])
        for text in ('我们学校的实验课也没用', '我们那实验课就是走个形式', '我室友昨天也这样', '确实，我们公司也这么干'):
            self.assertEqual('invented_personal_experience', reply_rejection(text, current, window))
            self.assertEqual('invented_personal_experience', reply_rejection(text, current, window, interject=True))
        for text in ('你说的“我们学校”是哪所', '我没有学校', '我们学校是编的，我刚才说错了',
                     '我们群还挺热闹', '我觉得这课没意思', '“我室友昨天也这样”是他原话'):
            self.assertEqual('', reply_rejection(text, current, window), text)

    def test_embedded_school_and_work_experiences_are_rejected_on_both_paths(self):
        current = self.row('先去听课了，许愿工具自己跑完', 'current', 'A', directed=True)
        window = Window(Policy(), [current])
        for text in ('祈祷真有用我早不用写作业了', '要是能自动做完我今天也不用交作业了',
                     '我马上要去上课了', '我以前读书的时候也这样', '我当年读过大学',
                     '我昨晚还在公司', '我也得去上班', '哈哈我室友也这么说',
                     '我没写完作业', '我不想上学', '我刚毕业', '我们学校没有食堂', '我室友没有交作业',
                     '我在读大学', '我读的大学可没有这些', '我一会儿需要写作业', '我昨晚交论文去了'):
            for auto in (False, True):
                self.assertEqual('invented_personal_experience', reply_rejection(text, current, window, interject=auto), text)

    def test_help_opinion_denial_and_explicit_fiction_are_not_biography(self):
        current = self.row('帮忙看看作业', 'current', 'A', directed=True)
        window = Window(Policy(), [current])
        for text in ('我帮你写作业', '我可以解释这道题', '我觉得上课确实挺累',
                     '我不用上学，也不需要写作业', '我从来没上过学', '我没有上过课', '我又没上过学',
                     '他刚才说“我以前读书的时候也这样”', '他原话是「我要去上班了」',
                     '如果我是学生，我也得写作业', '虚构例子：我当年读书时挺认真',
                     '让我写作业是吧，题目贴一下', '我写作业批改脚本',
                     '我正在查资料', '我刚刚发了两条消息', '我读这份报告再帮你看', '我读书给你听',
                     '我写论文会先列提纲', '我写完实验报告了，你看一下'):
            for auto in (False, True):
                self.assertEqual('', reply_rejection(text, current, window, interject=auto), text)

    def test_a_quote_or_fiction_does_not_hide_a_separate_personal_claim(self):
        current = self.row('说说这门课', 'current', 'A', directed=True)
        window = Window(Policy(), [current])
        for text in ('他原话是“我要上课了”，我也要上课',
                     '如果我是学生，我也得写作业。我以前上学时就是这样'):
            self.assertEqual('invented_personal_experience', reply_rejection(text, current, window), text)

    def test_generic_pronouns_cannot_retrieve_an_unrelated_old_topic(self):
        old = self.row('我自己在折腾编译器', 'old', 'A', 600)
        recent = self.row('刚回来，差点错过', 'recent', 'B', 5)
        current = self.row('那你自己呢', 'current', 'C', directed=True)
        examples = style_examples([old, recent, current], Policy(), self.now, current)
        self.assertEqual('recent', examples[0].row['id'])


class GroundingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_prompt_aliases_native_identity_and_preserves_attachments(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = context()
            event = Event('<@12345> 你戳到 <@TARGET> 的痛处了')
            event.is_at_or_wake_command = True
            event.message_obj.message.append(type('Reply', (), {'id': 'seed', 'sender_id': 'TARGET', 'message_str': '原话'})())
            rt = Runtime(root, ctx, {'groups': [{'umo': event.unified_msg_origin, 'capture_only': False}]})
            try:
                await rt.observe(Event('原话', mid='seed', sender='TARGET'))
                req = request()
                req.prompt = event.message_str
                req.image_urls, req.audio_urls = ['local-image.png'], ['local-audio.wav']
                reminder = Part('<system_reminder>User ID: USER_1, Nickname: 小王\nCurrent datetime: now</system_reminder>')
                quoted = Part('<Quoted Message>\n原话\n[Image Caption in quoted message]: 红色卡通形象\n</Quoted Message>')
                req.extra_user_content_parts.extend([reminder, quoted])
                await rt.inject(event, req, Part)
                pack = json.loads(req.extra_user_content_parts[-1].text)['ambient_data']
                self.assertEqual(pack['current_message']['text'], req.prompt)
                rendered = req.prompt+' '.join(p.text for p in req.extra_user_content_parts)
                self.assertNotIn('USER_1', rendered)
                self.assertNotIn('TARGET', rendered)
                self.assertIn('红色卡通形象', rendered)
                self.assertIn('Current datetime: now', rendered)
                self.assertIn('其他插件', rendered)
                self.assertIn('USER_1', reminder.text)  # Do not mutate the host's original objects.
                ctx.llm_generate.return_value = NS(completion_text='你这课听着就够折腾的')
                answer = await rt.checked_reply(event, rt.routes[next(iter(rt.routes))], '我们学校的课也没用')
                self.assertEqual('你这课听着就够折腾的', answer)
                args = ctx.llm_generate.call_args.kwargs
                self.assertEqual(['local-image.png'], args['image_urls'])
                self.assertEqual(['local-audio.wav'], args['audio_urls'])
                self.assertIn('红色卡通形象', args['prompt'])
                ctx.llm_generate.assert_awaited_once()
            finally:
                await rt.close()


if __name__ == '__main__':
    unittest.main()
