"""Synthetic equivalents of lost followups and mistaken chat ownership."""
from dataclasses import replace
import tempfile
import time
import unittest

from test_ambient import Event, context
from ambient.context import Window, build_pack, compact, event_message, route_for
from ambient.learning import addressing
from ambient.policy import Policy, candidate
from ambient.runtime import Runtime


class RecallTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.route = route_for(context(), Event())

    def row(self, text, ident, sender='A', ago=0, **extra):
        return {**event_message(Event(text, mid=ident, sender=sender), self.route),
                'at': self.now-ago, **extra}

    def pack(self, rows, current, snapshot=None, **limits):
        return build_pack(Window(replace(Policy(), **limits), rows), snapshot or {}, current, now=self.now)

    def test_mention_survives_three_message_burst_and_has_no_durable_fact(self):
        start = self.row('<@TARGET> 哦对了', 'start', ago=29)
        detail = self.row('播放器发布了兼容性更新', 'detail', ago=10)
        current = self.row('你最好更新一下', 'current')
        pack = self.pack([start, detail, current], current)
        self.assertEqual('recent_same_sender_mention', pack['current_message']['addressing']['basis'])
        self.assertEqual([detail['text'], '@'+pack['current_message']['addressing']['targets'][0]+' 哦对了'],
                         [r['text'] for r in pack['focus_context']])
        self.assertEqual([], pack['memory_snapshot'])
        follow = self.row('可以等一下兼容补丁', 'follow', ago=-35)
        self.assertEqual((['TARGET'], 'recent_same_sender_mention'), addressing(follow, [start, detail, current, follow]))

    def test_quote_followup_keeps_old_source_and_original_addressee(self):
        source = self.row('这个版本缺了后半段', 'source', 'B', 1800)
        reply = self.row('看过差不多的版本', 'reply', ago=12,
                         quote={'id': '', 'sender': '', 'text': source['text']})
        reaction = self.row('[表情:闭嘴]', 'reaction', ago=10, attachment=True)
        other = self.row('晚饭出发', 'other', 'C', 5)
        current = self.row('没那么全', 'current')
        pack = self.pack([source, reply, reaction, other, current], current)
        self.assertEqual([reply['text'], source['text']], [r['text'] for r in pack['focus_context']])
        target = pack['focus_context'][1]['speaker']
        self.assertEqual([target], pack['current_message']['addressing']['targets'])
        self.assertEqual('recent_same_sender_quote', pack['current_message']['addressing']['basis'])

    def test_image_caption_keeps_media_without_guessing_its_contents(self):
        image = self.row('[图片]', 'image', ago=20, attachment=True)
        current = self.row('文化程度挺高', 'current')
        pack = self.pack([image, current], current)
        self.assertEqual('[图片]', pack['focus_context'][0]['text'])
        self.assertTrue(pack['focus_context'][0]['media_content_unavailable'])
        self.assertNotIn('addressing', pack['current_message'])

    def test_new_topic_and_expired_burst_do_not_inherit_target(self):
        start = self.row('<@TARGET> 哦对了', 'start', ago=29)
        changed = self.row('先不说这个，午饭吃什么', 'changed', ago=10)
        current = self.row('那吃什么', 'current')
        self.assertNotIn('addressing', self.pack([start, changed, current], current)['current_message'])
        full = self.row('今天晚饭只有白粥', 'full')
        complete = {**start, 'text': '<@TARGET> 播放器无法启动'}
        self.assertEqual([], self.pack([complete, full], full)['focus_context'])
        old = {**start, 'at': self.now-100}
        self.assertEqual(([], 'unspecified'), addressing(current, [old, current]))

    def test_direct_bot_request_overrides_inherited_person(self):
        start = self.row('<@TARGET> 这个版本不全', 'start', ago=4)
        current = self.row('那你说怎么办', 'current', directed=True)
        self.assertEqual((['bot'], 'directed_to_bot'), addressing(current, [start, current]))
        current.update(directed=False, quote={'id': 'missing', 'sender': '', 'text': start['text']})
        self.assertEqual(([], 'unspecified'), addressing(current, [start, current]))

    def test_explicit_recall_finds_old_original_without_turning_it_into_a_fact(self):
        old = self.row('缓存文件放在临时目录', 'old', 'B', 3500)
        unrelated = [self.row('午饭还有米粉'+str(i), str(i), 'C', 50-i) for i in range(30)]
        current = self.row('之前谁说过缓存文件', 'current', directed=True)
        pack = self.pack([old, *unrelated, current], current)
        self.assertEqual([old['text']], [r['text'] for r in pack['recall_context']])
        self.assertNotEqual(pack['current_speaker'], pack['recall_context'][0]['speaker'])
        self.assertEqual(3500, pack['recall_context'][0]['seconds_ago'])
        self.assertFalse(pack['memory_lookup']['complete_history'])
        self.assertEqual([], pack['memory_snapshot'])
        self.assertEqual(1, pack['memory_lookup']['matched_messages'])
        self.assertNotIn(old['text'], [r['text'] for r in pack['recent_context']])

    def test_author_scope_does_not_borrow_another_members_statement(self):
        mine = self.row('我买了吉他', 'mine', ago=1800)
        theirs = self.row('吉他周末送到', 'theirs', 'B', 1400)
        current = self.row('我之前说吉他怎么了', 'current', directed=True)
        pack = self.pack([mine, theirs, current], current)
        self.assertEqual([mine['text']], [r['text'] for r in pack['recall_context']])

    def test_explicit_target_is_distinct_from_asker(self):
        mine = self.row('电池要换了', 'mine', ago=1800)
        theirs = self.row('电池刚充满', 'theirs', 'B', 1400)
        current = self.row('<@bot> <@B> 之前说电池怎么了', 'current', directed=True)
        pack = self.pack([mine, theirs, current], current)
        self.assertEqual([theirs['text']], [r['text'] for r in pack['recall_context']])
        self.assertNotEqual(pack['current_speaker'], pack['recall_context'][0]['speaker'])

    def test_old_bot_recall_is_only_its_confirmed_reply_to_this_person(self):
        mine = self.row('网络连接超时', 'mine', ago=1805)
        ours = self.row('先检查本地网络', 'ours', '12345', 1800,
                        quote={'id': 'mine', 'sender': 'A', 'text': mine['text']})
        other = self.row('换个电池', 'other', '12345', 1700,
                         quote={'id': 'absent', 'sender': 'B', 'text': '电池坏了'})
        current = self.row('你之前说了什么', 'current', directed=True)
        pack = self.pack([mine, ours, other, current], current)
        self.assertEqual([ours['text'], mine['text']], [r['text'] for r in pack['recall_context']])

    def test_mentioned_subject_is_not_mistaken_for_the_author_to_search(self):
        statement = self.row('小林喜欢吉他', 'statement', 'B', 1800)
        target = self.row('今晚先不来了', 'target', 'TARGET', 1700)
        snapshot = {'members': [{'id': target['member'], 'account': 'TARGET', 'name': '小林', 'seen': target['at']}]}
        for text in ('之前谁说小林喜欢吉他', '之前谁说 <@TARGET> 喜欢吉他'):
            current = self.row(text, 'current', directed=True)
            pack = self.pack([statement, target, current], current, snapshot)
            self.assertEqual([statement['text']], [r['text'] for r in pack['recall_context']])

    def test_name_author_search_requires_unique_card(self):
        statement = self.row('电池充满了', 'statement', 'B', 1800, name='小林')
        snapshot = {'members': [{'id': statement['member'], 'account': 'B', 'name': '小林', 'seen': statement['at']}]}
        current = self.row('小林之前说电池怎么了', 'current', directed=True)
        pack = self.pack([statement, current], current, snapshot)
        self.assertEqual([statement['text']], [r['text'] for r in pack['recall_context']])

    def test_prior_sender_order_and_quoted_evidence_keep_account_scope(self):
        mine = self.row('缓存刚刚清理完', 'mine', ago=1500)
        other = self.row('电池要换了', 'other', 'B', 1400)
        current = self.row('之前我说了啥', 'current', directed=True)
        pack = self.pack([mine, other, current], current)
        self.assertEqual([mine['text']], [r['text'] for r in pack['recall_context']])
        statement = self.row('我说电池坏了', 'statement', 'B', 1800)
        current = self.row('之前谁说“我说电池坏了”', 'current', directed=True)
        pack = self.pack([statement, current], current)
        self.assertEqual([statement['text']], [r['text'] for r in pack['recall_context']])

    def test_casual_topic_does_not_retrieve_old_speech(self):
        old = self.row('缓存文件放在临时目录', 'old', ago=3500)
        current = self.row('缓存文件怎么清理', 'current', directed=True)
        pack = self.pack([old, current], current)
        self.assertFalse(pack['memory_lookup']['requested'])
        self.assertEqual([], pack['recall_context'])

    def test_expired_future_and_other_groups_are_not_recall_evidence(self):
        old = self.row('缓存文件放在临时目录', 'old', ago=7*3600)
        current = self.row('之前谁说过缓存文件', 'current', directed=True)
        future = self.row('缓存文件都删除了', 'future', ago=-1)
        pack = self.pack([old, current, future], current)
        self.assertEqual([], pack['recall_context'])
        other = self.row('另一个群没有旧记录', 'other')
        self.assertEqual([], self.pack([other], current)['recall_context'])

    def test_unknown_name_is_not_invented_from_unrelated_records(self):
        old = self.row('我们今天看过电影', 'old', 'B', 1800)
        current = self.row('周小北是谁', 'current', directed=True)
        pack = self.pack([old, current], current)
        self.assertEqual(0, pack['memory_lookup']['matched_messages'])
        self.assertEqual([], pack['memory_snapshot'])

    def test_retrieval_and_focus_share_original_prompt_budget(self):
        rows = [self.row('缓存文件'+str(i)+'内容'*100, str(i), 'B', 100-i) for i in range(25)]
        current = self.row('之前谁说过缓存文件', 'current', directed=True)
        for limit in (2000, 8300):
            pack = self.pack([*rows, current], current, pack_chars=limit)
            self.assertLessEqual(len(compact(pack)), limit)
            self.assertLessEqual(sum(len(compact(pack[k])) for k in ('focus_context', 'recall_context', 'recent_context')), 4500)
            self.assertEqual(len(pack['recall_context']), pack['memory_lookup']['matched_messages'])

    def test_natural_explicit_memory_requests_preserve_opt_in_and_correction(self):
        for text in ('记住，我叫小林', '请记一下：以后叫我小林', '记住，我的昵称是小林'):
            self.assertEqual('address', candidate(text)['kind'])
            self.assertEqual('address', candidate(text)['key'])
            self.assertEqual('小林', candidate(text)['text'])
        self.assertEqual('preference', candidate('记住，我喜欢柠檬茶')['kind'])
        for text in ('我叫小林', '我喜欢柠檬茶', '他说记住，我叫小林', '记住，我叫小林，开玩笑的'):
            self.assertIsNone(candidate(text))


class RecallRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_other_persons_burst_is_not_taken_over_but_direct_at_still_works(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = context()
            route = route_for(ctx, Event())
            rt = Runtime(root, ctx, {'groups': [{'umo': route.umo, 'capture_only': False, 'interject_enabled': True}]})
            try:
                for i, text in enumerate(('<@GUEST> 哦对了', '播放器有兼容性更新', '你最好更新一下')):
                    await rt.observe(Event(text, mid=str(i)))
                    self.assertFalse(rt.interjections.states[route.key].pending)
                state = rt.interjections.states[route.key]
                self.assertEqual('addressed_to_other_member', state.outcome)
                event = Event('<@bot> 记住，我叫小林', mid='direct')
                event.is_at_or_wake_command = True
                await rt.observe(event)
                pack = await rt.prepare(event)
                self.assertEqual(['bot'], pack['current_message']['addressing']['targets'])
                self.assertEqual('小林', pack['memory_snapshot'][0]['text'])
                self.assertEqual('pending', pack['memory_snapshot'][0]['state'])
                store, _ = await rt.ensure(route)
                self.assertEqual(1, len(store.snapshot()['members']))
            finally:
                await rt.close()
