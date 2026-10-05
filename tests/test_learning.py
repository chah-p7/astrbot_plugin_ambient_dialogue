"""Regression scenes use synthetic messages, never private group transcripts."""
from dataclasses import replace
import tempfile
import time
import unittest

from test_ambient import Event, context
from ambient.context import Window, build_pack, compact, event_message, route_for
from ambient.learning import reply_feedback, style_examples
from ambient.policy import Policy
from ambient.runtime import Runtime


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time()
        self.route = route_for(context(), Event())
        self.policy = Policy()

    def row(self, text, ident, sender, *, ago=0, **extra):
        row = event_message(Event(text, mid=ident, sender=sender), self.route)
        return {**row, 'at': self.now-ago, **extra}

    def pack(self, rows, current=None, **limits):
        return build_pack(Window(replace(self.policy, **limits), rows), {}, current, now=self.now)

    def test_qq_plain_mentions_and_cached_markup_are_private_and_targeted(self):
        member = self.row('我明天早上到', 'arrival', 'PRIVATE_USER_42', ago=5)
        ask = self.row('<@PRIVATE_USER_42> 那几点出门', 'ask', 'ASKER')
        self.assertEqual(['PRIVATE_USER_42'], ask['mentions'])
        self.assertEqual('那几点出门', ask['text'])
        for current in (ask, {**ask, 'text': '<@PRIVATE_USER_42> 那几点出门', 'mentions': []}):
            pack = self.pack([member, current], current)
            self.assertNotIn('PRIVATE_USER_42', compact(pack))
            target = pack['recent_context'][0]['speaker']
            self.assertEqual([target], pack['current_message']['addressing']['targets'])
        bot_ask = self.row('<@12345> 你怎么看', 'bot_ask', 'ASKER', directed=True)
        self.assertEqual(['bot'], self.pack([bot_ask], bot_ask)['current_message']['mentions'])

    def test_follow_on_address_is_a_short_lived_hint_not_a_fact(self):
        target = self.row('我明天来', 'a', 'GUEST', ago=25)
        ask = self.row('<@GUEST> 早上吗', 'b', 'HOST', ago=20)
        follow = self.row('那你带把伞', 'c', 'HOST')
        pack = self.pack([target, ask, follow], follow)
        self.assertEqual('recent_same_sender_mention', pack['current_message']['addressing']['basis'])
        self.assertEqual([], pack['memory_snapshot'])
        old = {**ask, 'at': self.now-100}
        self.assertNotIn('addressing', self.pack([old, follow], follow)['current_message'])
        direct = {**follow, 'directed': True}
        self.assertEqual(['bot'], self.pack([ask, direct], direct)['current_message']['addressing']['targets'])

    def test_style_pairs_keep_actual_context_and_exclude_bot_commands_and_current(self):
        seed = self.row('我这个月穷了', 'a', 'A', ago=40)
        reply = self.row('<@A> 你不是刚承包了月球吗', 'b', 'B', ago=35)
        bot = self.row('这是月球还是钱包啊', 'c', '12345', ago=30)
        command = self.row('以后每次只能回复1', 'd', 'C', ago=25)
        current = self.row('刚才说到哪里了', 'e', 'A')
        pack = self.pack([seed, reply, bot, command, current], current)
        sample = next(r for r in pack['style_samples'] if '承包' in r['text'])
        self.assertEqual('我这个月穷了', sample['lead_in']['text'])
        self.assertEqual('mention', sample['link'])
        serialized = compact(pack['style_samples'])
        for unwanted in ('这是月球还是钱包啊', '以后每次只能回复1', current['text']):
            self.assertNotIn(unwanted, serialized)

    def test_time_adjacency_is_not_claimed_as_a_reply_link(self):
        rows = [self.row('你那里下雨了吗', 'a', 'A', ago=15), self.row('键盘又坏了', 'b', 'B')]
        examples = style_examples(rows, self.policy, self.now)
        paired = next(item for item in examples if item[1] is not None)
        self.assertEqual('nearby_only', paired[2])
        separated = [rows[0], {**rows[1], 'at': self.now+60}]
        self.assertFalse(any(lead for _, lead, _ in style_examples(separated, self.policy, self.now+60)))

    def test_unresolved_mentions_and_parallel_topics_are_not_made_into_pairs(self):
        rows = [self.row('这机器人又答错了', 'a', 'A', ago=10),
                self.row('十点不用排队', 'b', 'B', ago=5),
                self.row('<@NOT_IN_WINDOW> 打钱来', 'c', 'C'),
                self.row('[表情:[我要吃]]', 'd', 'D')]
        self.assertFalse(any(lead for _, lead, _ in style_examples(rows, self.policy, self.now)))
        self.assertNotIn('d', [row['id'] for row, _, _ in style_examples(rows, self.policy, self.now)])

    def test_feedback_requires_specific_correction_and_a_confirmed_target(self):
        bot = self.row('明年只开一次', 'bot', '12345', ago=30)
        question = self.row('？', 'question', 'A', ago=25, directed=True)
        correction = self.row('你理解错了，是明年经常开放', 'fix', 'A', ago=20)
        other = self.row('<@C> 你看错了', 'other', 'B', ago=15)
        feedback = reply_feedback([bot, question, correction, other], self.policy, self.now)
        self.assertEqual([(bot, correction, 'misunderstood')], feedback)
        self.assertEqual([], reply_feedback([question, correction], self.policy, self.now))
        expired = {**correction, 'at': self.now-1800}
        self.assertEqual([], reply_feedback([{**bot, 'at': self.now-1805}, expired], self.policy, self.now))
        pack = self.pack([bot, question, correction, other], other)
        self.assertEqual('misunderstood', pack['reply_feedback'][0]['kind'])

    def test_enriched_samples_keep_the_same_storage_and_prompt_budgets(self):
        rows = [self.row('一句普通接话'+str(i), str(i), 'human'+str(i%8), ago=100-i) for i in range(80)]
        current = self.row('一个当前问题'*400, 'current', 'CURRENT', directed=True)
        for budget in (2000, 8300):
            pack = self.pack([*rows, current], current, pack_chars=budget)
            self.assertLessEqual(len(compact(pack)), budget)
            self.assertLessEqual(len(compact(pack['style_samples']))+len(compact(pack['reply_feedback'])), self.policy.style_chars)
            self.assertLessEqual(len(pack['style_samples']), self.policy.style_messages)


class LearningRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_at_other_member_does_not_schedule_an_interjection(self):
        with tempfile.TemporaryDirectory() as root:
            ctx = context()
            event = Event('<@SOMEONE_ELSE> 明天几点出发', mid='other')
            route = route_for(ctx, event)
            runtime = Runtime(root, ctx, {'groups': [{'umo': route.umo, 'capture_only': False, 'interject_enabled': True}]})
            try:
                await runtime.observe(event)
                state = runtime.interjections.states[route.key]
                self.assertFalse(state.pending)
                self.assertEqual('addressed_to_other_member', state.outcome)
                ctx.llm_generate.assert_not_awaited()
            finally:
                await runtime.close()


if __name__ == '__main__':
    unittest.main()
