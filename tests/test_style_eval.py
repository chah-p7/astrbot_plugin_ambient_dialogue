"""Keep held-out human answers out of both retrieval and recent context."""
import unittest

import test_learning
from ambient.context import compact
from tools.evaluate_style import blind_review, replay_case


class StyleEvalTests(unittest.TestCase):
    def test_replay_excludes_later_same_timestamp_answer_from_every_part(self):
        fixture = test_learning.LearningTests()
        fixture.setUp()
        past = fixture.row('昨天玩过一次', 'past', 'A', ago=10)
        current = fixture.row('好不好玩', 'current', 'B')
        answer = fixture.row('原答案只能用于事后对照', 'answer', 'A')
        case = {'id': 'game', 'rows': [past, current, answer], 'current_id': 'current', 'reference_id': 'answer'}
        result = replay_case(case)
        self.assertNotIn(answer['text'], compact(result))
        self.assertEqual(1, result['metrics']['held_out_rows'])
        self.assertEqual(current['text'], result['pack']['current_message']['text'])
        case['reference_id'] = 'past'
        with self.assertRaises(ValueError):
            replay_case(case)

    def test_blind_review_never_exposes_variant_names_or_assigns_a_winner(self):
        cases = [{'id': 'one', 'current': '刚睡醒', 'variants': {'old_model': '早', 'new_model': '醒了啊'}}]
        review, key = blind_review(cases, 3)
        self.assertNotIn('old_model', compact(review))
        self.assertNotIn('new_model', compact(review))
        self.assertEqual('', review[0]['winner'])
        for label in ('A', 'B'):
            self.assertEqual(cases[0]['variants'][key[0][label]], review[0][label])


if __name__ == '__main__':
    unittest.main()
