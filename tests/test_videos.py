"""Video sharing uses the same focus, freshness, receipt and group boundaries."""
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from test_ambient import Event
import test_unified as unified
from ambient.context import compact, route_for
from ambient.policy import normalized
from ambient.store import Store
from ambient.videos import BVID, VIDEO_RULE, big_image_card, parse_video, video_row


FIRST, SECOND = 'BV1234567890', 'BVabcdefghij'


def raw_video(identity=FIRST, title='企鹅跳舞，离谱游戏实况'):
    return {'bvid': identity, 'title': title, 'desc': '企鹅跳舞，游戏整活片段。',
            'pic': 'http://i0.hdslb.com/bfs/archive/0123456789.jpg',
            'owner': {'name': '作者'}, 'tname': '搞笑', 'duration': 60,
            'pubdate': int(time.time())-60, 'state': 0,
            'redirect_url': 'https://untrusted.invalid/not-a-video'}


class VideoDataTests(unittest.TestCase):
    def test_card_uses_fixed_template_canonical_link_and_only_bilibili_cover(self):
        row = video_row(raw_video())
        card = big_image_card(row)
        values = {r['key']: r['value'] for r in card['ark']['kv']}
        self.assertEqual(37, card['ark']['template_id'])
        self.assertEqual('https://www.bilibili.com/video/'+FIRST+'/', values['#METAURL#'])
        self.assertEqual('https://i0.hdslb.com/bfs/archive/0123456789.jpg', values['#METACOVER#'])
        self.assertEqual('哔哩哔哩 · 作者', values['#METASUBTITLE#'])
        self.assertIn(FIRST, card['text'])
        for bad in ('https://i0.hdslb.com.evil.test/bfs/a.jpg', 'https://127.0.0.1/a.jpg',
                    'https://user@i0.hdslb.com/bfs/a.jpg', 'https://i0.hdslb.com/bfs/../a.jpg',
                    'https://i0.hdslb.com/bfs/a.jpg?redirect=secret', 'file:///secret', ''):
            value = video_row({**raw_video(), 'pic': bad})
            self.assertEqual('', value['cover'])
            with self.assertRaisesRegex(ValueError, 'video_cover_unavailable'):
                big_image_card(value)

    def test_only_offered_ids_become_links_and_source_urls_are_discarded(self):
        raw = raw_video()
        row = video_row(raw)
        self.assertNotIn('redirect_url', row)
        self.assertEqual(('来点乐子', FIRST), parse_video('来点乐子 [[bilibili:'+FIRST+']]', {FIRST: row}))
        self.assertEqual(('看看这个', ''), parse_video('看看这个 [[bilibili:'+SECOND+']]', {FIRST: row}))
        self.assertIsNone(video_row({**raw, 'bvid': FIRST+'/../../file'}))
        self.assertIsNone(video_row({**raw, 'state': -4}))
        self.assertIsNone(video_row({**raw, 'duration': []}))
        self.assertEqual([], BVID.findall('prefix'+FIRST))


class VideoFlowTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = unified.UnifiedTests.asyncTearDown
    answer = unified.UnifiedTests.answer
    interject = unified.UnifiedTests.interject

    async def asyncSetUp(self):
        await unified.UnifiedTests.asyncSetUp(self)
        self.config['groups'][0]['video_sharing_enabled'] = True
        self.raw = {FIRST: raw_video(), SECOND: raw_video(SECOND, '猫咪离谱游戏实况')}
        self.rt.videos.candidates = {key: video_row(value) for key, value in self.raw.items()}
        self.rt.videos.fetched_at = time.time()
        self.api = AsyncMock(side_effect=lambda: {'list':list(self.raw.values())})
        self.patches.append(patch('ambient.videos.public_feed', self.api))
        self.patches[-1].start()
        await self.rt.ensure(self.route)

    async def test_reply_and_interjection_share_real_links_context_and_deduplication(self):
        await self.answer(unified.directed('发个搞笑视频', 'normal'), '这企鹅挺忙 [[bilibili:'+FIRST+']]')
        self.assertEqual('这企鹅挺忙\nhttps://www.bilibili.com/video/'+FIRST+'/', self.sent[0])
        await self.interject(Event('还有其他搞笑的没', mid='next'), '猫也没闲着 [[bilibili:'+SECOND+']]')
        pack = json.loads(self.ctx.llm_generate.call_args.kwargs['prompt'])['ambient_data']
        self.assertNotIn(FIRST, [v['id'] for v in pack['video_candidates']])
        self.assertIn(FIRST, compact(pack['recent_context']+pack['focus_context']))
        self.assertIn(VIDEO_RULE, self.ctx.llm_generate.call_args.kwargs['system_prompt'])
        self.assertEqual({FIRST, SECOND}, self.rt.stores[self.route.key].video_history())
        rows = [r for r in self.rt.windows[self.route.key].rows if r['self']]
        self.assertEqual({'reply', 'interjection'}, {r['source'] for r in rows})
        self.assertEqual([normalized(s) for s in self.sent], [r['text'] for r in rows])
        self.assertNotIn('[[bilibili:', ''.join(self.sent))
        self.assertEqual(2, self.api.await_count)

    async def test_new_target_during_source_check_cancels_old_interjection(self):
        async def interrupt():
            await self.rt.observe(unified.directed('先别分享，回答我', 'new-target'))
            return {'list':list(self.raw.values())}
        self.api.side_effect = interrupt
        await self.interject(Event('想看搞笑的', mid='old'), '这企鹅挺忙 [[bilibili:'+FIRST+']]')
        self.assertEqual([], self.sent)
        self.assertEqual(set(), self.rt.stores[self.route.key].video_history())

    async def test_disable_after_model_prevents_send(self):
        self.late_change = lambda: self.config['groups'][0].update(video_sharing_enabled=False)
        await self.answer(unified.directed('发点乐子'), '这企鹅挺忙 [[bilibili:'+FIRST+']]')
        self.assertEqual([], self.sent)
        self.assertEqual(set(), self.rt.stores[self.route.key].video_history())

    async def test_expired_offered_candidate_cannot_leave_an_orphan_caption(self):
        event = unified.directed('发点乐子')
        from test_ambient import Part, request
        from types import SimpleNamespace
        await self.rt.inject(event, request(), Part)
        self.rt.videos.fetched_at -= 3601
        await self.rt.response(event, SimpleNamespace(completion_text='这企鹅挺忙 [[bilibili:'+FIRST+']]'))
        self.assertTrue(event.stopped)
        self.assertEqual([], self.sent)
        self.api.assert_not_awaited()

    async def test_changed_source_or_unavailable_source_does_not_post_caption(self):
        self.api.side_effect = None
        self.api.return_value = {'list':[{**self.raw[FIRST], 'title': '被替换的视频'}]}
        await self.answer(unified.directed('发点乐子'), '这企鹅挺忙 [[bilibili:'+FIRST+']]')
        self.assertEqual([], self.sent)
        self.assertEqual('video_source_changed', self.rt.replies[self.route.key]['outcome'])
        self.api.side_effect = TimeoutError('private response should not appear')
        await self.answer(unified.directed('还有吗', 'other'), '企鹅上班了 [[bilibili:'+FIRST+']]')
        self.assertEqual([], self.sent)
        self.assertEqual(set(), self.rt.stores[self.route.key].video_history())

    async def test_unknown_selection_cannot_leak_marker_or_construct_arbitrary_link(self):
        await self.answer(unified.directed('来点乐子'), '今天先聊聊天 [[bilibili:BV0000000000]]')
        self.assertEqual(['今天先聊聊天'], self.sent)
        self.api.assert_not_awaited()
        self.assertEqual(set(), self.rt.stores[self.route.key].video_history())

    async def test_uncertain_link_is_not_repeated_after_restart_or_recorded_as_delivered(self):
        # Two real bubbles: caption is confirmed, link outcome is unknown.
        self.config['limits']['reply_segment_chars'] = 20
        self.late_change = lambda: setattr(self, 'error', TimeoutError() if self.sent else None)
        await self.answer(unified.directed('发个搞笑视频'), '企鹅开始上班了。 [[bilibili:'+FIRST+']]')
        self.assertEqual(2, len(self.sent))
        rows = [r for r in self.rt.windows[self.route.key].rows if r['self']]
        self.assertEqual(['企鹅开始上班了。'], [r['text'] for r in rows])
        store = self.rt.stores[self.route.key]
        reopened = Store(store.directory.parent, self.route.key, store.policy)
        self.assertIn(FIRST, reopened.video_history())
        with reopened.connection() as db:
            self.assertEqual('', db.execute('SELECT receipt FROM video_shares WHERE bvid=?', (FIRST,)).fetchone()[0])
        self.assertEqual('partial_no_retry', self.rt.replies[self.route.key]['outcome'])

    async def test_member_shared_links_and_group_isolation(self):
        await self.rt.observe(Event('https://www.bilibili.com/video/'+FIRST+'/', 'member-link'))
        pack = {'current_message': {'text': '搞笑视频'}}
        choices = self.rt.videos.add_choices(self.route, pack)
        self.assertNotIn(FIRST, choices)
        self.assertIn(SECOND, choices)
        other = route_for(self.ctx, Event(group='OTHER'))
        self.config['groups'].append({'umo': other.umo, 'capture_only': False, 'video_sharing_enabled': True})
        await self.rt.ensure(other)
        self.assertIn(FIRST, self.rt.videos.add_choices(other, {'current_message': {'text': '搞笑视频'}}))
        self.assertEqual([], self.rt.stores[self.route.key].snapshot()['members'])

    async def test_candidates_keep_current_focus_budget_and_expire(self):
        pack = {'current_message': {'text': '企鹅'}, 'focus_context': [], 'recent_context': [{'text': '猫咪'}]}
        choices = self.rt.videos.add_choices(self.route, pack)
        self.assertEqual(FIRST, next(iter(choices)))
        self.assertLessEqual(len(compact(pack)), self.rt.policy(self.route).pack_chars)
        large = {'current_message': {'text': 'x'*self.rt.policy(self.route).pack_chars}}
        self.assertEqual({}, self.rt.videos.add_choices(self.route, large))
        self.assertEqual({}, self.rt.videos.add_choices(self.route, {}, now=time.time()+3601))
        self.config['groups'][0]['video_topics'] = ['完全无关的关键词']
        self.assertEqual({}, self.rt.videos.add_choices(self.route, {}))

    async def test_atomic_claim_and_expiration(self):
        store = self.rt.stores[self.route.key]
        self.assertTrue(store.claim_video(FIRST, now=100000))
        self.assertFalse(store.claim_video(FIRST, now=100001))
        reopened = Store(store.directory.parent, self.route.key, store.policy)
        self.assertFalse(reopened.claim_video(FIRST, now=100002))
        self.assertTrue(reopened.claim_video(FIRST, now=100000+31*86400))

    async def test_public_refresh_is_bounded_nonblocking_and_does_not_call_model(self):
        self.api.side_effect = None
        self.api.return_value = {'list': [*self.raw.values(), {'bad': True}]}
        self.rt.videos.tick()
        task = self.rt.videos.task
        self.assertIsNotNone(task)
        await task
        self.assertEqual({FIRST, SECOND}, set(self.rt.videos.candidates))
        self.assertEqual('ready', self.rt.videos.outcome)
        self.rt.videos.tick()
        self.assertIs(task, self.rt.videos.task)
        self.ctx.llm_generate.assert_not_awaited()
        self.api.assert_awaited_once_with()

    async def test_invented_experience_remains_blocked_before_link_check(self):
        await self.interject(Event('发个搞笑视频'), '我上学的时候就看过 [[bilibili:'+FIRST+']]')
        self.assertEqual([], self.sent)
        self.api.assert_not_awaited()


class VideoCardFlowTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = unified.UnifiedTests.asyncTearDown
    answer = unified.UnifiedTests.answer
    interject = unified.UnifiedTests.interject

    async def asyncSetUp(self):
        await VideoFlowTests.asyncSetUp(self)
        self.config['groups'][0]['video_card_enabled'] = True
        self.wires = []
        owner = self
        class Sender:
            def __init__(self, *args): pass
            def check(self): pass
            async def send(self, text, guard, *, ark=None):
                if owner.late_change:
                    owner.late_change()
                await guard()
                owner.sent.append(text)
                owner.wires.append({'text': text, 'ark': ark})
                if owner.error:
                    raise owner.error
                return 'receipt-'+str(len(owner.wires))
        for module in ('runtime', 'interjection'):
            item = patch('ambient.'+module+'.Transport', Sender)
            self.patches.append(item)
            item.start()

    async def test_both_paths_send_caption_then_card_and_share_confirmed_context(self):
        await self.answer(unified.directed('发个搞笑视频', 'normal-card'), '企鹅上班了 [[bilibili:'+FIRST+']]')
        await self.interject(Event('换个搞笑的看看', mid='auto-card'), '猫也上班了 [[bilibili:'+SECOND+']]')
        self.assertEqual(['企鹅上班了', '', '猫也上班了', ''], self.sent)
        self.assertEqual([None, 37, None, 37], [w['ark']['template_id'] if w['ark'] else None for w in self.wires])
        pack = json.loads(self.ctx.llm_generate.call_args.kwargs['prompt'])['ambient_data']
        self.assertIn(FIRST, compact(pack['recent_context']+pack['focus_context']))
        self.assertNotIn('hdslb.com', compact(pack))
        rows = [r for r in self.rt.windows[self.route.key].rows if r['self']]
        self.assertEqual(['reply', 'reply', 'interjection', 'interjection'], [r['source'] for r in rows])
        self.assertEqual(big_image_card(self.rt.videos.candidates[FIRST])['text'].replace('\n', ' '), rows[1]['text'])
        with self.rt.stores[self.route.key].connection() as db:
            self.assertEqual(2, db.execute("SELECT count(*) FROM video_shares WHERE receipt<>''").fetchone()[0])

    async def test_card_failure_keeps_caption_without_video_receipt_or_retry(self):
        self.late_change = lambda: setattr(self, 'error', TimeoutError() if self.sent else None)
        event = unified.directed('发个搞笑视频', 'failed-card')
        await self.answer(event, '企鹅上班了 [[bilibili:'+FIRST+']]')
        self.assertEqual(['企鹅上班了', ''], self.sent)
        self.assertEqual('partial_no_retry', self.rt.replies[self.route.key]['outcome'])
        self.assertEqual(['企鹅上班了'], [r['text'] for r in self.rt.windows[self.route.key].rows if r['self']])
        store = self.rt.stores[self.route.key]
        reopened = Store(store.directory.parent, self.route.key, store.policy)
        with reopened.connection() as db:
            self.assertEqual('', db.execute('SELECT receipt FROM video_shares WHERE bvid=?', (FIRST,)).fetchone()[0])
        self.assertIn(FIRST, reopened.video_history())

    async def test_interjection_preempted_after_caption_does_not_send_old_card(self):
        def change():
            if self.sent:
                self.rt.interjections.states[self.route.key].generation += 1
        self.late_change = change
        await self.interject(Event('看看搞笑的', mid='preempt-card'), '企鹅上班了 [[bilibili:'+FIRST+']]')
        self.assertEqual(['企鹅上班了'], self.sent)
        self.assertEqual(1, len([r for r in self.rt.windows[self.route.key].rows if r['self']]))

    async def test_card_reserves_slot_and_exhausted_anchor_sends_no_orphan_caption(self):
        store = self.rt.stores[self.route.key]
        for _ in range(4): store.claim_qq_part('exhausted')
        await self.answer(unified.directed('发个视频', 'exhausted'), '企鹅上班了 [[bilibili:'+FIRST+']]')
        self.assertEqual([], self.sent)
        self.assertEqual('qq_reply_slots_exhausted', self.rt.replies[self.route.key]['outcome'])
        self.assertEqual(set(), store.video_history())
        for _ in range(3): store.claim_qq_part('two-slots')
        await self.answer(unified.directed('发个视频', 'two-slots'), '前半句。\n\n后半句。 [[bilibili:'+FIRST+']]')
        self.assertEqual(['前半句。\n\n后半句。', ''], self.sent)
        self.assertEqual(0, store.remaining_qq_parts('two-slots'))

    async def test_config_change_after_caption_stops_card(self):
        def change():
            if self.sent: self.config['groups'][0]['video_card_enabled'] = False
        self.late_change = change
        await self.answer(unified.directed('发个视频'), '企鹅上班了 [[bilibili:'+FIRST+']]')
        self.assertEqual(['企鹅上班了'], self.sent)

    async def test_no_cover_not_offered_and_onebot_retains_link_mode(self):
        from dataclasses import replace
        self.rt.videos.candidates[FIRST]['cover'] = ''
        pack = {}
        offered = self.rt.videos.add_choices(self.route, pack)
        self.assertNotIn(FIRST, offered)
        self.assertIn(SECOND, offered)
        self.assertNotIn('cover', pack['video_candidates'][0])
        self.assertIsNone(self.rt.videos.card(replace(self.route, adapter='aiocqhttp'), offered[SECOND]))


if __name__ == '__main__':
    unittest.main()
