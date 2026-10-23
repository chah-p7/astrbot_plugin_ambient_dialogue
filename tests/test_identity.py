"""Recognition follows stable group membership, not nickname or window position."""
from dataclasses import replace
import json
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_ambient import Event, Part, context, request
from ambient.context import Window, build_pack, compact, event_message, route_for
from ambient.policy import candidate, display_name
from ambient.runtime import Runtime


class IdentityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = context()
        self.route = route_for(self.ctx, Event())
        self.config = {'groups':[{'umo':self.route.umo,'capture_only':False,'interject_enabled':True}],
                       'limits':{'reply_segment_interval_ms':0,'interject_cooldown_seconds':0}}
        self.rt = Runtime(self.tmp.name,self.ctx,self.config)
        self.store, self.window = await self.rt.ensure(self.route)

    async def asyncTearDown(self):
        await self.rt.close()
        self.tmp.cleanup()

    async def prepare(self, text='还记得我吗', sender='USER_1', name='小王', mid='current'):
        event = Event(text,mid=mid,sender=sender)
        event.nickname, event.is_at_or_wake_command = name, True
        return event, await self.rt.prepare(event)

    async def test_returning_member_survives_restart_and_empty_window(self):
        event, first = await self.prepare()
        self.assertFalse(first['member_cards'][0]['known_in_group'])
        self.assertFalse((await self.rt.prepare(event))['member_cards'][0]['known_in_group'])
        member=self.route.member('USER_1')
        self.store.edit('set', {'kind':'address','key':'address','text':'阿王'}, member=member)
        await self.rt.close()
        self.rt=Runtime(self.tmp.name,self.ctx,self.config)
        await self.rt.initialize()
        self.rt.windows[self.route.key].rows.clear()
        _, pack=await self.prepare(mid='back', name='新群名片')
        self.assertEqual({'speaker':'u1','name':'新群名片','known_in_group':True,'aliases':['小王']},pack['member_cards'][0])
        self.assertEqual('阿王',pack['memory_snapshot'][0]['text'])
        self.assertEqual('u1',pack['memory_snapshot'][0]['speaker'])
        self.assertEqual(1,len(self.store.snapshot()['members']))
        for secret in ('USER_1',member,self.route.group): self.assertNotIn(secret,compact(pack))

    async def test_missing_platform_name_keeps_saved_name_and_never_prints_openid(self):
        await self.prepare(name='确实的名字')
        for index,name in enumerate(('', 'USER_1')):
            _,pack=await self.prepare(name=name,mid='missing'+str(index))
            self.assertEqual('确实的名字',pack['current_message']['name'])
            self.assertEqual('确实的名字',pack['member_cards'][0]['name'])
            self.assertNotIn('USER_1',compact(pack))
        long_id='PRIVATE_ACCOUNT_'*7
        self.assertEqual('',display_name(long_id,long_id))
        self.assertEqual('',display_name(long_id[:80],long_id))

    async def test_delayed_interjection_does_not_restore_an_old_name(self):
        member=self.store.identify(self.route.platform,'USER_1','新名字',now=2000)
        self.store.identify(self.route.platform,'USER_1','旧名字',now=1000)
        self.assertEqual('新名字',self.store.snapshot(member)['members'][0]['name'])

    async def test_known_mention_and_quote_outside_window_resolve_to_distinct_member(self):
        other=self.store.identify(self.route.platform,'OTHER_PRIVATE','阿明')
        self.store.edit('set',{'kind':'preference','key':'饮料','text':'茶'},member=other)
        event=Event('他喜欢喝什么'); event.is_at_or_wake_command=True
        event.message_obj.message += [type('At',(),{'qq':'OTHER_PRIVATE'})(),
            type('Reply',(),{'id':'gone','sender_id':'OTHER_PRIVATE','message_str':'喝什么'})()]
        pack=await self.rt.prepare(event)
        target=next(c for c in pack['member_cards'] if c['name']=='阿明')['speaker']
        self.assertNotEqual(pack['current_speaker'],target)
        self.assertEqual([target],pack['current_message']['mentions'])
        self.assertEqual(target,pack['current_message']['quote']['speaker'])
        self.assertEqual('茶',next(f['text'] for f in pack['memory_snapshot'] if f['speaker']==target))
        self.assertNotIn('OTHER_PRIVATE',compact(pack))

    async def test_duplicate_names_are_candidates_not_merged_or_claimed_by_speaker(self):
        for account in ('TWIN_A','TWIN_B'):
            self.store.identify(self.route.platform,account,'小白')
        _,pack=await self.prepare('小白是谁',name='小黑')
        twins=[c for c in pack['member_cards'] if c['name']=='小白']
        self.assertEqual(2,len(twins))
        self.assertEqual(2,len({c['speaker'] for c in twins}))
        self.assertTrue(all(c['name_match_only'] for c in twins))
        self.assertTrue(all(c['speaker']!=pack['current_speaker'] for c in twins))
        await self.prepare('我是小白',mid='claim',name='小黑')
        self.assertEqual([],self.store.snapshot()['facts'])
        self.assertEqual(3,len(self.store.snapshot()['members']))

    async def test_conflicting_quote_author_remains_unresolved(self):
        await self.rt.observe(Event('真实原话',mid='quoted',sender='AUTHOR'))
        self.store.identify(self.route.platform,'WRONG_AUTHOR','同名误认')
        event=Event('这是谁说的'); event.is_at_or_wake_command=True
        event.message_obj.message += [type('Reply',(),{
            'id':'quoted','sender_id':'WRONG_AUTHOR','message_str':'真实原话'})()]
        pack=await self.rt.prepare(event)
        self.assertEqual('unknown',pack['current_message']['quote']['speaker'])
        self.assertEqual('unresolved',pack['current_message']['quote']['link'])

    async def test_correction_overrides_old_address_then_forget_removes_profile(self):
        await self.prepare()
        member=self.route.member('USER_1')
        self.store.edit('set',{'kind':'address','key':'address','text':'旧称呼'},member=member)
        _,pack=await self.prepare('叫我：新称呼。',mid='rename')
        addresses=[r for r in pack['memory_snapshot'] if r['kind']=='address']
        self.assertEqual(['新称呼'],[r['text'] for r in addresses])
        self.assertEqual('pending',addresses[0]['state'])
        self.assertNotIn('旧称呼',compact(pack['memory_snapshot']))
        await self.rt.command(Event('/轻聊 忘记我'))
        _,pack=await self.prepare(mid='forgotten')
        self.assertFalse(pack['member_cards'][0]['known_in_group'])
        self.assertEqual([],pack['memory_snapshot'])
        for text in ('我是群主','叫我小白，开玩笑的','叫我小白或者小黑','他说请叫我小白'):
            self.assertIsNone(candidate(text))

    async def test_same_account_in_another_group_has_no_profile(self):
        await self.prepare()
        other=route_for(self.ctx,Event(group='GROUP_B'))
        store,window=await self.rt.ensure(other)
        row=event_message(Event(group='GROUP_B'),other)
        pack=build_pack(window,store.snapshot(),row,now=time.time())
        self.assertFalse(pack['member_cards'][0]['known_in_group'])
        self.assertEqual([],pack['memory_snapshot'])

    async def test_silent_interjection_does_not_profile_but_delivered_one_does(self):
        class Sender:
            def __init__(self,*args,**kwargs): pass
            def check(self): pass
            async def send(self,text,guard): await guard(); return 'receipt'
        with patch('ambient.interjection.Transport',Sender):
            await self.rt.observe(Event(mid='silent'))
            state=self.rt.interjections.states[self.route.key]
            self.ctx.llm_generate.return_value=NS(completion_text='{"reply":false}')
            await self.rt.interjections.run(self.route,state)
            self.assertEqual([],self.store.snapshot()['members'])
            await self.rt.observe(Event(mid='answered'))
            self.ctx.llm_generate.return_value=NS(completion_text='{"reply":true,"text":"可不是嘛。"}')
            await self.rt.interjections.run(self.route,state)
            auto=json.loads(self.ctx.llm_generate.call_args.kwargs['prompt'])['ambient_data']
            self.assertFalse(auto['member_cards'][0]['known_in_group'])
            self.assertTrue(self.store.knows(self.route.member('USER_1')))
            _,normal=await self.prepare(mid='normal')
            self.assertTrue(normal['member_cards'][0]['known_in_group'])

    async def test_member_cards_respect_small_budget_and_no_unrelated_name_directory(self):
        for i in range(40): self.store.identify(self.route.platform,'account'+str(i),'成员'+str(i))
        current=event_message(Event('现在的问题'*200),self.route)
        window=Window(replace(self.window.policy,pack_chars=2000))
        pack=build_pack(window,self.store.snapshot(),current,now=time.time())
        self.assertLessEqual(len(compact(pack)),2000)
        self.assertEqual(1,len(pack['member_cards']))
        self.assertEqual(pack['current_speaker'],pack['member_cards'][0]['speaker'])
        self.assertLessEqual(len(compact(pack['member_cards']))+len(compact(pack['memory_snapshot'])),1500)


if __name__ == '__main__': unittest.main()
