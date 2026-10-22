"""Card collection regressions; fixtures describe supported payload shapes, not live receipts."""
import json
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from test_ambient import Event
import test_unified as unified
from test_videos import FIRST, SECOND, raw_video
from ambient.context import event_message
from ambient.video_sources import references, resolve_reference
from ambient.videos import video_row


def miniapp(link='https://b23.tv/AbCd123', title='奶蛙小剧场'):
    return {'app':'com.tencent.miniapp_01','desc':'新闻','meta':{'detail_1':{
        'title':'哔哩哔哩','desc':title,'qqdocurl':link,'preview':'https://image.invalid/'+SECOND}}}


class CardTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = unified.UnifiedTests.asyncSetUp
    asyncTearDown = unified.UnifiedTests.asyncTearDown

    async def setup_video(self):
        self.config['groups'][0].update(video_sharing_enabled=True,video_cover_selection_enabled=False)
        await self.rt.ensure(self.route)
        self.taste = self.rt.stores[self.route.key].video_taste

    def event(self, data, text='', mid='card', sender='human'):
        e=Event(text, mid=mid, sender=sender)
        e.message_obj.raw_message.raw_data = data
        return e

    async def test_native_raw_card_only_without_at_is_collected_and_has_context(self):
        await self.setup_video()
        e=self.event({'message_type':110,'msg_elements':[{'type':'json','data':json.dumps(miniapp(),ensure_ascii=False)}]})
        await self.rt.observe(e,request_type=NS)
        row=list(self.rt.windows[self.route.key].rows)[-1]
        self.assertIn('奶蛙小剧场',row['text'])
        self.assertEqual('https://b23.tv/AbCd123',row['video_refs'][0]['ref'])
        self.assertEqual(self.route.member('human'),self.taste.examples(now=time.time())[0]['member'])
        self.assertIsNone(e.get_extra('provider_request'))
        self.ctx.llm_generate.assert_not_awaited()
        self.assertEqual([],self.sent)
        state=self.rt.videos.selection.states[self.route.key]
        self.assertEqual(1,state['video_card_messages'])
        self.assertEqual(1,state['human_shares_collected'])
        self.assertNotIn('AbCd123',json.dumps(state))

    async def test_caption_and_card_are_both_visible_and_directed_empty_card_enters_host(self):
        await self.setup_video()
        e=self.event({'ark':miniapp()},text='看这个',mid='caption')
        row=event_message(e,self.route)
        self.assertIn('看这个',row['text'])
        self.assertIn('奶蛙小剧场',row['text'])
        empty=self.event({'ark':miniapp()},mid='at-card')
        empty.is_at_or_wake_command=True
        await self.rt.observe(empty,request_type=NS)
        self.assertIn('奶蛙小剧场',empty.get_extra('provider_request').prompt)

    async def test_quotes_and_identity_fields_are_not_human_share_evidence(self):
        await self.setup_video()
        raw={'message_type':103,'msg_elements':[{'id':FIRST,'author':{'id':SECOND},'ark':miniapp()}],
             'message_reference':{'content':miniapp()},'author':{'id':FIRST},'attachments':[{'url':'https://b23.tv/AbCd123'}]}
        self.assertEqual([],references('',raw=raw))
        self.assertEqual([],references('',[{'type':'Reply','data':miniapp()},{'type':'Forward','data':miniapp()}]))
        raw['msg_elements'].append({'ark':miniapp('https://www.bilibili.com/video/'+SECOND+'/','新的分享')})
        refs=references('',raw=raw)
        self.assertEqual([{'ref':SECOND,'title':'新的分享'}],refs)
        for author in ({'bot':True},{'is_you':True}):
            row=event_message(self.event({'ark':miniapp(),'author':author}),self.route)
            self.assertEqual(0,self.taste.observe(row))

    async def test_nested_json_xml_ark_and_percent_encoded_link(self):
        card=miniapp('https://www.bilibili.com/video/'+FIRST+'/?share_source=qq')
        nested={'type':'json','data':{'data':json.dumps(json.dumps(card,ensure_ascii=False),ensure_ascii=False)}}
        self.assertEqual([{'ref':FIRST,'title':'奶蛙小剧场'}],references('',[nested]))
        xml='<msg url="https://b23.tv/AbCd123"><item><title>小剧场 &amp; 二创</title></item></msg>'
        self.assertEqual([{'ref':'https://b23.tv/AbCd123','title':'小剧场 & 二创'}],references('',[{'type':'xml','data':{'data':xml}}]))
        ark={'template_id':37,'kv':[{'key':'#METATITLE#','value':'奶蛙'},
                                   {'key':'#METAURL#','value':'https%3A%2F%2Fwww.bilibili.com%2Fvideo%2F'+FIRST+'%2F'}]}
        self.assertEqual([{'ref':FIRST,'title':'奶蛙'}],references('',raw={'ark':ark}))

    async def test_miniapp_avid_and_failure_keeps_title_without_inventing_bvid(self):
        await self.setup_video()
        card=miniapp();detail=card['meta']['detail_1'];del detail['qqdocurl']
        detail['url']='pages/video/video?avid=12345678'
        row=event_message(self.event({'miniapp':card}),self.route)
        self.assertEqual('av12345678',row['video_refs'][0]['ref'])
        self.taste.observe(row)
        with patch('ambient.video_sources.public_json',AsyncMock(side_effect=ValueError('video_source_unavailable'))):
            await self.rt.videos.selection.refresh(self.route)
        example=self.taste.examples(now=time.time())[0]
        self.assertEqual('',example['video']['id'])
        self.assertEqual('奶蛙小剧场',example['video']['title'])
        with patch('ambient.video_sources.public_json',AsyncMock(return_value={'bvid':FIRST})) as api:
            self.assertEqual(FIRST,await resolve_reference('av12345678'))
            api.assert_awaited_once_with('view',{'aid':'12345678'})

    async def test_malformed_large_or_unrelated_cards_do_not_fail_chat(self):
        for value in ('{"broken":', '{'*2000, 'a'*100000, {'meta':{'url':'https://b23.tv.evil.test/AbCd123'}},
                      miniapp('https://[invalid'), {'kv':[{'key':[],'value':'invalid'}]},
                      '<!DOCTYPE x [<!ENTITY x "boom">]><x>&x;</x>'):
            self.assertEqual([],references('',raw={'ark':value}))
        cyclic={};cyclic['data']=cyclic
        self.assertEqual([],references('',raw={'ark':cyclic}))
        self.assertEqual([],references('',raw={'ark':{'id':FIRST,'author':{'name':FIRST}}}))

    async def test_unresolved_structured_messages_are_observable_without_raw_storage(self):
        await self.setup_video()
        e=self.event({'ark':{'secret':'PRIVATE_CARD_TEXT'}})
        await self.rt.observe(e)
        state=self.rt.videos.selection.states[self.route.key]
        self.assertEqual(1,state['empty_messages'])
        self.assertEqual(1,state['unresolved_structured_messages'])
        self.assertEqual(['ark'],state['last_carriers'])
        self.assertNotIn('PRIVATE_CARD_TEXT',json.dumps(self.rt.status()))
        self.assertEqual([],list(self.rt.windows[self.route.key].rows))

    async def test_backfill_reads_retained_raw_cards_and_excludes_quote(self):
        await self.setup_video()
        rows=[NS(id='raw-card',platform_id=self.route.platform,user_id=self.route.group,sender_id='human',created_at=time.time(),
                 content=json.dumps({'content':'','ark':miniapp()})),
              NS(id='nested',platform_id=self.route.platform,user_id=self.route.group,sender_id='human',created_at=time.time(),
                 content={'raw_message':{'raw_data':{'ark':miniapp()}}}),
              NS(id='quote',platform_id=self.route.platform,user_id=self.route.group,sender_id='other',created_at=time.time(),
                 content={'raw_data':{'message_type':103,'msg_elements':[{'ark':miniapp()}]}})]
        self.ctx.message_history_manager=NS(get=AsyncMock(side_effect=[rows,[]]))
        await self.rt.videos.selection.backfill(self.route)
        examples=self.taste.examples(now=time.time())
        self.assertEqual(1,len(examples))
        self.assertEqual(self.route.member('human'),examples[0]['member'])

    async def test_explicit_keyword_discovery_is_separate_from_human_learning(self):
        await self.setup_video()
        self.config['groups'][0].update(video_preferred_keywords=['奶蛙'],video_discovery_seeds=[FIRST,'invalid'],video_learning_enabled=False)
        self.rt.videos.candidates={FIRST:video_row(raw_video(FIRST,'服务器配置教程'))}
        self.rt.videos.fetched_at=time.time()
        with patch('ambient.video_selection.related_feed',AsyncMock(return_value=[raw_video(SECOND,'奶蛙上学')])) as feed:
            await self.rt.videos.selection.refresh(self.route)
        feed.assert_awaited_once_with(FIRST)
        self.assertEqual([],self.taste.examples(now=time.time()))
        state=self.rt.videos.selection.states[self.route.key]
        self.assertEqual(1,state['preferred_candidates'])
        pack={'current_message':{'text':'来点视频'}}
        offered=self.rt.videos.add_choices(self.route,pack)
        self.assertEqual(SECOND,next(iter(offered)))
        self.assertEqual(['奶蛙'],pack['video_candidates'][0]['requested_keywords'])
        self.assertNotIn('group_share_similarity',pack['video_candidates'][0])
        current={'current_message':{'text':'服务器配置'}}
        self.assertEqual(FIRST,next(iter(self.rt.videos.add_choices(self.route,current))))
        with patch('ambient.videos.related_feed',AsyncMock(return_value=[raw_video(SECOND,'奶蛙上学')])) as check:
            await self.rt.videos.verify(offered[SECOND],self.route)
            check.assert_awaited_once_with(FIRST)


if __name__=='__main__':unittest.main()
