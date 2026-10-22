"""Regenerate the native AstrBot settings schema from Policy defaults."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ambient.policy import Policy

labels = dict(zip(asdict(Policy()), ['每群存储额度（MiB）', '开始压缩占比（%）', '压缩目标占比（%）',
    '滚动摘要字数', '每人偏好条数', '单条记忆字数', '记忆整理间隔（秒）', '每批记忆条数',
    '候选队列条数', '普通成员过期天数', '普通话题过期天数', '原始聊天保存小时（0 仅内存）',
    '原始聊天最大条数', '近期消息条数', '近期消息分钟数', '近期消息字数', '口吻样本条数',
    '口吻样本分钟数', '每人口吻样本条数', '口吻样本字数', '每个样本的邻近前话条数（0 关闭）',
    '样本邻近前话最大间隔（秒）', '记忆注入字数', '节奏统计字数',
    '总上下文字数', '插话判断间隔（秒）', '插话等待安静（秒）', '插话冷却（秒，可为 0）',
    '插话消息新鲜期（秒）', '插话模型超时（秒）', '插话最大字数', '普通回复消息新鲜期（秒）',
    '回复修正超时（秒）', '分条目标字数（0 关闭，代码和链接保持完整）', '分条发送间隔（毫秒）']))
fields = {key:{'type': 'float' if isinstance(value,float) or key=='quota_mb' else 'int',
    'description': labels[key], 'default': value} for key,value in asdict(Policy()).items()}
schema = {'groups': {'type':'template_list', 'description':'启用的群', 'default':[],
    'hint':'按平台、机器人账号和群标识隔离。采集验证关闭后才注入上下文及接管记忆。',
    'templates': {'group': {'name':'群', 'display_item':'umo', 'items': {
        'umo': {'type':'string','description':'群会话标识','default':''},
        'account': {'type':'string','description':'机器人稳定账号（QQ 官方为 AppID）','default':''},
        'enabled': {'type':'bool','description':'启用','default':True},
        'capture_only': {'type':'bool','description':'仅采集验证','default':True},
        'interject_enabled': {'type':'bool','description':'自然插话','default':False},
        'video_sharing_enabled': {'type':'bool','description':'分享 B 站视频到本群','default':False,
            'hint':'从公开视频列表选材，由当前回复或插话决定是否分享短配文＋原链接；不登录 B 站、不在评论区发消息。'},
        'video_card_enabled': {'type':'bool','description':'QQ 官方视频大图卡片（配文分开发送）','default':False,
            'hint':'需 QQ 平台 ARK 权限及 B 站链接、封面域名许可。仅 QQ 官方适配器生效；失败或回执不明不自动补发。'},
        'video_topics': {'type':'list','description':'视频选材关键词（标题、简介、分区；空列表不限）',
            'default':['搞笑','鬼畜','抽象','整活','游戏','动画','动物']},
        'video_max_seconds': {'type':'int','description':'候选视频最长秒数','default':300},
        'video_refresh_seconds': {'type':'int','description':'公开视频列表刷新间隔（秒）','default':900},
        'video_candidate_minutes': {'type':'int','description':'候选列表有效分钟数','default':60},
        'video_dedupe_days': {'type':'int','description':'本群视频去重天数（1～90）','default':30},
        'stickers_enabled': {'type':'bool','description':'群表情采集与发送','default':False,
            'hint':'只采集本群实际收到的图片，按内容去重和频次入库；默认关闭。管理入口见插件 Pages 的 stickers。'},
        'stickers_top': {'type':'int','description':'本群可用表情数量','default':100},
        'stickers_min_occurrences': {'type':'int','description':'图片至少出现几次才审核入库','default':2},
        'stickers_quota_mb': {'type':'int','description':'本群表情文件额度（MiB，16～512）','default':64},
        'stickers_caption_provider': {'type':'string','description':'表情图片理解模型（留空沿用本群图片转述模型）','default':''},
        'style_excluded_senders': {'type':'list','description':'其他机器人账号（不学口吻）','default':[],
            'hint':'填写本群该平台下已确认的机器人稳定成员 ID；QQ 官方为成员 OpenID，OneBot 为 QQ 号。仍保留其近期消息理解现场，不按昵称猜测。'},
        'quota_mb': {'type':'float','description':'本群额度（MiB，0 继承默认）','default':0},
        'limits': {'type':'object','description':'本群参数','items': fields}}}}},
    'limits': {'type':'object','description':'默认参数','items':fields}}
(ROOT/'_conf_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
