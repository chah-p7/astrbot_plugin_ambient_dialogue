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
    '口吻样本分钟数', '每人口吻样本条数', '口吻样本字数', '记忆注入字数', '节奏统计字数',
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
        'style_excluded_senders': {'type':'list','description':'其他机器人账号（不学口吻）','default':[],
            'hint':'填写本群该平台下已确认的机器人稳定成员 ID；QQ 官方为成员 OpenID，OneBot 为 QQ 号。仍保留其近期消息理解现场，不按昵称猜测。'},
        'quota_mb': {'type':'float','description':'本群额度（MiB，0 继承默认）','default':0},
        'limits': {'type':'object','description':'本群参数','items': fields}}}}},
    'limits': {'type':'object','description':'默认参数','items':fields}}
(ROOT/'_conf_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
