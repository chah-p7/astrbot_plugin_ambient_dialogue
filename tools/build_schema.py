"""Regenerate the native AstrBot settings schema from Policy defaults."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ambient.policy import Policy

documented = json.loads((ROOT/'_conf_schema.json').read_text(encoding='utf-8'))
fields = {key:{**documented['limits']['items'][key], 'default':value}
          for key,value in asdict(Policy()).items()}
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
        'video_share_format': {'type':'string','description':'视频分享样式','default':'link','options':['link','image','ark'],
            'hint':'link：配文＋链接；image：配文＋链接后单独发封面预览，无需 ARK 权限（需中文字体）；ark：QQ 官方大图卡片，需平台权限。'},
        'video_topics': {'type':'list','description':'视频选材关键词（标题、简介、分区；空列表不限）',
            'default':['搞笑','鬼畜','抽象','整活','游戏','动画','动物']},
        'video_preferred_keywords': {'type':'list','description':'优先考虑的视频关键词','default':[],
            'hint':'明确设置的选材偏好，适合当前话题时优先排序；不会记作真人分享样本。'},
        'video_discovery_seeds': {'type':'list','description':'相似视频发现入口（BV 号，最多 3 个）','default':[],
            'hint':'从这些公开视频发现相关候选，适合没有历史样本时启动；仍须按关键词、时长、日期和现场话题选材。'},
        'video_auto_share_enabled': {'type':'bool','description':'在合适的群聊时机自动搬史','default':False,
            'hint':'启用自然插话后生效。复用最新真人消息与同一次模型请求，明确安排选片配文；QQ 官方仍需有效消息回复窗口。'},
        'video_auto_share_interval_seconds': {'type':'int','description':'自动视频分享最短间隔（秒，0 不额外间隔）','default':1800,
            'hint':'真人新分享和已尝试发送的视频也计入间隔；暂缓后最多每 5 分钟再次安排自动选片。不会限制普通聊天消息次数。'},
        'video_max_seconds': {'type':'int','description':'候选视频最长秒数','default':300},
        'video_refresh_seconds': {'type':'int','description':'公开视频列表刷新间隔（秒）','default':900},
        'video_candidate_minutes': {'type':'int','description':'候选列表有效分钟数','default':60},
        'video_dedupe_days': {'type':'int','description':'本群视频去重天数（1～90）','default':30},
        'video_learning_enabled': {'type':'bool','description':'从本群真人分享学习视频偏好','default':True},
        'video_learning_days': {'type':'int','description':'视频偏好样本保留天数（7～365）','default':90},
        'video_learning_samples': {'type':'int','description':'视频偏好最多样本数（16～512）','default':128},
        'video_cover_selection_enabled': {'type':'bool','description':'后台结合封面判断搬史梗点','default':True},
        'video_selection_provider': {'type':'string','description':'选片视觉模型（留空沿用本群读图模型）','default':''},
        'video_max_age_days': {'type':'int','description':'候选最早发布天数（1～730）','default':365},
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
# Keep documented settings outside Policy (vision, alias learning, and future
# extensions); positional label lists silently drift when Policy grows.
items = documented['groups']['templates']['group']['items']
for key,value in schema['groups']['templates']['group']['items'].items():
    items.setdefault(key,value)
for target in (documented['limits']['items'], items['limits']['items']):
    for key,value in fields.items():
        target[key]['default'] = value['default']
(ROOT/'_conf_schema.json').write_text(json.dumps(documented,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
