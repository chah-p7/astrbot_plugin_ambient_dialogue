"""Small, source-backed group names; never infer a person's biography."""
import re
import unicodedata

from .learning import plain_text
from .policy import normalized


def alias_cue(text):
    return bool(re.search(r'叫我|外号|绰号|别名|我也叫|就是\s*<@|<@\S+>\s*就是', text))


def name_key(name):
    return unicodedata.normalize('NFKC', normalized(name)).casefold()


def valid_name(name, *, platform=False):
    return bool(2 <= len(name) <= (40 if platform else 16) and
                any(unicodedata.category(c).startswith('L') for c in name) and
                not re.search(r'<@|https?://|[，,。！？!?；;：:="“”\n]', name) and
                not any(w in name for w in ('开玩笑', '假如', '假设', '或者', '玩梗')) and
                name_key(name) not in {'今天', '昨天', '明天', '现在', '大家', '你们', '我们',
                                       '你好', '怎么', '哈哈', '感觉', '这个', '那个'})


def observations(rows, members, *, excluded=()):
    accounts = {m['account']: m for m in members}
    for row in rows:
        if row['self'] or row.get('command') or row['sender'] in excluded:
            continue
        body = plain_text(row['text'])
        own = accounts.get(row['sender'])
        actions = []
        correction = re.fullmatch(r'我的(?:外号|绰号|别名)不是(.+?)[，,]是(.+?)[。!！]*', body)
        denial = re.fullmatch(r'(?:别(?:再)?叫我|不要叫我)(.+?)(?:了)?[。!！]*', body)
        declaration = re.fullmatch(r'(?:我的(?:外号|绰号|别名)是|大家(?:都)?叫我|也可以叫我|我也叫|'
                                   r'(?:以后)?请?叫我|称呼我为)(.+?)[。!！]*', body)
        if own:
            if correction:
                actions = [(own, correction[1], 'rejected'), (own, correction[2], 'self')]
            elif denial:
                actions = [(own, denial[1], 'rejected')]
            elif declaration:
                actions = [(own, declaration[1], 'self')]
        # A quoted name alone, an accusation, or a name merely mentioned in a
        # sentence is not an alias mapping. Keep third-party evidence tentative.
        mapping = re.fullmatch(r'\s*<@!?([A-Za-z0-9_-]+)>\s*(?:的(?:外号|绰号|别名)(?:是|叫)|就是)\s*(.+?)[。!！]*', row['text'])
        reverse = re.fullmatch(r'(.+?)\s*(?:就是|是)\s*<@!?([A-Za-z0-9_-]+)>[。!！]*', row['text'])
        vocative = re.match(r'^\s*<@!?([A-Za-z0-9_-]+)>\s*([^\s，,。！？!?<>]{2,12}?)(?:[，,]\s*你|你(?:在|来|怎么|咋|要))', row['text'])
        target, label = ((mapping[1], mapping[2]) if mapping else
                         (reverse[2], reverse[1]) if reverse else
                         (vocative[1], vocative[2]) if vocative else ('', ''))
        role_or_judgment = name_key(label) in {'群主', '管理员', '机器人', '人类', '老师', '学生',
            '大学生', '高中生', '博士', '土豪', '傻逼', '弱智', '若智', '笨蛋', '好人', '坏人'}
        explicit_alias = mapping and re.search(r'的(?:外号|绰号|别名)', mapping[0])
        if (target in accounts and target not in excluded and target != row['sender']
                and (explicit_alias or not role_or_judgment)):
            actions.append((accounts[target], label, 'candidate'))
        for member, name, kind in actions:
            name = normalized(name)
            if valid_name(name):
                yield {'member': member['id'], 'name': name, 'kind': kind, 'at': row['at'],
                       'source': row['id'], 'witness': row['member']}


def known_names(snapshot, member):
    """Current, explicit and accepted historical names; rejected names win."""
    key = member['id']
    aliases = [a for a in snapshot.get('aliases', []) if a['member'] == key]
    rejected = {a['key'] for a in aliases if a['kind'] == 'rejected'}
    names = [member.get('name', '')]
    names += [r['text'] for r in snapshot.get('facts', []) + snapshot.get('candidates', [])
              if r['member'] == key and r['kind'] == 'address']
    names += [a['name'] for a in aliases if a['kind'] not in {'candidate', 'inferred', 'rejected'}]
    return list(dict.fromkeys(n for n in names if n and name_key(n) not in rejected))


def name_targets(row, snapshot):
    if row.get('directed'):
        return []
    text = name_key(row['text'])
    matches = {m['account'] for m in snapshot.get('members', []) for name in known_names(snapshot, m)
               if re.match(re.escape(name_key(name)) + r'(?:[，,、]\s*|\s+|你|您|在吗|人呢|来不来)', text)}
    return list(matches) if len(matches) == 1 else []


def identity_lookup(row, snapshot):
    """Resolve an explicit name question, without claiming any real biography."""
    lookups = identity_lookups(row, snapshot)
    return lookups[0] if len(lookups) == 1 else None


def identity_lookups(row, snapshot):
    text = re.sub(r'^(?:请问|那|所以|再问一下)\s*', '', plain_text(row['text']))
    labels = re.findall(r'(?:^|[，,；;。\s])([^，,；;。?？]{2,40}?)(?:是谁|是哪位)[呀啊呢吗?？]*', text)
    return [lookup_name(label.strip(), snapshot) for label in dict.fromkeys(labels)
            if valid_name(label.strip(), platform=True)][:5]


def lookup_name(label, snapshot):
    members = [m for m in snapshot.get('members', [])
               if name_key(label) in {name_key(n) for n in known_names(snapshot, m)}]
    if not members:
        inferred = {a['member'] for a in snapshot.get('aliases', [])
                    if a['key'] == name_key(label) and a['kind'] == 'inferred'}
        pending = {a['member'] for a in snapshot.get('aliases', [])
                   if a['key'] == name_key(label) and a['kind'] == 'candidate'}
        matches = [m for m in snapshot.get('members', []) if m['id'] in inferred | pending]
        if matches:
            status = ('probable' if matches[0]['id'] in inferred else 'tentative') if len(matches) == 1 else 'ambiguous'
            return {'name': label, 'status': status,
                    'accounts': [m['account'] for m in matches]}
    return {'name': label, 'status': 'unique' if len(members) == 1 else 'ambiguous' if members else 'unconfirmed',
            'accounts': [m['account'] for m in members]}
