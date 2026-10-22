"""On-demand retrieval from the existing group window, without another archive."""
from collections import Counter
import math
import re

from .learning import MEDIA_ONLY, mentions, plain_text, quote_source, words
from .policy import noise
from .aliases import known_names


REQUEST = re.compile(r'记不记得|还记得|记得.{0,40}[吗?？]|认得我|我是谁|'
                     r'(?:刚才|之前|前面|上次|先前).{0,24}(?:说|提|聊|讲|问)|'
                     r'谁(?:说|提到|讲)|是谁[啊呀呢?？。]*$')
HISTORY = r'(?:刚才|之前|前面|上次|先前|曾经)'
QUERY_FILLER = re.compile(r'还记得|记不记得|记得|认得|到底|究竟|能不能|帮我|请问|'
                          + HISTORY + r'|(?:说|提到|聊|讲|问)(?:过|的|了)?|'
                          r'是什么|用什么|说什么|是谁|什么|哪些|怎么|啥|一下|那个|这个|'
                          r'你们|我们')
QUOTED = re.compile(r'[“「‘"]([^”」’"]+)[”」’"]')


def recall_rows(rows, current, snapshot, *, excluded=()):
    """Keep exact authors/times; matches are speech evidence, never identity facts."""
    lookup = {'requested': False, 'scope': 'retained_group_messages', 'complete_history': False,
              'matched_messages': 0}
    if not current or not REQUEST.search(plain_text(current['text'])):
        return [], lookup
    lookup['requested'] = True
    text = plain_text(current['text'])
    unquoted = QUOTED.sub(' ', current['text'])
    def author_query(label):
        return re.search(label + r'\s*(?:' + HISTORY + r')?\s*(?:说|提到|聊|讲|问)|'
                         + HISTORY + r'\s*' + label + r'\s*(?:说|提到|聊|讲|问)', unquoted)
    scope = None
    if author_query('我') or re.search(r'记得我|认得我|我是谁', text):
        scope = {current['sender']}
    elif author_query('你'):
        scope = {'bot'}
    elif re.search(HISTORY + r'|谁(?:说|提到|讲)', text):
        targets = {a for a in mentions(current) if a != 'bot' and author_query(r'<@!?' + re.escape(a) + '>')}
        if targets:
            scope = targets
        else:
            # A name can narrow an author search only if it maps unambiguously.
            named = [(m, name) for m in snapshot.get('members', []) for name in known_names(snapshot, m)
                     if len(name) >= 2 and author_query(re.escape(name))]
            if len({m['account'] for m, _ in named}) == 1:
                scope = {named[0][0]['account']}
                for _, name in named:
                    text = text.replace(name, '')
    query = QUERY_FILLER.sub('', text)
    query = re.sub(r'^[你我他她谁]+|[的吗呢啊呀了?？。]+$', '', query).strip()
    quoted = QUOTED.findall(text)
    if quoted:
        query = ' '.join(quoted)
    terms = words(query)
    before = rows[:next((i for i, r in enumerate(rows) if r['id'] == current['id']), len(rows))]
    available = []
    for row in before:
        body = plain_text(row['text'])
        if (row['at'] > current['at'] or row.get('command') or noise(body)
                or MEDIA_ONLY.fullmatch(body) or row['sender'] in excluded):
            continue
        author = 'bot' if row['self'] else row['sender']
        if scope is not None and author not in scope:
            continue
        if row['self'] and (scope != {'bot'} or (row.get('quote') or {}).get('sender') != current['sender']):
            continue
        # Reply payloads remain nested evidence, not statements by the quoter.
        available.append((row, words(body)))
    frequencies = Counter(t for _, tokens in available for t in tokens)
    ranked = []
    for row, tokens in available:
        shared = terms & tokens
        if terms:
            if len(shared) < min(2, len(terms)):
                continue
            score = sum(math.log(1 + len(available)/frequencies[t]) for t in shared)
        elif scope is None:
            continue
        else:
            score = 0
        ranked.append((score, row['at'], row))
    ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    result, seen = [], set()
    for _, _, row in ranked[:2]:
        if row['id'] not in seen:
            seen.add(row['id'])
            result.append((row, 'requested_history_match'))
        source, _ = quote_source(row, before)
        if source and source['id'] not in seen and not source.get('command'):
            seen.add(source['id'])
            result.append((source, 'history_quote_source'))
    return result[:4], lookup
