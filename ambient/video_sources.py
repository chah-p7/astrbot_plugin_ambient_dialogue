"""Bounded, public Bilibili reads and links actually present in received messages."""
import html
from itertools import islice
import json
import re
from urllib.parse import parse_qs, unquote, urlparse
from xml.etree import ElementTree

BVID = re.compile(r'(?<![A-Za-z0-9])BV[0-9A-Za-z]{10}(?![A-Za-z0-9])')
COVER = re.compile(r'https?://(i[0-9]+\.hdslb\.com)(/bfs/[A-Za-z0-9/_-]+\.(?:jpg|jpeg|png|webp))', re.I)
SHORT = re.compile(r'https?://b23\.tv/([A-Za-z0-9]{5,20})(?![A-Za-z0-9/])')
AVID = re.compile(r'av[1-9][0-9]{0,18}')
CARD_FIELDS = ('ark', 'miniapp', 'mini_app', 'json', 'xml', 'card')


def raw_payload(raw):
    data = raw if isinstance(raw, dict) else getattr(raw, 'raw_data', None)
    if isinstance(data,dict) and isinstance(data.get('raw_data'),dict):
        data = data['raw_data']
    return data if isinstance(data, dict) else {}


def card_carriers(raw):
    """Only the current message's carriers; QQ type 103 starts with a quote."""
    data = raw_payload(raw)
    result = [(key, data[key]) for key in CARD_FIELDS if data.get(key)]
    elements = data.get('msg_elements', getattr(raw, 'msg_elements', None))
    if isinstance(elements, list):
        if str(data.get('message_type', getattr(raw, 'message_type', ''))) == '103':
            elements = elements[1:]
        if elements:
            result.append(('msg_elements', elements[:32]))
    return result


def _decoded(value):
    value = html.unescape(value.replace('\\/', '/'))
    for _ in range(2):
        decoded = unquote(value)
        if decoded == value:
            break
        value = decoded
    return value


def _label(value):
    return ' '.join(re.sub(r'<[^>]*>', '', html.unescape(str(value))).split())[:100]


def references(text, parts=(), raw=None):
    """Bounded card decoding, without traversing quoted/forwarded identities."""
    found = {}
    def collect(value, title='', *, bare=False, bili=False):
        value = _decoded(value)
        bracket = re.search(r'【([^\n]{1,120}?)】', value)
        title = _label(bracket[1].removesuffix('-哔哩哔哩')) if bracket else _label(title)
        refs = BVID.findall(value) if bare else []
        for url in re.findall(r'https?://[^\s<>"\\]+', value):
            try:
                parsed = urlparse(url.rstrip('。，,;；)）]'))
                if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
                    continue
                if parsed.hostname == 'b23.tv':
                    if re.fullmatch(r'/[A-Za-z0-9]{5,20}', parsed.path):
                        refs.append('https://b23.tv'+parsed.path)
                elif parsed.hostname in {'bilibili.com', 'www.bilibili.com', 'm.bilibili.com'}:
                    match = re.fullmatch(r'/video/(BV[0-9A-Za-z]{10}|av[1-9][0-9]{0,18})/?', parsed.path)
                    if match:
                        refs.append(match[1])
            except ValueError:
                continue
        # Native miniapp paths can contain only an avid, without a web URL.
        if bili:
            try:
                query = parse_qs(urlparse(value).query)
            except ValueError:
                query = {}
            for key in ('bvid', 'avid', 'aid'):
                for item in query.get(key, [])[:1]:
                    ref = item if key == 'bvid' else 'av'+item
                    if BVID.fullmatch(ref) or AVID.fullmatch(ref):
                        refs.append(ref)
        for ref in refs:
            if ref not in found and len(found) < 3:
                found[ref] = {'ref': ref, 'title': title}
            elif ref in found and not found[ref]['title'] and title:
                found[ref]['title'] = title

    collect(str(text)[:16000], bare=True)
    sources = [value for _, value in card_carriers(raw)]
    data = raw_payload(raw)
    if isinstance(data.get('content'), str) and data['content'] != text:
        collect(data['content'][:16000], bare=True)
    for content in (str(text), data.get('content')):
        if isinstance(content,str) and content.lstrip().startswith(('{','<')):
            sources.append(content)
    for part in parts:
        kind = (type(part).__name__ if not isinstance(part, dict) else str(part.get('type', ''))).lower()
        if kind not in {'json', 'xml', 'ark', 'miniapp'}:
            continue
        value = part.get('data', '') if isinstance(part, dict) else getattr(part, 'data', '')
        sources.append(value)
    stack = [(value, '', False, 0) for value in sources[:36]]
    nodes, remaining = 0, 65536
    ignored = {'author','sender','mentions','message_reference','quote','quoted','reply','forward','nodes',
               'attachments','avatar','icon','preview','cover','id','message_id','appid','uin','openid'}
    while stack and nodes < 256 and remaining > 0:
        value, title, bili, depth = stack.pop()
        nodes += 1
        if depth > 10:
            continue
        if isinstance(value, dict):
            if str(value.get('type', '')).lower() in {'reply','quote','forward','node'}:
                continue
            # QQ ARK uses key/value entries, while miniapps nest their detail.
            kv = value.get('kv')
            if isinstance(kv, list):
                values = {r['key']:r.get('value') for r in kv[:32] if isinstance(r,dict) and isinstance(r.get('key'),str)}
                title = values.get('#METATITLE#') or title
            title = next((value[k] for k in ('desc','title','prompt') if isinstance(value.get(k),str) and value[k].strip()), title)
            bili = bili or any('bilibili' in str(value.get(k,'')).lower() or '哔哩哔哩' in str(value.get(k,''))
                               for k in ('app','title','desc','name'))
            for key, child in islice(value.items(),64):
                if str(key).lower() not in ignored:
                    stack.append((child, title, bili, depth+1))
        elif isinstance(value, list):
            stack.extend((child,title,bili,depth+1) for child in value[:32])
        elif isinstance(value, str):
            if len(value) > remaining:
                continue
            remaining -= len(value)
            decoded = _decoded(value).strip()
            serialized = value.strip() if value.strip().startswith(('{','[','"','<')) else decoded
            if serialized.startswith(('{','[','"')):
                try:
                    child = json.loads(serialized)
                    stack.append((child,title,bili,depth+1))
                    continue
                except (ValueError,RecursionError):
                    pass
            if serialized.startswith('<') and '<!DOCTYPE' not in serialized.upper():
                try:
                    root = ElementTree.fromstring(serialized)
                    title = next((r.text for r in root.iter('title') if r.text),title)
                    for element in list(root.iter())[:128]:
                        for val in [element.text or '', *element.attrib.values()]:
                            collect(val,title,bili=bili)
                    continue
                except (ElementTree.ParseError,ValueError):
                    pass
            collect(decoded,title,bili=bili)
    return list(found.values())


async def public_json(path, params):
    import aiohttp
    if path not in {'popular', 'view', 'archive/related'}:
        raise ValueError('video_source_unavailable')
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12), trust_env=False,
            headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bilibili.com/'}) as session:
        async with session.get('https://api.bilibili.com/x/web-interface/'+path,
                               params=params, allow_redirects=False) as response:
            if response.status != 200 or (response.content_length or 0) > 2*1024*1024:
                raise ValueError('video_source_unavailable')
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > 2*1024*1024:
                    raise ValueError('video_response_too_large')
            value = json.loads(data.decode('utf-8'))
            if not isinstance(value, dict) or value.get('code') != 0:
                raise ValueError('video_source_unavailable')
            return value.get('data')


async def public_feed():
    return await public_json('popular', {'ps': 50, 'pn': 1})


async def related_feed(identity):
    if not BVID.fullmatch(identity):
        return []
    data = await public_json('archive/related', {'bvid': identity})
    return data[:30] if isinstance(data, list) else []


async def resolve_reference(ref):
    if BVID.fullmatch(ref):
        return ref
    if AVID.fullmatch(ref):
        data = await public_json('view', {'aid':ref[2:]})
        identity = data.get('bvid','') if isinstance(data,dict) else ''
        if BVID.fullmatch(identity):
            return identity
        raise ValueError('video_reference_unavailable')
    if not SHORT.fullmatch(ref):
        raise ValueError('video_reference_invalid')
    import aiohttp
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8), trust_env=False) as session:
        async with session.get(ref, allow_redirects=False) as response:
            dest = urlparse(response.headers.get('Location', ''))
            if (response.status not in {301, 302, 303, 307, 308} or dest.scheme != 'https'
                    or dest.hostname not in {'www.bilibili.com', 'm.bilibili.com'}
                    or dest.username or dest.password or dest.port not in {None, 443}):
                raise ValueError('video_reference_unavailable')
            match = re.fullmatch(r'/video/(BV[0-9A-Za-z]{10})/?', dest.path)
            if not match:
                raise ValueError('video_reference_unavailable')
            return match[1]
