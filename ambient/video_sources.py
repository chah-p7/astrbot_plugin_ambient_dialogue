"""Bounded, public Bilibili reads and links actually present in received messages."""
import html
import json
import re
from urllib.parse import urlparse

BVID = re.compile(r'(?<![A-Za-z0-9])BV[0-9A-Za-z]{10}(?![A-Za-z0-9])')
COVER = re.compile(r'https?://(i[0-9]+\.hdslb\.com)(/bfs/[A-Za-z0-9/_-]+\.(?:jpg|jpeg|png|webp))', re.I)
SHORT = re.compile(r'https?://b23\.tv/([A-Za-z0-9]{5,20})(?![A-Za-z0-9/])')


def references(text, parts=()):
    # Only actual text/card payloads; never quote IDs, image URLs or sender IDs.
    sources = [str(text)[:16000]]
    for part in parts:
        kind = type(part).__name__ if not isinstance(part, dict) else str(part.get('type', '')).lower()
        if kind not in {'Json', 'XML', 'json', 'xml'}:
            continue
        value = part.get('data', '') if isinstance(part, dict) else getattr(part, 'data', '')
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        sources.append(str(value)[:16000].replace('\\/', '/'))
    found = {}
    for source in sources:
        title = re.search(r'【([^\n]{1,120}?)(?:-哔哩哔哩)?】', source)
        label = title[1].removesuffix('-哔哩哔哩') if title else ''
        if not label:
            match = re.search(r'"(?:desc|title)"\s*:\s*"([^"\n]{1,120})"', source)
            label = match[1] if match else ''
        for ref in [*BVID.findall(source), *("https://b23.tv/"+m[1] for m in SHORT.finditer(source))]:
            found.setdefault(ref, {'ref': ref, 'title': html.unescape(label)[:100]})
    return list(found.values())[:3]


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
