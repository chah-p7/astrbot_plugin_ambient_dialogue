"""Readable text parts, sent in order and recorded only after native receipts."""
from __future__ import annotations

import asyncio
import re

# Keep code, links, quoted phrases and formulas intact. The length target is
# soft: an indivisible token may exceed it rather than become unusable.
ATOMIC = re.compile(
    r'```[^\n]*\n[\s\S]*?(?:```|$)|~~~[^\n]*\n[\s\S]*?(?:~~~|$)'
    r'|`+[^`\n]*`+|\[[^\]\n]*\]\([^\s]*\)'
    r'|https?://[^\s<>，。！？；“”「」]+'
    r'|\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)'
    r'|\$[^$\n]+\$|“[^”\n]*”|「[^」\n]*」|『[^』\n]*』'
    r'|[A-Za-z0-9_]+(?:[./:@+\-][A-Za-z0-9_]+)*'
)


def split_reply(text, target=80):
    text = str(text).strip().replace('\r\n', '\n')
    if not text:
        return []
    if target <= 0:
        return [text]
    # Positions inside protected spans cannot be used as cut points.
    protected = set()
    for match in ATOMIC.finditer(text):
        protected.update(range(match.start()+1, match.end()))
    paragraph, sentence, clause, fallback = [], [], [], []
    for index, char in enumerate(text, 1):
        if index in protected:
            continue
        following = text[index:index+1]
        if following and following in '”’」』】）)]!?！？。；;':
            continue
        if char == '\n' and (index == len(text) or following == '\n'):
            paragraph.append(index)
        if char in '。！？!?；;\n' or (char == '.' and (not following or following.isspace())):
            sentence.append(index)
        if char in '，,、：:' or char.isspace():
            clause.append(index)
        if '\u4e00' <= char <= '\u9fff' and (not following or '\u4e00' <= following <= '\u9fff'):
            fallback.append(index)
    paragraph.append(len(text))
    safe = sorted(set(sentence + clause + fallback + paragraph))
    parts, start = [], 0
    while start < len(text):
        end = next(pos for pos in paragraph if pos > start)
        if end-start > target:
            limit = start+target
            # Prefer complete sentences, then clauses/spaces. Do not cut a
            # sentence into isolated characters unless no punctuation exists.
            end = next((max(points) for positions in (sentence, clause, fallback)
                        if (points := [p for p in positions if start < p <= limit])),
                       next(p for p in safe if p > limit))
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return parts


async def send_reply_parts(transport, text, policy, before_send, confirmed, *, max_parts=None):
    """Caller owns the group lock and a durable claim for the entire draft.

    Every part repeats the caller's freshness/config checks at the HTTP boundary.
    Any uncertain part stops the sequence; no retry or automatic resumption.
    """
    parts = split_reply(text, policy.reply_segment_chars)
    if max_parts is not None:
        if max_parts < 1:
            raise ValueError('qq_reply_slots_exhausted')
        if len(parts) > max_parts:
            # Preserve the rest of the answer when QQ's native per-anchor
            # allowance is smaller than the number of natural paragraphs.
            parts = parts[:max_parts-1] + ['\n\n'.join(parts[max_parts-1:])]
    receipts = []
    for index, part in enumerate(parts):
        if index and policy.reply_segment_interval_ms:
            await asyncio.sleep(policy.reply_segment_interval_ms / 1000)
        receipt = await asyncio.wait_for(transport.send(part, before_send), 20)
        await confirmed(part, receipt)
        receipts.append(receipt)
    return receipts
