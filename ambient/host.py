"""Request-local AstrBot bridges; never patch global classes or contexts."""
from copy import copy
import re

from .context import compact, model_text


def canonical_parts(parts, pack, row):
    """Align native identity/quote decorations without discarding attachments."""
    current = pack['current_message']
    accounts = dict(zip(row.get('mentions', []), current.get('mentions', [])))
    accounts[row['sender']] = current['speaker']
    result = []
    for part in parts:
        text = getattr(part, 'text', '')
        if text.startswith('<system_reminder>') and text.endswith('</system_reminder>'):
            replacement = 'Current speaker: '+current['speaker']+', Nickname: '+current.get('name', '')
            text = re.sub(r'User ID: [^\n<]*', lambda _: replacement, text)
        elif text.startswith('<Quoted Message>\n') and text.endswith('\n</Quoted Message>'):
            quote = current.get('quote', {})
            # The canonical quote already has a verified source when resolvable.
            # Keep native image captions; unresolved nested text stays quotation.
            body = text[len('<Quoted Message>\n'):-len('\n</Quoted Message>')]
            caption = body.partition('[Image Caption in quoted message]:')[2]
            body = caption if quote.get('link') in {'native_quote', 'quoted_text_exact'} else body
            if not body.strip():
                continue
            text = compact({'quoted_attachment': {'ref': quote.get('ref', 'outside_window'),
                           'speaker': quote.get('speaker', 'unknown'),
                           'text': model_text(body, accounts)}})
        if text != getattr(part, 'text', ''):
            part = copy(part)
            part.text = text
        result.append(part)
    return result


def route_send_tool(request, event):
    tools = request.func_tool
    original = tools.get_tool('send_message_to_user') if hasattr(tools, 'get_tool') else None
    if original is None or getattr(original, '_ambient_event', None) is event:
        return
    tools, tool = copy(tools), copy(original)
    tools.tools = list(tools.tools)

    async def call(context, **kwargs):
        wrapper, agent = copy(context), copy(context.context)
        host = agent.context
        outcomes = []

        class ScopedContext:
            def __getattr__(self, name):
                return getattr(host, name)

            async def send_message(self, session, message):
                if str(session) == event.unified_msg_origin:
                    receipt = await event.send(message)
                    outcomes.append(bool(isinstance(receipt, dict) and receipt.get('id')))
                    return bool(outcomes[-1])
                return await host.send_message(session, message)

        wrapper.context, agent.context = agent, ScopedContext()
        result = await original.call(wrapper, **kwargs)
        if outcomes and not all(outcomes):
            return 'Delivery was not confirmed. Do not resend or claim successful delivery.'
        return result

    tool.call = call
    tool._ambient_event = event
    tools.add_tool(tool)
    request.func_tool = tools
