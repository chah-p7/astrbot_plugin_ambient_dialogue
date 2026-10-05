"""Request-local AstrBot bridges; never patch global classes or contexts."""
from copy import copy


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
