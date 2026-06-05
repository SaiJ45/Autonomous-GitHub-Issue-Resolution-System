import re

with open('langgraph_flow/nodes.py', 'r', encoding='utf-8') as f:
    content = f.read()

def replacer(match):
    indent = match.group(1)
    return match.group(0) + f'\n{indent}    state["exception_type"] = type(e).__name__\n{indent}    state["exception_msg"] = str(e)'

new_content = re.sub(r'^(\s*)except Exception as e:', replacer, content, flags=re.MULTILINE)

with open('langgraph_flow/nodes.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
