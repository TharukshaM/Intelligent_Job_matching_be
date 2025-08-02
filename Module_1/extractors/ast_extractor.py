import javalang
import esprima

def extract_ast_nodes(code: str, language: str) -> list:
    nodes = []

    if language.lower() == 'java':
        try:
            tree = javalang.parse.parse(code)
            for path, node in tree:
                nodes.append(type(node).__name__)
        except:
            pass

    elif language.lower() == 'javascript':
        try:
            tree = esprima.parseScript(code, tolerant=True)
            def walk(n):
                if isinstance(n, list):
                    for i in n: walk(i)
                elif hasattr(n, 'type'):
                    nodes.append(n.type)
                    for attr in dir(n):
                        if not attr.startswith('_'):
                            walk(getattr(n, attr))
            walk(tree.body)
        except:
            pass

    return nodes
