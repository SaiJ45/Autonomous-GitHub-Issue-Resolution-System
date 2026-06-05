import ast


def build_code_graph(files):

    graph = {}

    for file in files:

        path = file["path"]

        try:
            tree = ast.parse(file["content"])
        except:
            continue

        functions = []

        for node in ast.walk(tree):

            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        graph[path] = functions

    return graph