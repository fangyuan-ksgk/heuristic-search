import os
import ast
import git
import networkx as nx
import matplotlib.pyplot as plt
import os
import ast
import networkx as nx
import matplotlib.pyplot as plt
import re
import base64
import requests
from IPython.display import Image, display


def clone_repo(repo_url, target_dir):
    git.Repo.clone_from(repo_url, target_dir)

def parse_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # This is a relative import
                imports.add(node.module if node.module else '')
            elif node.module and '.' in node.module:  # This is likely a local absolute import
                imports.add(node.module)
    
    return imports

def find_file(import_name, directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                module_name = os.path.relpath(file_path, directory).replace('/', '.').replace('\\', '.')[:-3]
                if module_name == import_name or module_name.endswith('.' + import_name):
                    return os.path.relpath(file_path, directory)
    return None



def build_mermaid_graph(start_file, directory):
    graph = []
    visited = set()

    def trace_dependencies(file_path):
        if file_path in visited:
            return
        visited.add(file_path)
        
        imports = parse_imports(os.path.join(directory, file_path))
        for imp in imports:
            dep_file = find_file(imp, directory)
            if dep_file:
                graph.append(f'    "{file_path}" --> "{dep_file}"')
                trace_dependencies(dep_file)

    trace_dependencies(start_file)
    return graph

def visualize_mermaid_graph(graph):
    mermaid_code = "graph TD\n" + "\n".join(graph)
    
    # Display the Mermaid code (for Jupyter notebook)
    from IPython.display import display, Markdown
    display(Markdown(f"```mermaid\n{mermaid_code}\n```"))
    
    # Optionally, save the Mermaid code to a file
    with open("dependency_graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("Mermaid graph saved to 'dependency_graph.mmd'")

def simplify_node_names(graph):
    node_map = {}
    simplified_graph = []
    
    for edge in graph:
        match = re.match(r'    "(.*)" --> "(.*)"', edge)
        if match:
            source, target = match.groups()
            
            if source not in node_map:
                node_map[source] = f"Node{len(node_map) + 1}"
            if target not in node_map:
                node_map[target] = f"Node{len(node_map) + 1}"
            
            simplified_graph.append(f'    {node_map[source]}["{os.path.basename(source)}"] --> {node_map[target]}["{os.path.basename(target)}"]')
    
    return simplified_graph


def visualize_mermaid_graph(graph):
    mermaid_code = "graph TD\n" + "\n".join(graph)
    
    # Encode the Mermaid code
    graphbytes = mermaid_code.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    
    # Display the graph
    display(Image(url="https://mermaid.ink/img/" + base64_string))
    
    # Optionally, save the Mermaid code to a file
    with open("dependency_graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("Mermaid graph saved to 'dependency_graph.mmd'")


def simplify_node_names(graph):
    node_map = {}
    simplified_graph = []
    
    for edge in graph:
        match = re.match(r'    "(.*)" --> "(.*)"', edge)
        if match:
            source, target = match.groups()
            
            if source not in node_map:
                node_map[source] = f"Node{len(node_map) + 1}"
            if target not in node_map:
                node_map[target] = f"Node{len(node_map) + 1}"
            
            simplified_graph.append(f'    {node_map[source]}["{os.path.basename(source)}"] --> {node_map[target]}["{os.path.basename(target)}"]')
    
    return simplified_graph

# Replace the old visualize_graph function with this:
def visualize_graph(graph):
    simplified_graph = simplify_node_names(graph)
    visualize_mermaid_graph(simplified_graph)
    
    
def visualize_mmd_file(file_path):
    with open(file_path, 'r') as f:
        mermaid_code = f.read()
    
    # Encode the Mermaid code
    graphbytes = mermaid_code.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    
    # Generate the visualization URL
    visualization_url = f"https://mermaid.ink/img/{base64_string}"
    
    # Display the graph
    display(Image(url=visualization_url))
    
    print(f"Visualized Mermaid graph from '{file_path}'")

def get_python_files(directory):
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.relpath(os.path.join(root, file), directory))
    return python_files

# Add parser for function parsing within the file (function-level dependency parsing, not just file-level)
def parse_functions(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    
    functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            called_functions = set()
            
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Call) and isinstance(sub_node.func, ast.Name):
                    called_functions.add(sub_node.func.id)
            
            functions[function_name] = called_functions
    
    return functions

def build_function_level_mermaid_graph(start_file, directory):
    graph = []
    visited = set()

    def trace_dependencies(file_path):
        if file_path in visited:
            return
        visited.add(file_path)
        
        imports = parse_imports(os.path.join(directory, file_path))
        functions = parse_functions(os.path.join(directory, file_path))
        
        file_id = f"file_{len(visited)}"
        graph.append(f'    {file_id}["{os.path.basename(file_path)}"]')
        
        for func_name, called_funcs in functions.items():
            func_id = f"{file_id}_{func_name}"
            graph.append(f'    {func_id}["{func_name}"]')
            graph.append(f'    {func_id} --> {file_id}')
            for called_func in called_funcs:
                called_func_id = f"{file_id}_{called_func}"
                graph.append(f'    {func_id} --> {called_func_id}')
        
        for imp in imports:
            dep_file = find_file(imp, directory)
            if dep_file:
                dep_file_id = f"file_{len(visited) + 1}"
                graph.append(f'    {file_id} --> {dep_file_id}')
                trace_dependencies(dep_file)

    trace_dependencies(start_file)
    return graph

def visualize_function_level_graph(graph):
    mermaid_code = "graph TD\n" + "\n".join(graph)
    
    # Encode the Mermaid code
    graphbytes = mermaid_code.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    
    # Generate the image URL
    image_url = f"https://mermaid.ink/img/{base64_string}"
    
    # Display the graph
    display(Image(url=image_url))
    
    # Download and save the image
    response = requests.get(image_url)
    if response.status_code == 200:
        with open("function_level_dependency_graph.png", "wb") as f:
            f.write(response.content)
        print("Function-level dependency graph saved as 'function_level_dependency_graph.png'")
    else:
        print("Failed to download the image")

    # Optionally, save the Mermaid code to a file
    with open("function_level_dependency_graph.mmd", "w") as f:
        f.write(mermaid_code)
    print("Function-level Mermaid graph saved to 'function_level_dependency_graph.mmd'")

def create_function_level_graph(start_file, directory):
    graph = build_function_level_mermaid_graph(start_file, directory)
    visualize_function_level_graph(graph)
