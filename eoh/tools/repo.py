import os
import ast
import git
import os
import ast
import re
import base64
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
import ast
from typing import Dict, Set, Any
def parse_module(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    
    module_info = {
        "functions": {},
        "classes": {},
        "global_vars": {},
        "imports": [],
        "module_level_code": [],
        "decorators": set(),
        "type_aliases": {},
    }
    
    def extract_calls(node: ast.AST) -> Set[str]:
        calls = set()
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Call):
                if isinstance(sub_node.func, ast.Name):
                    calls.add(sub_node.func.id)
                elif isinstance(sub_node.func, ast.Attribute):
                    calls.add(f"{ast.unparse(sub_node.func.value)}.{sub_node.func.attr}")
                else:
                    calls.add(ast.unparse(sub_node.func))
        return calls

    def process_function(node: ast.FunctionDef, parent: str = None) -> Dict[str, Any]:
        func_info = {
            "calls": extract_calls(node),
            "decorators": [ast.unparse(d) for d in node.decorator_list],
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }
        if parent:
            func_info["parent"] = parent
        return func_info

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            module_info["functions"][node.name] = process_function(node)
        
        elif isinstance(node, ast.ClassDef):
            class_info = {"methods": {}, "class_vars": {}}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    class_info["methods"][item.name] = process_function(item, node.name)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            class_info["class_vars"][target.id] = ast.unparse(item.value)
            module_info["classes"][node.name] = class_info
        
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            module_info["global_vars"][node.targets[0].id] = ast.unparse(node.value)
        
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_info["imports"].append(ast.unparse(node))
        
        elif isinstance(node, ast.Expr):
            module_info["module_level_code"].append(ast.unparse(node))
        
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_info["type_aliases"][node.target.id] = ast.unparse(node.annotation)

    return module_info


def assign_importance_score_to_dag(dag: dict) -> dict:
    """ 
    Importance Score Calculation
    - v1. Important node has more sub-nodes
    """
    def calc_subnodes(node_id, visited=None):
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return 0
        
        visited.add(node_id)
        count = 1  # Count the node itself
        
        for edge in dag[node_id]['edges']:
            count += calc_subnodes(edge, visited)
        
        return count

    importance_scores = {}
    for node_id in dag:
        importance_scores[node_id] = calc_subnodes(node_id)

    # Add importance scores to the original DAG
    for node_id in dag:
        dag[node_id]['importance'] = importance_scores[node_id]

    return dag


def build_cross_file_dag(directory, file_name):
    """ 
    Build all dependencies starting of a certain file
    """
    dag = {}
    node_counter = 0
    processed_files = set()

    def get_node_id():
        nonlocal node_counter
        node_counter += 1
        return f'node{node_counter}'

    def process_file(file_path):
        if file_path in processed_files:
            return
        processed_files.add(file_path)

        rel_file_name = os.path.relpath(file_path, directory)
        module_dict = parse_module(file_path)
        
        # Add file node
        file_id = get_node_id()
        dag[file_id] = {
            'name': rel_file_name,
            'type': 'file',
            'file': rel_file_name,
            'file_path': file_path,  # Add full file path
            'edges': set()
        }
        
        # Process top-level functions
        for func_name, func_info in module_dict['functions'].items():
            func_id = get_node_id()
            dag[func_id] = {
                'name': func_name,
                'type': 'function',
                'file': rel_file_name,
                'file_path': file_path,  # Add full file path
                'edges': set()
            }
            dag[file_id]['edges'].add(func_id)
            
            # Add function calls
            for called_func in func_info['calls']:
                add_call_edge(func_id, called_func)
        
        # Process classes
        for class_name, class_info in module_dict['classes'].items():
            class_id = get_node_id()
            dag[class_id] = {
                'name': class_name,
                'type': 'class',
                'file': rel_file_name,
                'file_path': file_path,  # Add full file path
                'edges': set()
            }
            dag[file_id]['edges'].add(class_id)
            
            for method_name, method_info in class_info['methods'].items():
                method_id = get_node_id()
                dag[method_id] = {
                    'name': method_name,
                    'type': 'method',
                    'file': rel_file_name,
                    'file_path': file_path,  # Add full file path
                    'edges': set()
                }
                dag[class_id]['edges'].add(method_id)
                
                # Add method calls
                for called_func in method_info['calls']:
                    add_call_edge(method_id, called_func)

        # Process imports and add connections
        imports = parse_imports(file_path)
        for imp in imports:
            imp_file = find_file(imp, directory)
            if imp_file:
                imp_file_id = process_file(os.path.join(directory, imp_file))
                if imp_file_id:
                    dag[file_id]['edges'].add(imp_file_id)

        return file_id

    def add_call_edge(caller_id, called_func):
        for node_id, node_info in dag.items():
            if node_info['name'] == called_func:
                dag[caller_id]['edges'].add(node_id)
                break

    # Start processing from the given file
    start_file_path = os.path.join(directory, file_name)
    process_file(start_file_path)

    return assign_importance_score_to_dag(dag)


def extract_subgraph_dag(dag, center_node, depth=6, filter_nonclass=False):
    """ 
    Extract SubGraph of any 'node' from DAG into a new DAG dictionary, max_depth is set.
    """
    def get_neighbors(node, current_depth):
        if current_depth > depth:
            return set()
        neighbors = dag[node]['edges']
        for neighbor in list(neighbors):
            neighbors.update(get_neighbors(neighbor, current_depth + 1))
        return neighbors

    def should_include_node(node):
        if not filter_nonclass:
            return True
        node_name = dag[node]['name']
        return node_name[0].isupper() if node_name else False

    subgraph_nodes = {center_node} | get_neighbors(center_node, 1)
    
    subgraph_dag = {}
    
    for node in subgraph_nodes:
        if should_include_node(node):
            subgraph_dag[node] = {
                'name': dag[node]['name'],
                'type': dag[node]['type'],
                'file': dag[node]['file'],
                'edges': set()
            }
            
            for edge in dag[node]['edges']:
                if edge in subgraph_nodes and should_include_node(edge):
                    subgraph_dag[node]['edges'].add(edge)
    
    return subgraph_dag