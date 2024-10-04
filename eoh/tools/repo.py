import os
import ast
import git
import os
import ast
import re
import base64
from datetime import datetime
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


def merge_special_method(dag: dict):
    """ 
    Merge __init__ method back into Class Object itself
    - Hint: name of method is ClassName::methodname
    Also remove any self-to-self edges
    """
    merged_dag = dag.copy()
    to_remove = []

    for node_id, node in merged_dag.items():
        if node['type'] == 'method' and '::__init__' in node['name']:
            class_name = node['name'].split('::')[0]
            
            # Find the corresponding class node
            class_node_id = next((cid for cid, cnode in merged_dag.items() 
                                  if cnode['type'] == 'class' and cnode['name'] == class_name), None)
            
            if class_node_id:
                # Merge __init__ method into class node
                class_node = merged_dag[class_node_id]
                class_node['init_method'] = node
                
                # Update edges
                class_node['edges'] = class_node.get('edges', set()) | node.get('edges', set())
                
                # Mark __init__ node for removal
                to_remove.append(node_id)
                
                # Update references to __init__ node in other nodes' edges
                for other_node in merged_dag.values():
                    if node_id in other_node.get('edges', set()):
                        other_node['edges'].remove(node_id)
                        other_node['edges'].add(class_node_id)

    # Remove merged __init__ nodes
    for node_id in to_remove:
        del merged_dag[node_id]

    # Remove self-to-self edges
    for node_id, node in merged_dag.items():
        if 'edges' in node:
            node['edges'] = set(edge for edge in node['edges'] if edge != node_id)

    return merged_dag


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
        file_name = rel_file_name.split("/")[-1]
        module_dict = parse_module(file_path)
        
        # Add file node
        file_id = get_node_id()
        dag[file_id] = {
            'name': file_name,
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
                    'name': class_name + "::" + method_name,
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

    dag = merge_special_method(dag)
    dag = assign_importance_score_to_dag(dag)
    return dag 




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
                'importance': dag[node]['importance'],
                'file_path': dag[node]['file_path'],
                'edges': set()
            }
            
            for edge in dag[node]['edges']:
                if edge in subgraph_nodes and should_include_node(edge):
                    subgraph_dag[node]['edges'].add(edge)
    
    return subgraph_dag


# FunPlot of Github Commit history

import os
import ast
from typing import Dict, Any, Set

def commit_tree_to_file_dag(repo: git.Repo, commit: git.Commit, base_path: str = '') -> Dict[str, Dict[str, Any]]:
    file_dag = {}
    node_id = 1

    def parse_imports(file_content: str) -> Set[str]:
        imports = set()
        try:
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except SyntaxError:
            # If there's a syntax error, we'll just skip the imports
            pass
        return imports

    def traverse_tree(tree, current_path=''):
        nonlocal node_id
        for item in tree.traverse():
            full_path = os.path.join(current_path, item.name)
            if item.type == 'blob' and item.name.endswith('.py'):
                # It's a Python file
                file_content = item.data_stream.read().decode('utf-8')
                imports = parse_imports(file_content)
                
                file_dag[f"node{node_id}"] = {
                    'name': item.name,
                    'type': 'file',
                    'file': full_path,
                    'file_path': os.path.join(base_path, full_path),
                    'edges': set(),
                    'imports': imports
                }
                node_id += 1

    traverse_tree(commit.tree)

    # Add edges based on imports
    for node, data in file_dag.items():
        for imp in data['imports']:
            for other_node, other_data in file_dag.items():
                if imp in other_data['file'].replace('/', '.'):
                    data['edges'].add(other_node)

    return file_dag


def obtain_repo_evolution(repo_path):
    # Open the repository
    repo = git.Repo(repo_path)

    # Get all commits
    commits = list(repo.iter_commits('main'))

    # Prepare data
    dates = []
    file_dags = []

    for commit in commits:
        dates.append(datetime.fromtimestamp(commit.committed_date))
        file_dags.append(commit_tree_to_file_dag(repo, commit, repo_path))

    return dates, file_dags