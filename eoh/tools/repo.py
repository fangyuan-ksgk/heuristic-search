import os
import ast
import git
import os
import ast
import re
import base64
import glob
import requests
from typing import Dict, Any, Set, Callable
from datetime import datetime, timedelta
from IPython.display import Image, display
import shutil
from tools.diagram import *

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
                    
    print(f"  Number of objects in the subgraph: {len(subgraph_dag.keys())}")
    return subgraph_dag


# FunPlot of Github Commit history

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

    def traverse_tree(tree):
        nonlocal node_id
        for item in tree.traverse():
            if item.type == 'blob' and item.name.endswith('.py'):
                # It's a Python file
                file_content = item.data_stream.read().decode('utf-8')
                imports = parse_imports(file_content)
                
                file_path = item.path
                if "/" in file_path.replace(".py", ""):
                    module, name = file_path.replace(".py", "").rsplit("/", 1)
                    name = name + ".py"
                else:
                    module = ""
                    name = file_path.replace(".py", "")

                file_dag[f"node{node_id}"] = {
                    'name': name,
                    'type': 'file',
                    'file': file_path.replace("/","."),
                    'file_path': os.path.join(base_path, file_path),
                    'module': module,
                    'edges': set(),
                    'imports': imports
                }
                node_id += 1

    traverse_tree(commit.tree)

    # Add edges based on imports
    for node, data in file_dag.items():
        for imp in data['imports']:
            for other_node, other_data in file_dag.items():
                if imp == other_data['module'] or imp == other_data['file'].replace('/', '.').replace('.py', ''):
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
        file_dag = commit_tree_to_file_dag(repo, commit, repo_path)
        file_dag = assign_importance_score_to_dag(file_dag)
        if len(file_dag) > 0:
            dates.append(datetime.fromtimestamp(commit.committed_date))
            file_dags.append(file_dag)

    return dates, file_dags

def present_repo_info(fastest_repos):
    for i, repo in enumerate(fastest_repos, 1):
        repo_url =repo['html_url']
        print(f"{i}. {repo['full_name']}")
        print(f"   Stars: {repo['stargazers_count']}")
        print(f"   Growth Rate: {repo['growth_rate']:.2f} stars/day")
        print(f"   URL: {repo['html_url']}")
        
        print()
        
def get_fastest_growing_repos(days_ago=7, top_n=10, print=False):
    # GitHub API endpoint
    url = "https://api.github.com/search/repositories"
    
    # Calculate the date for 7 days ago
    date_7_days_ago = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    # Query parameters
    params = {
        "q": f"created:>{date_7_days_ago}",
        "sort": "stars",
        "order": "desc",
        "per_page": 100
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    repos = response.json()["items"]
    
    # Calculate growth rate (stars per day)
    for repo in repos:
        created_at = datetime.strptime(repo["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        days_since_creation = (datetime.now() - created_at).days or 1  # Avoid division by zero
        repo["growth_rate"] = repo["stargazers_count"] / days_since_creation
    
    # Sort by growth rate and get top N
    fastest_growing = sorted(repos, key=lambda x: x["growth_rate"], reverse=True)[:top_n]
    
    if print:
        present_repo_info(fastest_growing)
    return fastest_growing


def build_commit_evolution_gif_of_repo(repo_url: str, temp_repo: str = "temp_repo", output_dir: str = "d2_output", clean: bool = True, n_frames: int = 10):
    """ 
    Still in Bad Shape, to be improved
    """
    # Print some terminal info about the function
    print("Starting build_commit_evolution_gif_of_repo function")
    print(f"Repository URL: {repo_url}")
    print(f"Temporary repository path: {temp_repo}")
    print(f"Output directory: {output_dir}")
    print("This function will clone the repository, analyze its evolution,")
    print("generate dependency diagrams, and create a GIF of the evolution.")
    
    if not os.path.exists(temp_repo):
        clone_repo(repo_url, temp_repo)
    else:
        print("Repo already cloned, skipping cloning...")
    
    _, dags = obtain_repo_evolution(temp_repo)
    
    write_dependency_dags(dags, output_dir=output_dir, n_frames=n_frames) # write interpolated frames
    
    gif_file = f"{temp_repo}_evolution.gif"
    png_files = sorted(glob.glob(f"{output_dir}/*.png"))
    create_gif(png_files, gif_file) # create gif
    
    img = Image.open(png_files[-1])
    
    if clean:
        shutil.rmtree(output_dir)  # delete output_dir
    
    return img


def create_evolution_gif(sub_dag, frame_count, cap_node_number: int = 15, output_dir="d2_output", output_file="evolve_graph.gif", fps=2, static_portion: float = 0.2):
    
    sub_dag = cap_dag_count(sub_dag, cap_node_number=cap_node_number)
    sub_dag = assign_levels(sub_dag)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate opacity frames
    opacity_frames = generate_opacity_frames(sub_dag, frame_count, static_portion=static_portion)

    # Generate and save PNG files
    png_files = []
    for i, frame in enumerate(opacity_frames):
        d2_code = build_d2_from_dag(frame, include_overhead=True)
        png_file = f"evolve_graph_{i}.png"
        save_png_from_d2(d2_code, f"evolve_graph_{i}", output_dir=output_dir)
        png_files.append(os.path.join(output_dir, png_file))

    # Sort PNG files
    png_files = sorted(png_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Create GIF
    create_gif(png_files, output_file=output_file, fps=fps)

    # Get the last frame as an image
    img = Image.open(png_files[-1])

    # Clean up temporary PNG files
    shutil.rmtree(output_dir)

    return img, output_file


def create_gif_from_repo(repo_url: str, 
                         temp_repo: str = "temp_repo", 
                         output_dir: str = "d2_output", 
                         cap_node_number: int = 15, 
                         frame_count: int = 100, 
                         fps: int = 2,
                         static_seconds: int = 4,
                         output_name: str = "evolve_graph"):
    """
    Create a GIF of the repository evolution with level-wise opacity change.

    Args:
        temp_repo (str): The path to the repository to visualize.
        cap_node_number (int): The maximum number of nodes to display.
        frame_count (int): The number of frames in the GIF.
    """
    
    if not os.path.exists(temp_repo):
        clone_repo(repo_url, temp_repo)
    else:
        print("Repo already cloned, skipping cloning...")
    
    dates, file_dags = obtain_repo_evolution(temp_repo)
    dag = file_dags[-1]

    # Get Sub-DAG and assign level values to node 
    static_portion = static_seconds * fps / frame_count 
    img, output_file = create_evolution_gif(dag, frame_count=frame_count, cap_node_number=cap_node_number, output_dir=output_dir, output_file=f"{output_name}.gif", fps=fps, static_portion=static_portion)
    
    return img



def hot_fix_on_method_content(node_content: str) -> str:
    """ 
    Address indentation issue for method content
    """
    lines = node_content.split('\n')
    if len(lines) > 1:
        # Remove any leading whitespace from the first line (method definition)
        first_line = lines[0].lstrip()
        # Remove one level of indentation from the rest of the lines
        rest_lines = [line[4:] if line.startswith('    ') else line for line in lines[1:]]
        node_content = first_line + '\n' + '\n'.join(rest_lines)
    return node_content


def read_node_content(node: dict) -> str:
    """ 
    Parse code-string content from a given node in a DAG object
    Work for node type: file, class, method (standalone and class-related)
    """
    assert node['type'] in ['file', 'class', 'method', 'function'], f"Node type {node['type']} is not supported"

    # Extract information from the node
    file_path = node['file_path']
    node_type = node['type']
    node_name = node['name']


    with open(file_path, 'r') as file:
        content = file.read()

    if node_type == 'file':
        return content

    tree = ast.parse(content)

    for item in ast.walk(tree):
        if isinstance(item, ast.FunctionDef) and node_type == 'function': # Caveat: Only works for standalone functions
            if item.name == node_name:
                node_content = ast.get_source_segment(content, item)
                return node_content


        if isinstance(item, ast.ClassDef) and node_type == 'class':
            if item.name == node_name:
                node_content = ast.get_source_segment(content, item)
                return node_content

        if isinstance(item, ast.ClassDef) and node_type == 'method':
            methods = []
            for sub_item in item.body:
                if isinstance(sub_item, ast.FunctionDef):
                    methods.append(sub_item.name)
                    name_str = f"{item.name}::{sub_item.name}"

                    if name_str == node_name:
                        node_content = ast.get_source_segment(content, sub_item)
                        node_content = hot_fix_on_method_content(node_content)
                        return node_content
                    
    raise ValueError(f"Node content not found for {node_type} '{node_name}' in {file_path}")
    
    
SUMMARY_WITH_CODE_PROMPT = """
Analyze the following python program: 

{code}

Please provide: 
1. A concise and intuitive summary of the code. 
2. A minimal implementation of the code.

Example Output:

Summary: 
[Your Summary]

Minimal Implementation: 
```python 
[Your Minimal Implementation]
```
"""

def analyze_node_content(node_content: str, get_llm_response: Callable) -> str:
    analysis_response = get_llm_response(SUMMARY_WITH_CODE_PROMPT.format(code=node_content))
    return analysis_response.strip()


def sanitize_summary(summary_str: str) -> str:
    for line in summary_str.split("\n"):
        if line.strip():
            return line.strip()
        
def parse_summary_and_minimal_implementation(response: str):
    summary_str = response.split("Minimal Implementation")[0].strip().split("Summary:")[1].strip()
    summary_str = sanitize_summary(summary_str)
    code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    code_str = code_match.group(1).strip() if code_match else ""
    return summary_str, code_str


def get_summary_and_code(node: dict, get_llm_response: Callable):
    response = analyze_node_content(read_node_content(node), get_llm_response)
    summary_str, code_str = parse_summary_and_minimal_implementation(response)
    return summary_str, code_str


def summarize_node(node, sub_dag, get_llm_response):
    node_content = read_node_content(node)
    
    # Get summaries of dependent nodes (nodes with higher level)
    dependent_summaries = []
    for dep_id, dep_node in sub_dag.items():
        if dep_node['level'] > node['level'] and 'summary' in dep_node:
            dependent_summaries.append(f"{dep_node['name']}: {dep_node['summary']}")
    
    dependent_context = "\n".join(dependent_summaries)
    
    full_code = f"""
# Dependent node summaries:
'''
{dependent_context}
'''

# Current node code:
{node_content}
"""
    
    prompt = SUMMARY_WITH_CODE_PROMPT.format(code=full_code)
    
    response = get_llm_response(prompt)
    summary, minimal_code = parse_summary_and_minimal_implementation(response)
    
    return summary, minimal_code

def bottom_up_summarization(sub_dag, get_llm_response):
    # Sort nodes by level in descending order (highest level first)
    sorted_nodes = sorted(sub_dag.keys(), key=lambda x: sub_dag[x]['level'], reverse=True)
    
    # Summarize nodes in bottom-up order
    for node_id in sorted_nodes:
        summary, minimal_code = summarize_node(sub_dag[node_id], sub_dag, get_llm_response)
        sub_dag[node_id]['summary'] = summary
        sub_dag[node_id]['minimal_code'] = minimal_code
    
    return sub_dag