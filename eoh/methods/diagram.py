import os
import subprocess
import numpy as np
from PIL import Image
import time
import copy
import base64
import io
from tqdm import tqdm
from .meta_prompt import parse_plan_graph

d2_prefix = """vars: {
  d2-config: {
    sketch: true
  }
}
classes: {
  code: {
    label: ""
    shape: diamond
    style: {
      fill: yellow
      shadow: true
    }
  }
}

classes: {
  llm: {
    label: ""
    shape: hexagon
    style: {
      fill: lightblue
      shadow: true
    }
  }
}
"""

object_template_with_overhead = """{object_name}.class: {object_type}
{object_name}.label: "{object_name}"
{object_name}: {{
  style: {{
    opacity: {opacity}
    stroke: "black"
    stroke-width: 4
    shadow: true
  }}
}}"""

object_template = """{object_name}.class: {object_type}
{object_name}.label: "{object_name}"
{object_name}: {{
  style: {{
    opacity: {opacity}
    stroke: "black"
    stroke-width: 4
    shadow: true
  }}
}}"""

code_template = """{object_name}_code: |python
{code_str}
|
{object_name}_code: {{
  style: {{
    opacity: {opacity}
  }}
}}"""

get_object_name = lambda node: node['name'].replace(".","_")


def build_d2_code_node(node: dict) -> str:
    object_name = get_object_name(node)
    code_str = node["code_str"]
    opacity = node["opacity"]
    opacity = min(1.0, max(0.0, opacity))
    opacity_str = f"{opacity:.2f}"
    return code_template.format(object_name=object_name, code_str=code_str, opacity=opacity_str)

def build_d2_node(node: dict) -> str:
   
    object_name = get_object_name(node)
    object_type = node["type"]
    assert object_type in ["code", "llm"]
    opacity = node["opacity"]
    opacity = min(1.0, max(0.0, opacity))
    opacity_str = f"{opacity:.2f}"
    
    return object_template.format(object_name=object_name, object_type=object_type, opacity=opacity_str)


link_template = """{start_object_name} -> {end_object_name}: {{
  style.stroke: black
  style.opacity: {opacity}
  style.stroke-width: 2
}}"""


link_code_template = """{start_object_name} -> {start_object_name}_code: {{
  style.stroke: black
  style.opacity: {opacity}
  style.stroke-width: 2
}}"""


def build_d2_code_edge(str_node: dict) -> str:
    opacity = min(1.0, max(0.0, str_node["opacity"]))
    opacity_str = f"{opacity:.2f}"
    start_object_name = get_object_name(str_node)
    return link_code_template.format(start_object_name=start_object_name,  opacity=opacity_str)


def build_d2_edge(str_node: dict, end_node: dict) -> str:
    opacity = min(1.0, max(0.0, end_node["opacity"]))
    opacity_str = f"{opacity:.2f}"
    
    start_object_name = get_object_name(str_node)
    end_object_name = get_object_name(end_node)
    
    return link_template.format(start_object_name=start_object_name, end_object_name=end_object_name, opacity=opacity_str)


def build_d2_from_dag(dag: dict) -> str:
    """
    Convert Sub-DAG dictionary into d2 code
    """
    d2_code = d2_prefix 

    for node_id, node in dag.items():
        object_str = build_d2_node(node)
        d2_code += "\n" + object_str
        if node["code_str"]:
            code_str = build_d2_code_node(node)
            d2_code += "\n" + code_str
            link_str = build_d2_code_edge(node)
            if link_str:
                d2_code += f"\n{link_str}"

    for node_id, node in dag.items():
        edge_pairs = [(node_id, end_node) for end_node in node['edges']]    
        for start, end in edge_pairs:
            link_str = build_d2_edge(dag[start], dag[end])
            if link_str:
                d2_code += f"\n{link_str}"
            
    return d2_code


def visualize_plan_dict(plan_dict: dict):
    parsed_dag = parse_plan_graph(plan_dict)
    plot_plan_graph(parsed_dag)


def plot_plan_graph(dag, output_dir="d2_output", show=True, name="plan_graph"):
    d2_code = build_d2_from_dag(dag)
    png_file_path = save_png_from_d2(d2_code, name, output_dir=output_dir)
    
    if png_file_path and show:
        visualize_dag(dag, output_dir=output_dir, show=show, name=name)
    
    return png_file_path


def save_png_from_d2(d2_code, file_name, output_dir="d2_output"):
    """
    Save the d2_code as a .d2 file and generate the corresponding .svg file.
    
    Args:
    d2_code (str): The D2 diagram code.
    file_name (str): The base name for the output files (without extension).
    output_dir (str): The directory to save the files in.
    
    Returns:
    str: The path to the saved PNG file, or None if an error occurred.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the .d2 file
    d2_file_path = os.path.join(output_dir, f"{file_name}.d2")
    with open(d2_file_path, "w") as f:
        f.write(d2_code)
    
    # Generate the PNG file using the d2 command-line tool
    png_file_path = os.path.join(output_dir, f"{file_name}.png")
    try:
        subprocess.run(["d2", d2_file_path, png_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating PNG: {e}")
        png_file_path = None
    except FileNotFoundError:
        print("Error: d2 command not found. Make sure d2 is installed and in your PATH.")
        png_file_path = None
    
    os.remove(d2_file_path)
    
    return png_file_path


def visualize_dag(dag: dict, output_dir="sandbox", show: bool = True, cap_node_number = 50, name: str = ""):
    """
    Visualize the DAG using d2
    """
    if 'opacity' not in dag[list(dag.keys())[0]]:
        dag = decide_opacity_of_dag(dag, progress=1.0, cap_node_number=cap_node_number)
        
    d2_code = build_d2_from_dag(dag)
    png_file_path = save_png_from_d2(d2_code, f"{name}_dag", output_dir=output_dir)
    if png_file_path:
        if show:    
            dag_graph = Image.open(png_file_path)
            dag_graph.show()
    else:
        print("Error: PNG file could not be generated.")
    
    return png_file_path

def visualize_plan_graph(dag: dict, output_dir="sandbox", show: bool = True, name: str = ""):
    """ 
    Visualize the plan graph using d2
    """
    raise NotImplementedError
    
    
    
    
def save_svg_from_d2(d2_code, file_name, output_dir="d2_output"):
    """
    Save the d2_code as a .d2 file and generate the corresponding .svg file.
    
    Args:
    d2_code (str): The D2 diagram code.
    file_name (str): The base name for the output files (without extension).
    output_dir (str): The directory to save the files in.
    
    Returns:
    tuple: Paths to the saved .d2 and .svg files.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the .d2 file
    d2_file_path = os.path.join(output_dir, f"{file_name}.d2")
    with open(d2_file_path, "w") as f:
        f.write(d2_code)
    
    # Generate the .svg file using the d2 command-line tool
    svg_file_path = os.path.join(output_dir, f"{file_name}.svg")
    try:
      # Redirect stdout and stderr to devnull
        subprocess.run(["d2", d2_file_path, svg_file_path], check=True)
    except subprocess.CalledProcessError as e:
        svg_file_path = None
    except FileNotFoundError:
        print("Error: d2 command not found. Make sure d2 is installed and in your PATH.")
        svg_file_path = None
    
    return d2_file_path, svg_file_path


def filter_opacity_graph(graph):
    """ 
    Filter SubGraph, remove nodes and edges:  
    - Keep nodes with non-zero opacity
    - Remove nodes with zero opacity
    - Remove edges accordingly 
    """
    filtered_graph = {}
    for node_id, node_data in graph.items():
        if node_data['opacity'] > -1:
            filtered_graph[node_id] = node_data.copy()
            filtered_graph[node_id]['edges'] = [
                edge for edge in node_data['edges']
                if graph[edge]['opacity'] > 0
            ]
            filtered_graph[node_id]['edges'] = list(set(filtered_graph[node_id]['edges']))
    return filtered_graph



def decide_opacity_of_dag(dag: dict, progress: float = 1.0, cap_node_number: int = 200) -> dict:
    # Adjust importance scores based on hierarchy
    importance_groups = {}
    for node, data in dag.items():
        importance = data.get('importance', 0)
        file_path = data.get('file_path', '')
        level = file_path.count('/')
        adjusted_importance = importance + (1 / (level + 1))  # Adjust importance based on level
        if adjusted_importance not in importance_groups:
            importance_groups[adjusted_importance] = []
        importance_groups[adjusted_importance].append(node)

    # Sort nodes by adjusted importance
    sorted_nodes = sorted(dag.items(), key=lambda x: x[1]['importance'], reverse=True)

    # Calculate opacities
    scores = np.array([data['importance'] for (_, data) in sorted_nodes])
    opacities = scores / scores.max()
    
    # buffer period 
    bp = 0.2
    if progress < 0.2:
        max_opacity = opacities[opacities < 1.0].max() if len(opacities[opacities < 1.0]) > 0 else 0
        target_add_opacity = (1.0 - max_opacity) * bp 
        target_opacities = np.minimum(opacities + target_add_opacity, 1.0)
        begin_opacity = np.where(opacities < 1.0, 0.0, opacities)
        # interpolate between 
        interpolate_progress = progress * (1/bp)
        opacities = interpolate_progress * (target_opacities - begin_opacity) + begin_opacity
    else: 
        # Apply progress
        max_opacity = opacities[opacities < 1.0].max() if len(opacities[opacities < 1.0]) > 0 else 0
        add_opacity = (1.0 - max_opacity) * progress
        opacities = np.minimum(opacities + add_opacity, 1.0)

    # Cap the number of visible nodes
    opacities[cap_node_number:] = -1

    # Update the dag with new opacities
    for (node, data), opacity in zip(sorted_nodes, opacities):
        dag[node]['opacity'] = float(opacity)

    dag = filter_opacity_graph(dag)
    dag = assign_levels(dag)

    return dag


# SUDO Gif: level-wise appearing animation with D2-diagram

def cap_dag_count(dag: dict, cap_node_number: int = 200) -> dict:
    # Adjust importance scores based on hierarchy
    importance_groups = {}
    for node, data in dag.items():
        importance = data.get('importance', 0)
        file_path = data.get('file_path', '')
        level = file_path.count('/')
        adjusted_importance = importance + (1 / (level + 1))  # Adjust importance based on level
        if adjusted_importance not in importance_groups:
            importance_groups[adjusted_importance] = []
        importance_groups[adjusted_importance].append(node)

    # Sort nodes by adjusted importance
    sorted_nodes = sorted(dag.items(), key=lambda x: x[1]['importance'], reverse=True)
    
    # Assign 1.0 opacity to the first `cap_node_number` nodes, rest are set to -1
    opacities = np.ones(len(sorted_nodes))
    opacities[cap_node_number:] = -1

    # Update the dag with new opacities
    for (node, data), opacity in zip(sorted_nodes, opacities):
        dag[node]['opacity'] = float(opacity)

    return filter_opacity_graph(dag)


def assign_levels(sub_dag):
    # Create a dictionary to store incoming edges for each node
    incoming_edges = {node: set() for node in sub_dag}
    for node, data in sub_dag.items():
        for edge in data.get('edges', []):
            incoming_edges[edge].add(node)
    
    # Find nodes with no incoming edges (level 1)
    level = 1
    current_level_nodes = [node for node, edges in incoming_edges.items() if not edges]
    
    # If there are no nodes without incoming edges, start with any node
    if not current_level_nodes:
        current_level_nodes = [next(iter(sub_dag))]
    
    # Keep track of assigned nodes
    assigned_nodes = set()
    
    # Assign levels to all nodes
    while current_level_nodes:
        for node in current_level_nodes:
            sub_dag[node]['level'] = level
            assigned_nodes.add(node)
        
        # Find next level nodes
        next_level_nodes = []
        for node in sub_dag:
            if node not in assigned_nodes:
                if all(parent in assigned_nodes for parent in incoming_edges[node]):
                    next_level_nodes.append(node)
        
        # If no new nodes were found, but there are still unassigned nodes,
        # add one of the unassigned nodes to break potential cycles
        if not next_level_nodes and len(assigned_nodes) < len(sub_dag):
            unassigned = set(sub_dag.keys()) - assigned_nodes
            next_level_nodes.append(next(iter(unassigned)))
        
        current_level_nodes = next_level_nodes
        level += 1
    
    # Assign the highest level + 1 to any remaining unassigned nodes
    for node in sub_dag:
        if 'level' not in sub_dag[node]:
            sub_dag[node]['level'] = level
    
    return sub_dag


def set_full_opacity(dag): 
    """
    Set the opacity of all nodes in the DAG to 1.0 (fully opaque).
    
    Args:
    dag (dict): The directed acyclic graph represented as a dictionary.
    
    Returns:
    dict: The updated DAG with all nodes set to full opacity.
    """
    for node in dag:
        dag[node]['opacity'] = 1.0
    return dag


def generate_opacity_frames(sub_dag, frame_count, static_portion: float = 0.2):
    """ 
    Generate opacity frames for the DAG animation
    - Nodes appear gradually from top to bottom
    - At the end, all nodes will be fully visible for 'static_portion * frame_count' frames
    """
    # Reset opacity to zero for all nodes
    for node in sub_dag:
        sub_dag[node]['opacity'] = 0.0
    
    # Sort nodes by level and then by their order in the dictionary
    sorted_nodes = sorted(sub_dag.items(), key=lambda x: (x[1]['level'], list(sub_dag.keys()).index(x[0])))
    
    frames = []
    for frame in range(frame_count):
        current_frame_dag = copy.deepcopy(sub_dag)
        
        # Calculate the overall progress for this frame
        overall_progress = (frame + 1) / frame_count
        
        for i, (node_id, node_data) in enumerate(sorted_nodes):
            # Calculate the node's individual progress
            node_progress = i / (len(sorted_nodes) - 1)
            
            # If the overall progress has reached this node's turn to appear
            if overall_progress >= node_progress:
                # Calculate the node's opacity based on how long it's been visible
                node_opacity = min(1.0, (overall_progress - node_progress) / (1 / (len(sorted_nodes) - 1)))
                current_frame_dag[node_id]['opacity'] = node_opacity
            else:
                current_frame_dag[node_id]['opacity'] = 0.0
                    
                    
        # Check if we're in the last 20% of frames
        if frame >= frame_count * 0.8:
            # Calculate progress for the final interpolation
            final_progress = (frame - frame_count * 0.8) / (frame_count * 0.2)
            
            # Gradually increase opacity for all nodes to reach 1.0 at the end
            for node_id in current_frame_dag:
                current_opacity = current_frame_dag[node_id]['opacity']
                # Ensure we don't decrease opacity, only increase it
                new_opacity = max(current_opacity, min(1.0, current_opacity + (1.0 - current_opacity) * final_progress))
                current_frame_dag[node_id]['opacity'] = new_opacity            
        
        frames.append(current_frame_dag)
        
    # Add static frames at the end
    for _ in range(int(frame_count * static_portion)):
        frames.append(copy.deepcopy(current_frame_dag))
    
    return frames
  

def create_gif(png_files: list, output_file: str = "commit_dag_evolution.gif", fps: int = 2):
    # Define a common size for all frames
    MAX_SIZE = (2048, 1024)  # You can adjust this as needed

    # Create GIF
    images = []
    for png_file in tqdm(png_files, desc="Creating GIF"):
        if os.path.exists(png_file):
            # Open the image
            img = Image.open(png_file)
            
            # Resize the image while maintaining aspect ratio
            img.thumbnail(MAX_SIZE, Image.LANCZOS)
            
            # Create a new image with white background
            new_img = Image.new("RGB", MAX_SIZE, (255, 255, 255))
            
            # Paste the resized image onto the center of the new image
            new_img.paste(img, ((MAX_SIZE[0] - img.size[0]) // 2,
                                (MAX_SIZE[1] - img.size[1]) // 2))
            
            images.append(new_img)

    if images:
        # Calculate duration based on fps
        duration = int(1000 / fps)  # Convert fps to milliseconds between frames
        images[0].save(output_file, save_all=True, append_images=images[1:], 
                    duration=duration, loop=0)
        # print(f"Animation saved as {output_file}")
    else:
        print("No PNG files were found to create the GIF.")
    
    
def write_dependency_dags(dags, output_dir="d2_output", n_frames=10, cap_node_number=99):
    
    pbar = tqdm(total=len(dags)*n_frames, desc="Processing DAGs into PNG frames")

    for i, dag in enumerate(dags):
        
        # Animate the growing process by increasing progress from 0 to 1
        for progress in np.linspace(0, 1, n_frames):
            sub_dag = decide_opacity_of_dag(dag, progress=progress, cap_node_number=cap_node_number)
            d2_code = build_d2_from_dag(sub_dag, include_overhead=True)
            
            # Save each frame as an SVG
            filename = f"commit_{i}_progress_{progress:.2f}"
            save_png_from_d2(d2_code, filename, output_dir=output_dir)
            
            time.sleep(0.5)  # Add a small delay between frames
            
            pbar.update(1)  # Update progress bar

    pbar.close()  # Close the progress bar when done
    
    
def file_to_preprocessed_img(file_path):
    
    if file_path.endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
    else:
        raise ValueError("Unknown file format")

    # Convert image to PNG
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    # Convert PNG to base64
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_base64