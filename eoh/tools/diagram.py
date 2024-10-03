import os
import subprocess
import numpy as np

d2_prefix = """vars: {
  d2-config: {
    sketch: true
  }
}
classes: {
  file: {
    label: ""
    shape: diamond
    style: {
      fill: yellow
      shadow: true
    }
  }
}

classes: {
  class: {
    label: ""
    shape: rectangle
    style: {
      fill: lightblue
      shadow: true
    }
  }
}

classes: {
  object: {
    label: ""
    shape: hexagon
    style: {
      fill: while
      shadow: false
    }
  }
}"""






object_template_with_overhead = """{object_name}.class: {object_type}
{object_name}.label: "{object_label}"
{object_name}: {{
  style: {{
    opacity: {opacity}
    stroke: "black"
    stroke-width: 4
    shadow: true
  }}
}}"""

object_template = """{object_name}.class: {object_type}
{object_name}.label: "{object_label}"
{object_name}: {{
  style: {{
    opacity: {opacity}
    stroke: "black"
    stroke-width: 4
    shadow: true
  }}
}}"""


def build_d2_node(node: dict, node_id: str, include_overhead: bool = False) -> str:
    parent_file = node['file'].replace(".", "_")
    object_label = node['name'].split("::")[-1].replace(".", "_")
    if include_overhead:
        object_name = f"{parent_file}.{object_label}"  
    else:
        object_name = object_label 
    object_type = node["type"]
    if object_type not in ["file", "class"]:
        object_type = "function"
    opacity = node["opacity"]
    opacity = min(1.0, max(0.0, opacity))
    opacity_str = f"{opacity:.2f}"
    
    return object_template.format(object_name=object_name, object_type=object_type, object_label=object_label, opacity=opacity_str)

# Would love to create more caveat line visual
link_template = """{start_object_id} -> {end_object_id}: {{
  style.stroke: black
  style.opacity: {opacity}
  style.stroke-width: 2
}}"""

link_file_template = """{start_object_id} -> {end_object_id}: {{
  style.stroke: yellow
  style.opacity: {opacity}
  style.stroke-width: 2
  style.stroke-dash: 5
  style.animated: true
}}"""

def get_object_name(node: dict) -> str:
    parent_file = node['file'].replace(".", "_")
    object_label = node['name'].split("::")[-1].replace(".", "_")
    object_name = f"{parent_file}.{object_label}" 
    return object_name

def get_label_name(node: dict) -> str:
    return node['name'].split("::")[-1].replace(".", "_")

def build_d2_edge(str_node: dict, end_node: dict, include_overhead: bool = False) -> str:
    opacity = min(1.0, max(0.0, end_node["opacity"]))
    opacity_str = f"{opacity:.2f}"
    if include_overhead:
        start_object_name = get_object_name(str_node)
        end_object_name = get_object_name(end_node)
    else:
        start_object_name = get_label_name(str_node)
        end_object_name = get_label_name(end_node)
    
    if str_node["type"] == "file" and end_node["type"] == "file":
        return link_file_template.format(start_object_id=start_object_name, end_object_id=end_object_name, opacity=opacity_str)
    if str_node["type"] == "file" and end_node["type"] != "file":
        return link_file_template.format(start_object_id=start_object_name, end_object_id=end_object_name, opacity=opacity_str)
    if str_node["type"] != "file":
        return link_template.format(start_object_id=start_object_name, end_object_id=end_object_name, opacity=opacity_str)


def build_d2_from_dag(dag: dict, include_overhead: bool = False) -> str:
    """
    Convert Sub-DAG dictionary into d2 code
    """
    d2_code = d2_prefix 

    for node_id, node in dag.items():
        object_str = build_d2_node(node, node_id, include_overhead)
        d2_code += "\n" + object_str

    for node_id, node in dag.items():
        edge_pairs = [(node_id, end_node) for end_node in node['edges']]    
        for start, end in edge_pairs:
            link_str = build_d2_edge(dag[start], dag[end], include_overhead)
            if link_str:
                d2_code += f"\n{link_str}"
            
    return d2_code


def save_d2_and_svg(d2_code, file_name, output_dir="d2_output"):
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
        subprocess.run(["d2", d2_file_path, svg_file_path], check=True)
        print(f"D2 diagram saved as {d2_file_path}")
        print(f"SVG file generated at {svg_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating SVG: {e}")
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
        if node_data['opacity'] > 0:
            filtered_graph[node_id] = node_data.copy()
            filtered_graph[node_id]['edges'] = [
                edge for edge in node_data['edges']
                if graph[edge]['opacity'] > 0
            ]
    return filtered_graph


def decide_opacity_of_dag(dag: dict, progress: float, cap_node_number: int = 15) -> dict:
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
    opacities[cap_node_number:] = 0

    # Update the dag with new opacities
    for (node, data), opacity in zip(sorted_nodes, opacities):
        dag[node]['opacity'] = float(opacity)

    return filter_opacity_graph(dag)