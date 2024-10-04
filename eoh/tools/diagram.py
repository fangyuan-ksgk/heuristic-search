import os
import subprocess
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

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

get_parent_file = lambda node: node['file'].replace(".py", "").replace("/",".")
get_object_label = lambda node: node['name'].split("::")[-1].replace(".py", "")

def build_d2_node(node: dict, node_id: str, include_overhead: bool = False) -> str:
    parent_file = get_parent_file(node)
    object_label = get_object_label(node)
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
  style.stroke: red
  style.opacity: {opacity}
  style.stroke-width: 2
  style.stroke-dash: 5
  style.animated: true
}}"""

def get_object_name(node: dict) -> str:
    parent_file = get_parent_file(node)
    object_label = get_object_label(node)
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
    
    return png_file_path
    
    
    
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
      # Redirect stdout and stderr to devnull
        subprocess.run(["d2", d2_file_path, svg_file_path], check=True)
    except subprocess.CalledProcessError as e:
        svg_file_path = None
    except FileNotFoundError:
        print("Error: d2 command not found. Make sure d2 is installed and in your PATH.")
        svg_file_path = None
    
    return d2_file_path, svg_file_path
  
def save_d2_as_png(save_name: str):
  d2_file_path = save_name + ".d2"
  png_file_path = save_name + ".png"
  try: 
    with open(os.devnull, 'w') as devnull:
      subprocess.run(["d2", d2_file_path, png_file_path], check=True, stdout=devnull, stderr=devnull)
    print(f"D2 diagram saved as {png_file_path}")
  except subprocess.CalledProcessError as e:
    print(f"Error generating PNG: {e}")
    png_file_path = None
  except FileNotFoundError:
    print("Error: d2 command not found. Make sure d2 is installed and in your PATH.")
    png_file_path = None
  return png_file_path


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
  
  
def create_gif(png_files: list, output_file: str = "commit_dag_evolution.gif"):
    # Define a common size for all frames
    MAX_SIZE = (1024, 512)  # You can adjust this as needed

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
        images[0].save(output_file, save_all=True, append_images=images[1:], 
                    duration=500, loop=0)
        # print(f"Animation saved as {output_file}")
    else:
        print("No PNG files were found to create the GIF.")
        
        
def create_gif_from_d2_files(d2_files):
    # Convert D2 to PNGs
    png_files = []
    for d2_file in d2_files:
        png_file = d2_file.replace(".d2", ".png")
        save_d2_as_png(d2_file, png_file)
        png_files.append(png_file)

    # Create GIF
    create_gif(png_files)
    
    
def write_dependency_dags(dags, output_dir="d2_output"):
    
    pbar = tqdm(total=len(dags)*10, desc="Processing DAGs into PNG frames")

    for i, dag in enumerate(dags):
        
        # Animate the growing process by increasing progress from 0 to 1
        for progress in np.linspace(0, 1, 10):
            sub_dag = decide_opacity_of_dag(dag, progress=progress, cap_node_number=15)
            d2_code = build_d2_from_dag(sub_dag, include_overhead=True)
            
            # Save each frame as an SVG
            filename = f"commit_{i}_progress_{progress:.2f}"
            save_png_from_d2(d2_code, filename, output_dir=output_dir)
            
            time.sleep(0.5)  # Add a small delay between frames
            
            pbar.update(1)  # Update progress bar

    pbar.close()  # Close the progress bar when done