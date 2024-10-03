import os
import subprocess

d2_prefix = """vars: {
  d2-config: {
    sketch: true
  }
}
classes: {
  file: {
    label: ""
    shape: circle
    width: 40
    style: {
      fill: yellow
      shadow: true
    }
  }
}

classes: {
  class object: {
    label: ""
    shape: diamond
    width: 45
    height: 40
    style: {
      fill: blue
      shadow: true
    }
  }
}

classes: {
  class object: {
    label: ""
    shape: square
    width: 40
    height: 30
    style: {
      fill: gray
      shadow: false
    }
  }
}"""






object_template = """{object_name}.class: {object_type}
{object_name}.label: "{object_label}"
{object_name}: {{
  style: {{
    opacity: {opacity}
    stroke: "black"
    stroke-width: 5
    shadow: true
  }}
}}"""


def build_d2_node(node: dict, node_id: str) -> str:
    parent_file = node['file'].replace(".", "_")
    object_label = node['name'].replace(".", "_")
    object_name = f"{parent_file}.{object_label}"   
    object_type = node["type"]
    if object_type not in ["file", "class", "function", "object"]:
        object_type = "function"
    opacity = node["opacity"]
    opacity = min(1.0, max(0.0, opacity))
    opacity_str = f"{opacity:.2f}"
    
    return object_template.format(object_name=object_name, object_type=object_type, object_label=object_label, opacity=opacity_str)


link_template = """{start_object_id} -> {end_object_id}: {{
  style.stroke: green
  style.opacity: {opacity}
  style.stroke-width: 2
  style.stroke-dash: 5
  style.animated: true
}}"""

def get_object_name(node: dict) -> str:
    parent_file = node['file'].replace(".", "_")
    object_label = node['name'].replace(".", "_")
    object_name = f"{parent_file}.{object_label}" 
    return object_name

def build_d2_edge(str_node: dict, end_node: dict) -> str:
    opacity = min(1.0, max(0.0, end_node["opacity"]))
    opacity_str = f"{opacity:.2f}"
    start_object_name = get_object_name(str_node)
    end_object_name = get_object_name(end_node)
    return link_template.format(start_object_id=start_object_name, end_object_id=end_object_name, opacity=opacity_str)



def build_d2_from_dag(dag: dict) -> str:
    """
    Convert Sub-DAG dictionary into d2 code
    """
    d2_code = d2_prefix 

    for node_id, node in dag.items():
        object_str = build_d2_node(node, node_id)
        d2_code += "\n" + object_str

    for node_id, node in dag.items():
        edge_pairs = [(node_id, end_node) for end_node in node['edges']]    
        for start, end in edge_pairs:
            link_str = build_d2_edge(dag[start], dag[end])
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


