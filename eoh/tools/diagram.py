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



link_template = """{start_object_id} -> {end_object_id}: {
  style: {
    stroke: green
    opacity: {opacity}
    stroke-width: 2
    stroke-dash: 5
    animated: true
  }
}"""


object_template = """{object_id}: {{
  shape: {object_type}
  label: "{object_name}"
  style: {{
    opacity: {opacity}
    stroke: black
    stroke-width: 2
    shadow: true
  }}
}}"""


def build_d2_node(node: dict, node_id: str) -> str:
    object_name = f"{node['file'].replace('.', '_')}.{node['name']}"
    object_type = node["type"]
    if object_type not in ["file", "class", "function", "object"]:
        object_type = "function"
    opacity = node["opacity"]
    opacity = min(1.0, max(0.0, opacity))
    opacity_str = f"{opacity:.2f}"
    
    return object_template.format(object_id=node_id, object_name=object_name, object_type=object_type, opacity=opacity_str)


def build_link_str(start_object_id: str, end_object_id: str, opacity: float):
    opacity = min(1.0, max(0.0, opacity))
    opacity_str = f"{opacity:.2f}"
    return link_template.format(start_object_id=start_object_id, end_object_id=end_object_id, opacity=opacity_str)