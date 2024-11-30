import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test data
test_node = {
    "id": 2,
    "x": 300,
    "y": 300,
    "name": "Testy",
    "target": "test",
    "input": ["input1"],
    "output": ["output1"],
    "code": "print('test')",
    "fitness": 0.8,
    "reasoning": "test reasoning",
    "inputTypes": ["str"],
    "outputTypes": ["str"]
}

test_edge = {
    "id": 1,
    "source": 1,
    "target": 2
}

def get_graph_state():
    response = requests.get(f"{BASE_URL}/api/graph")
    assert response.status_code == 200
    graph_state = response.json()
    return graph_state 

def add_node(node):
    response = requests.post(f"{BASE_URL}/api/nodes", json=node)
    try:
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        return response.json()
    except:
        print("Add Node Issue Message: ", response.json())
        return False
    
def update_graph_state(state):
    """Update the entire graph state"""
    try:
        response = requests.put(f"{BASE_URL}/api/graph", json=state)
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Update Graph State Issue Message: {response.json()}")
            return False
    except Exception as e:
        print(f"Update Graph State Issue Message: {str(e)}")
        return False
    
def add_edge(edge):
    try:
        # Print request details for debugging
        endpoint = f"{BASE_URL}/api/edges"
        print(f"Sending request to: {endpoint}")
        print(f"Edge data: {json.dumps(edge, indent=2)}")
        
        response = requests.post(endpoint, json=edge)
        
        # Print response details
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        return response_data
    except requests.exceptions.ConnectionError:
        print("Connection Error: Is the server running?")
        return False
    except json.JSONDecodeError:
        print(f"Failed to decode response: {response.text}")
        return False
    except Exception as e:
        print(f"Add Edge Issue Message: {str(e)}")
        return False
    
def update_node(node_id: int, node: dict): 
    response = requests.put(f"{BASE_URL}/api/nodes/{node_id}", json=node)
    try:
        assert response.status_code == 200
        # Update assertion to match server response structure
        assert "status" in response.json() and response.json()["status"] == "success"
        return response.json()
    except AssertionError:
        print("Update Node Issue Message:", response.json())
        return False
    
def delete_node(node_id):
    response = requests.delete(f"{BASE_URL}/api/nodes/{node_id}")
    try:
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        return response.json()
    except:
        print("Delete Node Issue Message: ", response.json())
        return False
    
def delete_edge(edge_id):
    response = requests.delete(f"{BASE_URL}/api/edges/{edge_id}")
    try:
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        return response.json()
    except:
        print("Delete Edge Issue Message: ", response.json())
        return False
    
# Derivative Functions

def decide_xy_pos(source_id, graph_state, same_level_nodes, X_OFFSET, Y_OFFSET):
    # Decide X position 
    avg_x = graph_state["nodes"][source_id]["x"] + X_OFFSET
    if same_level_nodes:
        avg_x = sum(n["x"] for n in same_level_nodes) / len(same_level_nodes)
    x_pos = avg_x
    
    # Decide Y position 
    y_positions = [n["y"] for n in same_level_nodes]
    y_min, y_max = min(y_positions), max(y_positions) if y_positions else (0, 0)
    
    y_start = graph_state["nodes"][source_id]["y"]
    
    if abs(y_start - y_min) > abs(y_start - y_max): 
        y_pos = y_max + Y_OFFSET
    else:
        y_pos = y_min - Y_OFFSET
        
    return x_pos, y_pos 


def add_nodes(nodes, edges):

    graph_state = get_graph_state()

    X_OFFSET = 250
    Y_OFFSET = 100

    # Group nodes by their source nodes
    source_groups = {}
    for node in nodes:
        source_ids = [edge['source'] for edge in edges if edge['target'] == node['id']]
        source_id = source_ids[0] if source_ids else None
        if source_id not in source_groups:
            source_groups[source_id] = []
        source_groups[source_id].append(node)

    # Collect node at same level
    for source_id, group_nodes in source_groups.items():
        for target_node in group_nodes:
            same_level_node_ids = [connection["target"] for connection in graph_state["connections"] if connection["source"] == source_id]
            same_level_nodes = [node for node in graph_state["nodes"] if node["id"] in same_level_node_ids]  
            
            x_pos, y_pos = decide_xy_pos(source_id, graph_state, same_level_nodes, X_OFFSET, Y_OFFSET)
            
            target_node["x"], target_node["y"] = x_pos, y_pos
            graph_state["nodes"].append(target_node)
            graph_state["connections"].append({"source": source_id, "target": target_node["id"]})

    # update graph state 
    update_graph_state(graph_state)

    return
