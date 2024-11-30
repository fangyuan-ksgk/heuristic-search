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
        print("Add Node Message: ", response.json())
        return False
    
def add_edge(edge):
    response = requests.post(f"{BASE_URL}/api/edges", json=edge)
    try:
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        return response.json()
    except:
        print("Add Edge Message: ", response.json())
        return False
    
def update_node(node_id: int, node: dict): 
    response = requests.put(f"{BASE_URL}/api/nodes/{node_id}", json=node)
    try:
        assert response.status_code == 200
        # Update assertion to match server response structure
        assert "status" in response.json() and response.json()["status"] == "success"
        return response.json()
    except AssertionError:
        print("Update Node Message:", response.json())
        return False
    
def delete_node(node_id):
    response = requests.delete(f"{BASE_URL}/api/nodes/{node_id}")
    try:
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        return response.json()
    except:
        print("Delete Node Message: ", response.json())
        return False
    
def delete_edge(edge_id):
    response = requests.delete(f"{BASE_URL}/api/edges/{edge_id}")
    try:
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        return response.json()
    except:
        print("Delete Edge Message: ", response.json())
        return False
    
# pass node position from front-end
