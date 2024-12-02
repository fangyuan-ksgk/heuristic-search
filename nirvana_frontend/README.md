# TinyDAG

A lightweight package for real-time manipulation of directed acyclic graphs (DAGs).

Features:
- Create and manage graph nodes
- Update connections in real-time
- Visualize graph structures
- Track node properties and relationships

### Installation
```bash
pip install -e .
```

### Run the server
Following commands can be used to run the server: 

- `tiny_dag_backend`: Starts the backend server
- `tiny_dag_frontend`: Launches the frontend development server
- `serve_tiny_dag`: Runs both frontend and backend in development mode


### Use front-end API 

```python
from tiny_dag import get_graph_state

# Get initial DAG state
graph_state = get_graph_state()

# Add nodes to the DAG
node1 = {
    "id": 1,
    "x": 300,
    "y": 300,
    "name": "Input Node",
    "target": "input",
    "input": [],
    "output": ["data"],
    "code": "data = 42",
    "fitness": 0.8,
    "reasoning": "Initial data input",
    "inputTypes": [],
    "outputTypes": ["int"]
}

node2 = {
    "id": 2,
    "x": 550,  # Will be automatically positioned by add_nodes()
    "y": 300,
    "name": "Processing Node",
    "target": "process",
    "input": ["data"],
    "output": ["result"],
    "code": "result = data * 2",
    "fitness": 0.7,
    "reasoning": "Double the input value",
    "inputTypes": ["int"],
    "outputTypes": ["int"]
}

# Create an edge connecting the nodes
edge = {
    "source": 1,
    "target": 2
}

# Add nodes and connections to the graph
add_nodes([node1, node2], [edge])
```

for more examples, refer to 'demo_frontend.ipynb' notebook
