from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ready to slot this class into library file
# - Where do I put the util functions?
class GraphStateManager:
    def __init__(self):
        self._state = self._get_initial_state()
        self._websocket_connections = set()  # Track active connections
        
    def _get_initial_state(self):
        return {
            'nodes': [
                {
                    'id': 1,
                    'x': 300,
                    'y': 300,
                    'name': 'Black Node',
                    'target': '',
                    'input': [],
                    'output': [],
                    'code': '',
                    'fitness': 0.7,
                    'reasoning': '',
                    'inputTypes': [],
                    'outputTypes': [],
                }
            ],
            'connections': []
        }
    
    async def broadcast_state(self):
        """Broadcast current state to all connected websockets"""
        message = {
            'type': 'graph_update', # Logging purpose?
            'nodes': self._state['nodes'],
            'connections': self._state['connections']
        }
        # Broadcast to all connected clients
        for websocket in self._websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                print("Temporary connection issue encoutered, redialing ...")
                continue 
    
    def update_state(self, data):
        """Update state and trigger broadcast - can be called from notebook"""
        if 'nodes' in data:
            self._state['nodes'] = data['nodes']
        if 'connections' in data:
            self._state['connections'] = data['connections']
        
        # Return the current state for notebook use
        return self._state
    
    def get_state(self):
        """Get current state - for notebook use"""
        return self._state.copy()
    
    def add_node(self, node_data: dict):
        # check if the node id exist, add if it does not
        if any(node['id'] == node_data['id'] for node in self._state['nodes']):
            raise HTTPException(status_code=400, detail=f"Node with id {node_data['id']} already exists")
        self._state["nodes"].append(node_data)
        return self._state 
    
    def add_edge(self, edge_data: dict):

        # Check if source node exists
        if not any(node['id'] == edge_data['source'] for node in self._state['nodes']):
            raise HTTPException(status_code=400, detail=f"Source node {edge_data['source']} does not exist")
            
        # Check if target node exists  
        if not any(node['id'] == edge_data['target'] for node in self._state['nodes']):
            raise HTTPException(status_code=400, detail=f"Target node {edge_data['target']} does not exist")
        
        # Check for duplicate edge (same source and target)
        if any(edge['source'] == edge_data['source'] and edge['target'] == edge_data['target'] 
            for edge in self._state['connections']):
            raise HTTPException(status_code=400, detail=f"Edge from node {edge_data['source']} to {edge_data['target']} already exists")
                
        self._state["connections"].append(edge_data)
        return self._state
    
    def remove_node(self, node_id: int):
        self._state["nodes"] = [node for node in self._state["nodes"] if node["id"] != node_id]
        return self._state 
    
    def remove_edge(self, source: int, target: int):
        self._state["connections"] = [
            edge for edge in self._state["connections"] 
            if not (edge["source"] == source and edge["target"] == target)
        ]
        return self._state
    
    def update_node(self, node_id: int, node_data: dict):
        for node in self._state["nodes"]:
            if node["id"] == node_id:
                node.update(node_data)
                break
        return self._state

# Global Instance :: Key value circled between frontend and backend
graph_manager = GraphStateManager()

# Following while loop run forever, dialing between frontend and backend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        graph_manager._websocket_connections.add(websocket)
        print("[WS] New connection opened")
        
        # Send initial state
        print("[WS] Sending initial state...")
        await websocket.send_json({
            'type': 'graph_update',
            'nodes': graph_manager._state['nodes'],
            'connections': graph_manager._state['connections']
        })
        print("[WS] Initial state sent successfully")
        
        while True:
            try:
                print("[WS] Waiting for message...")
                data = await websocket.receive_json()
                print(f"[WS] Received data: {data}")
                
                print("[WS] Updating state...")
                graph_manager.update_state(data)
                
                print("[WS] Broadcasting updated state...")
                await graph_manager.broadcast_state()
                print("[WS] Broadcast complete")
                
            except WebSocketDisconnect:
                print("[WS] Client disconnected with WebSocketDisconnect")
                break
                
            except Exception as e:
                print(f"[WS] Error occurred: {str(e)}")
                print(f"[WS] Error type: {type(e)}")
                print(f"[WS] Error details: {e.__dict__}")
                break
                
    finally:
        graph_manager._websocket_connections.discard(websocket)
        print("[WS] Connection cleaned up and closed")


# For the backend program to twist the values with simple api and forget about async stuff

class NodeData(BaseModel):
    id: int
    x: float
    y: float
    name: str
    target: str = ''
    input: list = []
    output: list = []
    code: str = ''
    fitness: float = 0.7
    reasoning: str = ''
    inputTypes: list = []
    outputTypes: list = []

class EdgeData(BaseModel):
    source: int
    target: int

# Add these new endpoints
@app.get("/api/graph")
def get_graph():
    """Get current graph state"""
    return graph_manager.get_state()

@app.post("/api/nodes")
async def add_node(node: NodeData):
    """Add a new node and broadcast update"""
    graph_manager.add_node(node.dict())
    await graph_manager.broadcast_state()
    return {"status": "success", "node": node}

@app.put("/api/nodes/{node_id}")
async def update_node(node_id: int, node: NodeData):
    """Update existing node and broadcast"""
    graph_manager.update_node(node_id, node.dict())
    await graph_manager.broadcast_state()
    return {"status": "success", "node": node}

@app.delete("/api/nodes/{node_id}")
async def delete_node(node_id: int):
    """Delete node and broadcast update"""
    graph_manager.remove_node(node_id)
    await graph_manager.broadcast_state()
    return {"status": "success"}

@app.post("/api/edges")
async def add_edge(edge: EdgeData):
    """Add new edge and broadcast update"""
    try:
        print(f"Received edge request: {edge.dict()}")
        # Check if nodes exist
        state = graph_manager.get_state()
        node_ids = [node['id'] for node in state['nodes']]
        print(f"Existing node IDs: {node_ids}")
        print(f"Looking for source: {edge.source}, target: {edge.target}")
        
        result = graph_manager.add_edge(edge.dict())
        await graph_manager.broadcast_state()
        return {"status": "success", "edge": edge}
    except Exception as e:
        print(f"Error adding edge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/edges/{source}/{target}")
async def delete_edge(source: int, target: int):
    """Delete edge and broadcast update"""
    graph_manager.remove_edge(source, target)
    await graph_manager.broadcast_state()
    return {"status": "success"}

@app.put("/api/graph")
async def update_graph_state(data: dict):
    """Update entire graph state"""
    try:
        graph_manager.update_state(data)
        await graph_manager.broadcast_state()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 