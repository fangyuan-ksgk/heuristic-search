from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
        self._state["nodes"].append(node_data)
        return self._state 
    
    def add_edge(self, edge_data: dict):
        self._state["connections"].append(edge_data)
        return self._state 
    
    def remove_node(self, node_id: int):
        self._state["nodes"] = [node for node in self._state["nodes"] if node["id"] != node_id]
        return self._state 
    
    def remove_edge(self, edge_id: int):
        self._state["connections"] = [edge for edge in self._state["connections"] if edge["id"] != edge_id]
        return self._state
    
    def update_node(self, node_id: int, node_data: dict):
        for node in self._state["nodes"]:
            if node["id"] == node_id:
                node.update(node_data)
                break
        return self._state

# Global instance
graph_manager = GraphStateManager()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 