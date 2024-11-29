from fastapi import WebSocket, WebSocketDisconnect
import json

class WebSocketHandler:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        print("Connection opened")
        # Send initial state immediately upon connection
        await websocket.send_json({
            'type': 'graph_update',
            **self.graph_manager.state
        })
    
    async def handle_message(self, websocket: WebSocket):
        """The exact working message handling from main_backup"""
        while True:
            try:
                frontend_data = await websocket.receive_json()
                print("\n=== Frontend Message Received ===")
                print(f"Message contains keys: {frontend_data.keys()}")
                
                self.graph_manager.update_state(frontend_data)
                
                # Send update back
                await websocket.send_json({
                    'type': 'graph_update',
                    **self.graph_manager.state
                })
                
            except Exception as e:
                print(f"\n=== Error Processing Message ===")
                print(f"Error: {str(e)}")
                if "disconnect message has been received" in str(e):
                    break
                continue
    
    async def disconnect(self, websocket: WebSocket):
        print("Client disconnected")
    
    async def send_state(self, websocket: WebSocket):
        """Send current graph state to client"""
        state_update = {
            'type': 'graph_update',
            **self.graph_manager.state
        }
        print(f"\n=== Sending State to Client ===")
        print(f"State being sent: {state_update}")
        await websocket.send_json(state_update)
    
    async def broadcast_state(self):
        """Send state to all connected clients"""
        print(f"\n=== Broadcasting State to {len(self.active_connections)} clients ===")
        for connection in self.active_connections:
            await self.send_state(connection)
    
    async def handle_message(self, websocket: WebSocket, message: dict):
        """Process incoming message and update state"""
        print("\n=== Received Message ===")
        print(f"Message content: {message}")
        print("Before state update:", self.graph_manager.state)
        
        self.graph_manager.update_state(message)
        print("After state update:", self.graph_manager.state)
        
        await self.broadcast_state() 