from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

class Cable:
    def __init__(self, init_state: dict):
        self.state = init_state
        self.active_connection: WebSocket = None
        
    async def init_connection(self, websocket: WebSocket):
        
        await websocket.accept()
        self.active_connection = websocket
        print("Connection initalized successfully ...")

        await websocket.send_json(self.state)
        print(f":: Initial state sent: {self.state}")
        
    async def commute(self, websocket: WebSocket):
        
        while True:
            try:
                frontend_data = await websocket.receive_json()
                print("\n=== Frontend Message Received ===")
                print(f"Message contains keys: {frontend_data.keys()}")
                
                # Only update what frontend sends
                if 'nodes' in frontend_data:
                    print("\n=== Updating Nodes ===")
                    print(f"Received nodes: {frontend_data['nodes']}")
                    graph_state['nodes'] = frontend_data['nodes']
                
                if 'connections' in frontend_data:
                    print("\n=== Updating Connections ===")
                    print(f"Received connections: {frontend_data['connections']}")
                    graph_state['connections'] = frontend_data['connections']
                
                # Send current state back
                await websocket.send_json({
                    'type': 'graph_update',
                    'nodes': graph_state['nodes'],
                    'connections': graph_state['connections']
                })
                
            except WebSocketDisconnect:
                print("Client disconnected")
                break  # Exit the loop when client disconnects
                
            except Exception as e:
                print(f"\n=== Error Processing Message ===")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                if "disconnect message has been received" in str(e):
                    print("WebSocket disconnected, closing connection")
                    break  # Exit the loop on disconnect
                continue