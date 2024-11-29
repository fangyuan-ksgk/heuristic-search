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

# Initialize graph state
# This is the global dictionary which everybody works on :: both human through frontend and AI through backend

graph_state = {
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        print("Connection opened")
        
        # Immediately send initial graph state to frontend
        initial_state = {
            'type': 'graph_update',
            'nodes': graph_state['nodes'],
            'connections': graph_state['connections']
        }
        await websocket.send_json(initial_state)
        print(f"Sent initial state: {initial_state}")
        
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
    
    except WebSocketDisconnect:
        print("Client disconnected during connection setup")
    except Exception as e:
        print(f"Connection error: {str(e)}")
    finally:
        print("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 