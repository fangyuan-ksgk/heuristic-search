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
        
        while True:
            try:
                frontend_data = await websocket.receive_json()
                print("\n=== Frontend Message Received ===")
                print(f"Message contains keys: {frontend_data.keys()}")
                
                # Log state before modifications
                print("\nBefore modifications:")
                print(f"Nodes: {graph_state['nodes']}")
                print(f"Connections: {graph_state['connections']}")
                
                if 'node_update' in frontend_data:
                    node_data = frontend_data['node_update']
                    print("\n=== Node Update Received ===")
                    print(f"Updating node: {node_data}")
                    
                    # Show what's being modified
                    for i, node in enumerate(graph_state['nodes']):
                        if node['id'] == node_data['id']:
                            print(f"Found existing node: {node}")
                            print(f"Updating with: {node_data}")
                            graph_state['nodes'][i] = {**node, **node_data}
                            print(f"Result: {graph_state['nodes'][i]}")
                
                if 'connection_update' in frontend_data:
                    connection_data = frontend_data['connection_update']
                    print("\n=== Connection Update Received ===")
                    print(f"Adding connection: {connection_data}")
                    graph_state['connections'].append(connection_data)
                
                # Log state after modifications
                print("\nAfter modifications:")
                print(f"Nodes: {graph_state['nodes']}")
                print(f"Connections: {graph_state['connections']}")
                
                # Send update back
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