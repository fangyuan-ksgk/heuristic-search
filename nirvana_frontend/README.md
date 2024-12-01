# TinyDAG

A lightweight package for real-time manipulation of directed acyclic graphs (DAGs).

Features:
- Create and manage graph nodes
- Update connections in real-time
- Visualize graph structures
- Track node properties and relationships

## Setup and Installation

TinyDAG can be installed directly from the source. The package provides several command-line tools for running both the backend and frontend services.

### Installation
```bash
pip install -e .
```

### Available Commands

After installation, the following commands will be available:

- `tiny_dag_backend`: Starts the backend server
- `tiny_dag_frontend`: Launches the frontend development server
- `serve_tiny_dag`: Runs both frontend and backend in development mode

### Dependencies

The package requires the following main dependencies:
- FastAPI
- Uvicorn
- WebSockets
- Honcho
