class GraphManager:
    def __init__(self):
        self._state = {
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
    
    @property
    def state(self):
        return self._state
    
    def update_state(self, frontend_data):
        """The working update logic from main_backup"""
        if 'nodes' in frontend_data:
            self._state['nodes'] = frontend_data['nodes']
        if 'connections' in frontend_data:
            self._state['connections'] = frontend_data['connections']
    
    def add_node(self, node):
        """Add a single node"""
        self._state['nodes'].append(node)
    
    def add_connection(self, connection):
        """Add a single connection"""
        self._state['connections'].append(connection) 