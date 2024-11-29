import React, { useState } from 'react';

const SimpleDag = () => {
  const [nodes, setNodes] = useState([
    { 
      id: 1, 
      x: 300, 
      y: 300, 
      name: 'Node 1',
      target: '',
      input: [],
      output: [],
      code: '',
      reasoning: '',
      inputTypes: [],
      outputTypes: [],
    },
    { 
      id: 2, 
      x: 700, 
      y: 300, 
      name: 'Node 2',
      target: '',
      input: [],
      output: [],
      code: '',
      reasoning: '',
      inputTypes: [],
      outputTypes: [],
    }
  ]);
  
  const [isDragging, setIsDragging] = useState(false);
  const [draggedNode, setDraggedNode] = useState(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  const [editingNode, setEditingNode] = useState(null);

  // Node dimensions
  const nodeWidth = 160;
  const nodeHeight = 40;
  const cornerRadius = 8;

  // Add new state for hover tracking
  const [hoveredNode, setHoveredNode] = useState(null);

  // Initialize connections with Node 1 -> Node 2
  const [connections, setConnections] = useState([
    {
      source: 1,  // Node 1's ID
      target: 2   // Node 2's ID
    }
  ]);

  const handleMouseDown = (e, node) => {
    setIsDragging(true);
    setDraggedNode(node);
    
    const rect = e.target.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !draggedNode) return;

    const svgRect = e.currentTarget.getBoundingClientRect();
    const newX = e.clientX - svgRect.left;
    const newY = e.clientY - svgRect.top;

    setNodes(nodes.map(node => 
      node.id === draggedNode.id 
        ? { ...node, x: newX, y: newY }
        : node
    ));
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setDraggedNode(null);
  };

  const handleNodeClick = (e, node) => {
    if (!isDragging) {
      setEditingNode(node);
    }
  };

  const handleNodeUpdate = (updatedProperties) => {
    setNodes(nodes.map(node => 
      node.id === editingNode.id 
        ? { ...node, ...updatedProperties }
        : node
    ));
    setEditingNode(null);
  };

  // Calculate the path for the edge including the arrow
  const calculatePath = (startNode, endNode) => {
    // Connection points
    const start = {
      x: startNode.x + nodeWidth/2,  // Right side
      y: startNode.y                 // Center y
    };
    const end = {
      x: endNode.x - nodeWidth/2,    // Left side
      y: endNode.y                   // Center y
    };

    // Calculate control points for a smooth curve
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    
    // Adjust control points based on vertical distance
    const offsetX = Math.min(Math.abs(dx) * 0.5, 100);
    const controlPoint1 = {
      x: start.x + offsetX,
      y: start.y
    };
    const controlPoint2 = {
      x: end.x - offsetX,
      y: end.y
    };

    // Create the main path
    const path = `M ${start.x},${start.y} 
                 C ${controlPoint1.x},${controlPoint1.y} 
                   ${controlPoint2.x},${controlPoint2.y} 
                   ${end.x},${end.y}`;

    return path;
  };

  // Helper function to determine text color based on score
  const getScoreColor = (score) => {
    return score >= 60 ? '#22c55e' : '#ef4444'; // green-600 : red-500
  };

  // Modify handleAddNode to create a connection
  const handleAddNode = (sourceNode) => {
    const newNodeId = nodes.length + 1;
    const newNode = {
      id: newNodeId,
      x: sourceNode.x + 300,
      y: sourceNode.y,
      name: `Node ${newNodeId}`,
      target: '',
      input: [],
      output: [],
      code: '',
      reasoning: '',
      inputTypes: [],
      outputTypes: [],
    };
    
    // Add new node and create connection
    setNodes([...nodes, newNode]);
    setConnections([...connections, {
      source: sourceNode.id,
      target: newNodeId
    }]);
  };

  return (
    <svg 
      className="w-full h-screen bg-gray-50"
      viewBox="0 0 1000 600"
      preserveAspectRatio="xMidYMid meet"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <defs>
        <marker
          id="arrowhead"
          viewBox="0 0 10 10"
          refX="5"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8" />
        </marker>
      </defs>

      {/* Modified Edges - now using connections array */}
      {connections.map((connection, index) => {
        const startNode = nodes.find(node => node.id === connection.source);
        const endNode = nodes.find(node => node.id === connection.target);
        if (startNode && endNode) {
          return (
            <path
              key={`edge-${connection.source}-${connection.target}`}
              d={calculatePath(startNode, endNode)}
              stroke="#94a3b8"
              strokeWidth="2"
              fill="none"
              markerEnd="url(#arrowhead)"
            />
          );
        }
        return null;
      })}

      {/* Nodes - Modified to include hover detection and plus button */}
      {nodes.map(node => (
        <g 
          key={node.id}
          transform={`translate(${node.x - nodeWidth/2},${node.y - nodeHeight/2})`}
          onMouseEnter={() => setHoveredNode(node.id)}
          onMouseLeave={() => setHoveredNode(null)}
        >
          {/* Node rectangle */}
          <rect
            width={nodeWidth}
            height={nodeHeight}
            rx={cornerRadius}
            fill="white"
            stroke="#e2e8f0"
            strokeWidth="1"
            onMouseDown={(e) => handleMouseDown(e, node)}
            onClick={(e) => handleNodeClick(e, node)}
            className="cursor-pointer hover:stroke-blue-200"
          />
          
          {/* Node label */}
          <text
            x={nodeWidth / 2}
            y={nodeHeight / 2}
            textAnchor="middle"
            dy=".3em"
            fill="#64748b"
            fontSize="14"
          >
            {node.name}
          </text>

          {/* Add Button - only show when node is hovered */}
          {hoveredNode === node.id && (
            <g 
              transform={`translate(${nodeWidth + 10}, ${nodeHeight/2})`}
              onClick={(e) => {
                e.stopPropagation();
                handleAddNode(node);
              }}
              className="cursor-pointer"
            >
              <circle
                r="12"
                fill="white"
                stroke="#94a3b8"
                strokeWidth="2"
              />
              <line
                x1="-6"
                y1="0"
                x2="6"
                y2="0"
                stroke="#94a3b8"
                strokeWidth="2"
              />
              <line
                x1="0"
                y1="-6"
                x2="0"
                y2="6"
                stroke="#94a3b8"
                strokeWidth="2"
              />
            </g>
          )}
          
          {/* Modified Badge with larger circle */}
          <g transform={`translate(${nodeWidth - 30}, ${nodeHeight - 5})`}>
            <circle 
              r="14" 
              fill="white" 
              stroke={getScoreColor(90)} 
              strokeWidth="2.5"
              strokeOpacity="0.7"
            />
            <text
              x="0"
              y="0"
              textAnchor="middle"
              dy=".3em"
              fill={getScoreColor(90)}
              fontSize="10"
              fontWeight="500"
            >
              90%
            </text>
          </g>
        </g>
      ))}

      {editingNode && (
        <foreignObject x="0" y="0" width="100%" height="100%">
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center overflow-y-auto">
            <div className="bg-white p-6 rounded-lg shadow-xl max-w-lg w-full my-8">
              <h2 className="text-xl font-bold mb-4">Edit Node</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Name</label>
                  <input
                    type="text"
                    value={editingNode.name}
                    onChange={(e) => setEditingNode({...editingNode, name: e.target.value})}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Target</label>
                  <input
                    type="text"
                    value={editingNode.target}
                    onChange={(e) => setEditingNode({...editingNode, target: e.target.value})}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Code</label>
                  <textarea
                    value={editingNode.code}
                    onChange={(e) => setEditingNode({...editingNode, code: e.target.value})}
                    rows={4}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Reasoning</label>
                  <textarea
                    value={editingNode.reasoning}
                    onChange={(e) => setEditingNode({...editingNode, reasoning: e.target.value})}
                    rows={4}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                {/* Add input/output arrays and types as comma-separated values */}
                <div>
                  <label className="block text-sm font-medium text-gray-700">Input (comma-separated)</label>
                  <input
                    type="text"
                    value={editingNode.input.join(', ')}
                    onChange={(e) => setEditingNode({...editingNode, input: e.target.value.split(',').map(s => s.trim())})}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Output (comma-separated)</label>
                  <input
                    type="text"
                    value={editingNode.output.join(', ')}
                    onChange={(e) => setEditingNode({...editingNode, output: e.target.value.split(',').map(s => s.trim())})}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Input Types (comma-separated)</label>
                  <input
                    type="text"
                    value={editingNode.inputTypes.join(', ')}
                    onChange={(e) => setEditingNode({...editingNode, inputTypes: e.target.value.split(',').map(s => s.trim())})}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Output Types (comma-separated)</label>
                  <input
                    type="text"
                    value={editingNode.outputTypes.join(', ')}
                    onChange={(e) => setEditingNode({...editingNode, outputTypes: e.target.value.split(',').map(s => s.trim())})}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    onClick={() => setEditingNode(null)}
                    className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => handleNodeUpdate(editingNode)}
                    className="px-4 py-2 bg-blue-600 border border-transparent rounded-md text-sm font-medium text-white hover:bg-blue-700"
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          </div>
        </foreignObject>
      )}
    </svg>
  );
};

export default SimpleDag;