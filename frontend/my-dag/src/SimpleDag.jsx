import React, { useState, useRef, useEffect } from 'react';

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
      fitness: .7,
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
      fitness: 0.0,
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

  // Add new state for pan and zoom
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, width: 1000, height: 600 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState({ x: 0, y: 0 });
  const svgRef = useRef(null);

  // Add a window resize handler
  const [windowSize, setWindowSize] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  });

  // Add useEffect to handle window resizing
  useEffect(() => {
    const handleResize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
      
      // Update viewBox to maintain aspect ratio
      setViewBox(prev => ({
        ...prev,
        width: window.innerWidth,
        height: window.innerHeight
      }));
    };

    window.addEventListener('resize', handleResize);
    
    // Initial setup
    handleResize();

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleMouseDown = (e, node) => {
    setIsDragging(true);
    setDraggedNode(node);
    
    // Calculate offset relative to the node's position, not the DOM element
    const svgRect = svgRef.current.getBoundingClientRect();
    const scale = svgRect.width / viewBox.width;
    
    setDragOffset({
      x: (e.clientX - svgRect.left) / scale + viewBox.x - node.x,
      y: (e.clientY - svgRect.top) / scale + viewBox.y - node.y
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging || !draggedNode) return;

    const svgRect = svgRef.current.getBoundingClientRect();
    const scale = svgRect.width / viewBox.width;
    
    // Calculate new position considering viewBox and scale
    const newX = (e.clientX - svgRect.left) / scale + viewBox.x - dragOffset.x;
    const newY = (e.clientY - svgRect.top) / scale + viewBox.y - dragOffset.y;

    setNodes(nodes.map(node => 
      node.id === draggedNode.id 
        ? { ...node, x: newX, y: newY }
        : node
    ));
  };

  const handleMouseUp = () => {
    // Only reset dragging states, don't trigger node click
    setIsDragging(false);
    setDraggedNode(null);
  };

  const handleNodeClick = (e, node) => {
    // Only show info box if we haven't been dragging
    if (!isDragging && !draggedNode) {
      const svgRect = svgRef.current.getBoundingClientRect();
      const scale = svgRect.width / viewBox.width;
      
      // Calculate initial position
      let screenX = (node.x - viewBox.x) * scale + svgRect.left;
      let screenY = (node.y - viewBox.y) * scale + svgRect.top;
      
      // Get viewport dimensions
      const viewportHeight = window.innerHeight;
      const viewportWidth = window.innerWidth;
      
      // Chat bubble dimensions (approximate)
      const bubbleWidth = 384; // w-96 = 24rem = 384px
      const bubbleHeight = Math.min(viewportHeight * 0.8, 600); // max-h-[80vh]
      const bottomPadding = 80; // Extra padding for buttons
      
      // Adjust position to keep bubble in viewport
      if (screenX + bubbleWidth + 20 > viewportWidth) {
        screenX = screenX - bubbleWidth - 40;
      }
      
      if (screenY + bubbleHeight + bottomPadding > viewportHeight) {
        screenY = Math.max(20, viewportHeight - bubbleHeight - bottomPadding);
      }
      
      if (screenY < 20) {
        screenY = 20;
      }
      
      setEditingNode({
        ...node,
        screenX: screenX,
        screenY: screenY
      });
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
    // Convert score from 0-1 to 0-100
    const percentage = score * 100;
    
    if (percentage <= 50) {
      // Red (#dc2626) to light red (#ef4444)
      const redIntensity = Math.floor(220 + (percentage * 0.48));
      return `rgb(${redIntensity}, ${Math.floor(38 + percentage)}, ${Math.floor(38 + percentage)})`;
    } else {
      // Yellow (#eab308) through light green (#22c55e) to dark green (#15803d)
      if (percentage <= 75) {
        // Yellow to light green (50-75%)
        const ratio = (percentage - 50) / 25;
        return `rgb(${Math.floor(234 - (ratio * 191))}, ${Math.floor(179 + (ratio * 18))}, ${Math.floor(8 + (ratio * 86))})`;
      } else {
        // Light green to dark green (75-100%)
        const ratio = (percentage - 75) / 25;
        return `rgb(${Math.floor(34 - (ratio * 13))}, ${Math.floor(197 - (ratio * 69))}, ${Math.floor(94 - (ratio * 33))})`;
      }
    }
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

  // Add pan handlers
  const handleSvgMouseDown = (e) => {
    // Only start panning if it's a middle-click or space + left-click
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      setIsPanning(true);
      setStartPan({ x: e.clientX, y: e.clientY });
      e.preventDefault();
    }
  };

  const handleSvgMouseMove = (e) => {
    if (isDragging) {
      handleMouseMove(e);
    } else if (isPanning) {
      const dx = (e.clientX - startPan.x) * (viewBox.width / svgRef.current.clientWidth);
      const dy = (e.clientY - startPan.y) * (viewBox.height / svgRef.current.clientHeight);
      
      setViewBox(prev => ({
        ...prev,
        x: prev.x - dx,
        y: prev.y - dy
      }));
      
      setStartPan({ x: e.clientX, y: e.clientY });
    }
  };

  const handleSvgMouseUp = () => {
    setIsPanning(false);
    handleMouseUp();
  };

  // Add zoom handler
  const handleWheel = (e) => {
    // If editing a node, don't prevent default scroll behavior
    if (editingNode) {
      return;
    }
    
    // Only handle zoom when not editing
    e.preventDefault();
    const delta = e.deltaY;
    const scaleFactor = delta > 0 ? 1.1 : 0.9;

    // Get mouse position relative to SVG
    const svgRect = svgRef.current.getBoundingClientRect();
    const mouseX = e.clientX - svgRect.left;
    const mouseY = e.clientY - svgRect.top;

    // Convert mouse position to SVG coordinates
    const svgX = (mouseX / svgRect.width) * viewBox.width + viewBox.x;
    const svgY = (mouseY / svgRect.height) * viewBox.height + viewBox.y;

    setViewBox(prev => {
      const newWidth = prev.width * scaleFactor;
      const newHeight = prev.height * scaleFactor;
      
      return {
        x: svgX - (mouseX / svgRect.width) * newWidth,
        y: svgY - (mouseY / svgRect.height) * newHeight,
        width: newWidth,
        height: newHeight
      };
    });
  };

  return (
    <>
      <svg 
        ref={svgRef}
        className="w-full h-screen"
        viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
        preserveAspectRatio="xMidYMid meet"
        onMouseDown={handleSvgMouseDown}
        onMouseMove={handleSvgMouseMove}
        onMouseUp={handleSvgMouseUp}
        onMouseLeave={handleSvgMouseUp}
        onWheel={handleWheel}
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
          <g key={node.id}>
            {/* Node rectangle */}
            <rect
              x={node.x - nodeWidth/2}
              y={node.y - nodeHeight/2}
              width={nodeWidth}
              height={nodeHeight}
              rx={cornerRadius}
              className="fill-white stroke-gray-300"
            />
            
            {/* Node name */}
            <text
              x={node.x - nodeWidth/5}
              y={node.y + 5}
              className="text-sm fill-gray-700"
            >
              {node.name}
            </text>
            
            {/* Node status circle with emoji or percentage */}
            <circle
              cx={node.x + nodeWidth/3}
              cy={node.y + nodeHeight/5}
              r="15"
              className="fill-white"
              stroke={getScoreColor(node.fitness)}
              strokeWidth="1.5"
            />
            <text
              x={node.x + nodeWidth/3}
              y={node.y + nodeHeight/4}
              className="text-center"
              fill={getScoreColor(node.fitness)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={node.fitness === undefined || node.fitness === null || node.fitness === 0 ? "14" : "11"}
            >
              {node.fitness === undefined || node.fitness === null || node.fitness === 0 ? "🚧" : `${Math.round(node.fitness * 100)}%`}
            </text>
          </g>
        ))}
      </svg>

      {editingNode && (
        <div className="fixed inset-0 z-50">
          <div className="absolute inset-0 bg-black bg-opacity-30" onClick={() => setEditingNode(null)} />
          
          <div 
            className="absolute bg-white rounded-lg shadow-xl w-96 overflow-y-auto"
            style={{
              left: `${editingNode.screenX + 20}px`,
              top: `${editingNode.screenY}px`,
              maxHeight: 'calc(100vh - 100px)'
            }}
          >
            <div className="relative p-6 space-y-4">
              <h2 className="text-xl font-bold mb-4">Edit Node</h2>
              
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

              <div className="flex justify-end space-x-3 mt-6 sticky bottom-0 bg-white py-3 border-t">
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
      )}
    </>
  );
};

export default SimpleDag;