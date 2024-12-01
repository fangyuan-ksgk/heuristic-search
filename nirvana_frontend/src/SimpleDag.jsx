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
    e.stopPropagation();
    setIsDragging(true);
    setDraggedNode(node);
    
    const svgRect = svgRef.current.getBoundingClientRect();
    const scale = svgRect.width / viewBox.width;
    
    setDragOffset({
      x: (e.clientX - svgRect.left) / scale + viewBox.x - node.x,
      y: (e.clientY - svgRect.top) / scale + viewBox.y - node.y
    });
  };

  const handleMouseMove = (e) => {
    if (isDragging && draggedNode) {
      e.preventDefault();
      const svgRect = svgRef.current.getBoundingClientRect();
      const scale = svgRect.width / viewBox.width;
      
      const newX = (e.clientX - svgRect.left) / scale + viewBox.x - dragOffset.x;
      const newY = (e.clientY - svgRect.top) / scale + viewBox.y - dragOffset.y;

      setNodes(nodes.map(node => 
        node.id === draggedNode.id 
          ? { ...node, x: newX, y: newY }
          : node
      ));
    }
  };

  const handleMouseUp = () => {
    if (isDragging && draggedNode) {
      const updatedNode = nodes.find(node => node.id === draggedNode.id);
      if (updatedNode && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
          nodes: nodes,
          connections: connections
        }));
      }
    }
    
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

  const handleNodeUpdate = (nodeId, updatedNode) => {
    // Update local state first
    const updatedNodes = nodes.map(node => 
        node.id === nodeId ? {...node, ...updatedNode} : node
    );
    setNodes(updatedNodes);

    // Send complete state to backend
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            nodes: updatedNodes,
            connections: connections  // Send current connections too
        }));
    }
  };

  // Calculate the path for the edge including the arrow
  const calculatePath = (startNode, endNode) => {
    const start = {
      x: startNode.x + nodeWidth/2,
      y: startNode.y
    };
    const end = {
      x: endNode.x - nodeWidth/2,
      y: endNode.y
    };

    const dx = end.x - start.x;
    const dy = end.y - start.y;
    
    // Increase curve intensity based on distance
    const offsetX = Math.min(Math.abs(dx) * 0.7, 150);
    
    // Add slight vertical offset for more natural curves
    const verticalOffset = Math.min(Math.abs(dy) * 0.2, 30);
    
    const controlPoint1 = {
      x: start.x + offsetX,
      y: start.y + (dy > 0 ? verticalOffset : -verticalOffset)
    };
    const controlPoint2 = {
      x: end.x - offsetX,
      y: end.y + (dy > 0 ? -verticalOffset : verticalOffset)
    };

    return `M ${start.x},${start.y} 
            C ${controlPoint1.x},${controlPoint1.y} 
              ${controlPoint2.x},${controlPoint2.y} 
              ${end.x},${end.y}`;
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

  // Modify handleAddNode to send the new node to backend
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
        fitness: 0.0,
    };
    
    const newConnection = {
        source: sourceNode.id,
        target: newNodeId
    };
    
    // Update local state
    const updatedNodes = [...nodes, newNode];
    const updatedConnections = [...connections, newConnection];
    setNodes(updatedNodes);
    setConnections(updatedConnections);

    // Send complete state to backend
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            nodes: updatedNodes,
            connections: updatedConnections
        }));
    }
  };

  // Also need to send connection updates
  const handleConnectionUpdate = (newConnection) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            connection_update: newConnection
        }));
    }
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
    handleConnectionMove(e);
  };

  const handleSvgMouseUp = () => {
    setIsPanning(false);
    handleMouseUp();
    if (isDrawingConnection) {
      setIsDrawingConnection(false);
      setConnectionStart(null);
      setHoveredNode(null);
    }
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

  // Add new state for connection drawing
  const [isDrawingConnection, setIsDrawingConnection] = useState(false);
  const [connectionStart, setConnectionStart] = useState(null);
  const [tempConnectionEnd, setTempConnectionEnd] = useState({ x: 0, y: 0 });

  // Add new handler for removing nodes
  const handleRemoveNode = (nodeId) => {
    // Update local state
    const updatedNodes = nodes.filter(node => node.id !== nodeId);
    const updatedConnections = connections.filter(conn => 
        conn.source !== nodeId && conn.target !== nodeId
    );
    
    setNodes(updatedNodes);
    setConnections(updatedConnections);

    // Send complete state to backend
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            nodes: updatedNodes,
            connections: updatedConnections
        }));
    }
  };

  // Add connection drawing handlers
  const handleConnectionMove = (e) => {
    if (isDrawingConnection) {
      const svgRect = svgRef.current.getBoundingClientRect();
      const scale = svgRect.width / viewBox.width;
      const x = (e.clientX - svgRect.left) / scale + viewBox.x;
      const y = (e.clientY - svgRect.top) / scale + viewBox.y;
      setTempConnectionEnd({ x, y });

      // Find node under cursor
      const point = { x, y };
      const nodeUnderCursor = nodes.find(node => 
        node.id !== connectionStart.id && // Ignore source node
        isPointInNodeBounds(point, node)
      );
      
      if (nodeUnderCursor) {
        setHoveredNode(nodeUnderCursor.id);
      } else {
        setHoveredNode(null);
      }
    }
  };

  // Add a small buffer constant for hit detection
  const HIT_DETECTION_BUFFER = 20; // pixels

  // Add helper function to check if point is inside node bounds
  const isPointInNodeBounds = (point, node) => {
    return point.x >= (node.x - nodeWidth/2 - HIT_DETECTION_BUFFER) &&
           point.x <= (node.x + nodeWidth/2 + HIT_DETECTION_BUFFER) &&
           point.y >= (node.y - nodeHeight/2 - HIT_DETECTION_BUFFER) &&
           point.y <= (node.y + nodeHeight/2 + HIT_DETECTION_BUFFER);
  };

  // Add these state handlers at the top with other useState declarations
  const [fileInput, setFileInput] = useState(null);

  // Add this function to handle JSON file loading
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const jsonData = JSON.parse(e.target.result);
          // Assuming the JSON has 'nodes' and 'connections' arrays
          if (jsonData.nodes && jsonData.connections) {
            setNodes(jsonData.nodes);
            setConnections(jsonData.connections);
          } else {
            console.error('Invalid JSON format: missing nodes or connections');
          }
        } catch (error) {
          console.error('Error parsing JSON:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  // Add WebSocket state
  const [ws, setWs] = useState(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/ws');
    
    websocket.onopen = () => {
      console.log('Connected to Python backend');
    };
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.nodes && data.connections) {
        setNodes(data.nodes);
        setConnections(data.connections);
      }
    };
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    setWs(websocket);
    
    return () => {
      websocket.close();
    };
  }, []);

  const handleEditNode = (node) => {
    // Create a copy of the node without modifying its position
    const nodeToEdit = {
        ...node,
        x: node.x,  // Preserve original position
        y: node.y   // Preserve original position
    };
    setEditingNode(nodeToEdit);
  };

  // Add this state near your other state declarations
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');

  return (
    <div className="relative w-full h-screen overflow-hidden">
      <div className="absolute top-4 left-4 z-10">
        <input
          type="file"
          accept=".json"
          onChange={handleFileUpload}
          className="hidden"
          id="json-upload"
        />
        <label
          htmlFor="json-upload"
          className="bg-white px-4 py-2 rounded-md shadow-sm border border-gray-300 cursor-pointer hover:bg-gray-50"
        >
          Load JSON
        </label>
      </div>
      
      <svg 
        ref={svgRef}
        className="w-full h-full"
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
            <path d="M 0 0 L 10 5 L 0 10 z" fill="url(#arrowGradient)" />
          </marker>

          <circle id="dataPoint" r="3" fill="url(#dataPointGradient)" />

          <linearGradient id="nodeGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#f8fafc" />
          </linearGradient>

          <linearGradient id="borderGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#cbd5e1" />
            <stop offset="50%" stopColor="#94a3b8" />
            <stop offset="100%" stopColor="#cbd5e1" />
          </linearGradient>

          <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.1"/>
          </filter>

          <linearGradient id="connectionGradient" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#94a3b8" />
            <stop offset="50%" stopColor="#cbd5e1" />
            <stop offset="100%" stopColor="#94a3b8" />
          </linearGradient>

          <linearGradient id="arrowGradient">
            <stop offset="0%" stopColor="#94a3b8" />
            <stop offset="100%" stopColor="#cbd5e1" />
          </linearGradient>

          <radialGradient id="dataPointGradient">
            <stop offset="0%" stopColor="#60a5fa" />
            <stop offset="100%" stopColor="#3b82f6" />
          </radialGradient>

          <filter id="connectionShadow" x="-20%" y="-20%" width="140%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="2" />
            <feOffset dx="1" dy="1" result="offsetblur" />
            <feComponentTransfer>
              <feFuncA type="linear" slope="0.2" />
            </feComponentTransfer>
            <feMerge>
              <feMergeNode />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Modified Edges - now using connections array */}
        {connections.map((connection, index) => {
          const startNode = nodes.find(node => node.id === connection.source);
          const endNode = nodes.find(node => node.id === connection.target);
          if (startNode && endNode) {
            const path = calculatePath(startNode, endNode);
            return (
              <g key={`edge-${connection.source}-${connection.target}`}>
                <path
                  d={path}
                  stroke="#94a3b8"
                  strokeWidth="2"
                  fill="none"
                  markerEnd="url(#arrowhead)"
                />
                <circle r="3" fill="#3b82f6">
                  <animateMotion
                    dur="1.5s"
                    repeatCount="indefinite"
                    path={path}
                  />
                </circle>
              </g>
            );
          }
          return null;
        })}

        {/* Nodes - Modified to include hover detection and plus button */}
        {nodes.map(node => (
          <g 
            key={node.id}
            onMouseDown={(e) => handleMouseDown(e, node)}
            onClick={(e) => handleNodeClick(e, node)}
            onMouseEnter={() => !isDrawingConnection && setHoveredNode(node.id)}
            onMouseLeave={() => !isDrawingConnection && setHoveredNode(null)}
            onMouseUp={(e) => {
              if (isDrawingConnection && hoveredNode === node.id && connectionStart) {
                e.stopPropagation();
                const newConnection = {
                  source: connectionStart.id,
                  target: node.id
                };
                setConnections(prevConnections => [...prevConnections, newConnection]);
                setIsDrawingConnection(false);
                setConnectionStart(null);
                setHoveredNode(null);
              }
            }}
            style={{
              transformOrigin: `${node.x}px ${node.y}px`,  // Set transform origin to node center
              transform: `scale(${1/viewBox.scale})`,       // Apply inverse scale to counteract zoom
            }}
          >
            {/* Node rectangle with highlight when hovered during connection */}
            <rect
              x={node.x - nodeWidth/2}
              y={node.y - nodeHeight/2}
              width={nodeWidth}
              height={nodeHeight}
              rx={cornerRadius}
              fill="white"
              stroke="url(#borderGradient)"
              strokeWidth="1.5"
              filter="url(#dropShadow)"
              className={`transition-all duration-300 ${
                isDrawingConnection && hoveredNode === node.id ? 'stroke-blue-500 stroke-2' : ''
              }`}
            />
            
            {/* Node name */}
            <text
              x={node.x}
              y={node.y + 1}
              className="text-sm fill-gray-700 font-bold"
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {node.name}
            </text>
            
            {/* Node status pill - moved outside and to bottom right */}
            <rect
              x={node.x + nodeWidth/4 - 1}  // Changed from nodeWidth*92/400
              y={node.y + nodeHeight/2 + 1}      // Changed from nodeHeight*20/500
              width="32"
              height="22"
              rx="10"
              className="fill-white"
              stroke={getScoreColor(node.fitness)}
              strokeWidth="1.5"
              style={{ 
                transform: `scale(${1/viewBox.scale})`,
                transformOrigin: `${node.x + nodeWidth/2 + 21}px ${node.y + nodeHeight/2 + 11}px`  // Updated transform origin
              }}
            />
            <text
              x={node.x + nodeWidth/3 + 1}  // Changed from nodeWidth/3
              y={node.y + nodeHeight/2 + 13}  // Changed from nodeHeight*130/500
              className="text-center font-bold"
              fill={getScoreColor(node.fitness)}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={node.fitness === undefined || node.fitness === null || node.fitness === 0 ? "13" : "11"}
              style={{ 
                transform: `scale(${1/viewBox.scale})`,
                transformOrigin: `${node.x + nodeWidth/2 + 21}px ${node.y + nodeHeight/2 + 11}px`,  // Updated transform origin
                backgroundColor: 'transparent',
                paintOrder: 'stroke',
                userSelect: 'none'
              }}
            >
              {node.fitness === undefined || node.fitness === null || node.fitness === 0 ? "🚧" : `${Math.round(node.fitness * 100)}%`}
            </text>

            {/* Control buttons - show on hover */}
            {hoveredNode === node.id && !isDrawingConnection && (
              <>
                {/* Add node button (right) */}
                <g
                  transform={`translate(${node.x + nodeWidth/2 + 10}, ${node.y})`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleAddNode(node);
                  }}
                  className="cursor-pointer"
                >
                  <circle
                    r="12"
                    className="fill-white"
                    stroke={getScoreColor(node.fitness)}
                  />
                  <text
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="text-gray-500 text-lg"
                    fontSize="20"
                  >
                    +
                  </text>
                </g>

                {/* Remove node button (×) */}
                <g
                  transform={`translate(${node.x - nodeWidth/4}, ${node.y + nodeHeight/2 + 10})`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveNode(node.id);
                  }}
                  className="cursor-pointer"
                >
                  <circle
                    r="12"
                    className="fill-white stroke-red-500"
                  />
                  <text
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="fill-red-500 text-lg"
                    fontSize="20"
                  >
                    ×
                  </text>
                </g>

                {/* Forward connection button (→) */}
                <g
                  transform={`translate(${node.x}, ${node.y + nodeHeight/2 + 10})`}
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    setIsDrawingConnection(true);
                    setConnectionStart(node);
                    setTempConnectionEnd({ x: node.x, y: node.y });
                  }}
                  className="cursor-crosshair"
                >
                  <circle
                    r="12"
                    className="fill-white stroke-blue-500"
                  />
                  <text
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="fill-blue-500 text-lg"
                    fontSize="20"
                  >
                    →
                  </text>
                </g>
              </>
            )}
          </g>
        ))}

        {/* Add visual feedback for the temporary connection */}
        {isDrawingConnection && connectionStart && (
          <path
            d={`M ${tempConnectionEnd.x},${tempConnectionEnd.y} L ${connectionStart.x},${connectionStart.y}`}
            stroke={hoveredNode ? "#3b82f6" : "#94a3b8"} // Blue when over valid target
            strokeWidth="2"
            strokeDasharray="5,5"
            fill="none"
            markerEnd="url(#arrowhead)"
          />
        )}
      </svg>

      {/* Chat Box - Show history on hover */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1/2 bg-gradient-to-b from-white/80 to-gray-50/90 shadow-lg backdrop-blur-sm rounded-t-2xl group">
        {/* Chat history - hidden by default, shown on group hover */}
        <div className="max-h-40 overflow-y-auto p-3 space-y-2 hidden group-hover:block">
          {messages.map((msg, index) => (
            <div 
              key={index} 
              className="px-4 py-2 rounded-full bg-gradient-to-b from-white to-gray-50 text-sm text-gray-700 shadow-sm"
              style={{
                filter: 'drop-shadow(0 2px 3px rgba(107, 114, 128, 0.1))',
              }}
            >
              {msg}
            </div>
          ))}
        </div>
        <div className="p-3">
          <form 
            onSubmit={(e) => {
              e.preventDefault();
              if (newMessage.trim()) {
                setMessages(prev => [...prev, newMessage]);
                setNewMessage('');
              }
            }}
            className="flex gap-2 items-center"
          >
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder="Type a message..."
              className="flex-1 rounded-full px-6 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-slate-400"
              style={{
                background: 'linear-gradient(to right, #f8fafc, #f1f5f9, #f8fafc)',
                boxShadow: 'inset 0 1px 2px rgba(160, 174, 192, 0.2)',
                border: 'none',
              }}
            />
            <button
              type="submit"
              className="p-2 w-10 h-10 flex items-center justify-center text-slate-600 rounded-full text-sm font-medium transition-all hover:bg-slate-100 active:scale-95"
              style={{
                background: 'linear-gradient(135deg, #f8fafc, #e2e8f0)',
                boxShadow: '0 2px 4px rgba(148, 163, 184, 0.1)',
                border: 'none',
              }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
              </svg>
            </button>
          </form>
        </div>
      </div>

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
                  onChange={(e) => {
                    // Update local state
                    setEditingNode({...editingNode, name: e.target.value});
                    
                    // Send update to backend
                    handleNodeUpdate(editingNode.id, { name: e.target.value});
                  }}
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Target</label>
                <input
                  type="text"
                  value={editingNode.target}
                  onChange={(e) => {
                    setEditingNode({...editingNode, target: e.target.value});
                    handleNodeUpdate(editingNode.id, { target: e.target.value});
                  }}
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
                  onClick={(e) => {
                    e.preventDefault(); // Prevent any default form submission
                    console.log("Save button clicked"); // Debug log
                    
                    const nodeUpdate = {
                      id: editingNode.id,
                      name: editingNode.name,
                      target: editingNode.target,
                      code: editingNode.code,
                      reasoning: editingNode.reasoning,
                      input: editingNode.input,
                      output: editingNode.output,
                      inputTypes: editingNode.inputTypes,
                      outputTypes: editingNode.outputTypes,
                      x: editingNode.x,
                      y: editingNode.y,
                      fitness: editingNode.fitness
                    };
                    
                    // Send update to backend
                    handleNodeUpdate(editingNode.id, nodeUpdate);
                    
                    // Update local state
                    setNodes(prevNodes => 
                      prevNodes.map(node => 
                        node.id === editingNode.id ? {...node, ...nodeUpdate} : node
                      )
                    );
                    
                    // Close the edit form
                    setEditingNode(null);
                  }}
                  className="px-4 py-2 bg-blue-600 border border-transparent rounded-md text-sm font-medium text-white hover:bg-blue-700"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleDag;