import React, { useState } from 'react';

const PrimsVisualization = () => {
  const [showMST, setShowMST] = useState(false);

  const nodePositions = [
    { x: 300, y: 100 }, // Node 0
    { x: 500, y: 200 }, // Node 1
    { x: 400, y: 350 }, // Node 2
    { x: 200, y: 350 }, // Node 3
    { x: 100, y: 200 }, // Node 4
    { x: 300, y: 500 }, // Node 5
  ];
  
  // Original edges with weights
  const originalEdges = [
    { from: 0, to: 1, weight: 6 },
    { from: 0, to: 2, weight: 1 },
    { from: 0, to: 3, weight: 5 },
    { from: 1, to: 2, weight: 5 },
    { from: 1, to: 4, weight: 3 },
    { from: 2, to: 3, weight: 2 },
    { from: 2, to: 4, weight: 6 },
    { from: 2, to: 5, weight: 4 },
    { from: 3, to: 5, weight: 4 },
    { from: 4, to: 5, weight: 6 },
  ];
  
  // MST edges based on Prim's algorithm output
  const mstEdges = [
    { from: 0, to: 2, weight: 1 }, // Connecting node 0 to 2
    { from: 2, to: 3, weight: 2 }, // Connecting node 2 to 3
    { from: 2, to: 1, weight: 5 }, // Connecting node 2 to 1
    { from: 1, to: 4, weight: 3 }, // Connecting node 1 to 4
    { from: 2, to: 5, weight: 4 }, // Connecting node 2 to 5
  ];

  const edges = showMST ? mstEdges : originalEdges;

  // Function to calculate edge label position
  const getEdgeLabelPosition = (from, to) => {
    const x1 = nodePositions[from].x;
    const y1 = nodePositions[from].y;
    const x2 = nodePositions[to].x;
    const y2 = nodePositions[to].y;
    return {
      x: (x1 + x2) / 2,
      y: (y1 + y2) / 2,
    };
  };

  return (
    <div className="flex flex-col items-center w-full max-w-3xl mx-auto">
      <div className="mb-4">
        <button
          onClick={() => setShowMST(!showMST)}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          {showMST ? 'Show Original Graph' : 'Show MST'}
        </button>
      </div>

      <div className="text-xl font-bold mb-4">
        {showMST ? 'Minimum Spanning Tree (MST)' : 'Original Graph'}
      </div>

      <svg className="w-full h-[600px] border rounded bg-white">
        {/* Draw edges */}
        {edges.map((edge, idx) => {
          const start = nodePositions[edge.from];
          const end = nodePositions[edge.to];
          const labelPos = getEdgeLabelPosition(edge.from, edge.to);

          return (
            <g key={`edge-${idx}`}>
              <line
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke={showMST ? '#4CAF50' : '#666'}
                strokeWidth="2"
              />
              <circle
                cx={labelPos.x}
                cy={labelPos.y}
                r="12"
                fill="white"
                stroke={showMST ? '#4CAF50' : '#666'}
              />
              <text
                x={labelPos.x}
                y={labelPos.y}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="12"
              >
                {edge.weight}
              </text>
            </g>
          );
        })}

        {/* Draw nodes */}
        {nodePositions.map((pos, idx) => (
          <g key={`node-${idx}`}>
            <circle
              cx={pos.x}
              cy={pos.y}
              r="20"
              fill={showMST ? '#4CAF50' : '#1E88E5'}
              stroke="white"
              strokeWidth="2"
            />
            <text
              x={pos.x}
              y={pos.y}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="white"
              fontSize="14"
              fontWeight="bold"
            >
              {idx}
            </text>
          </g>
        ))}
      </svg>

      <div className="mt-4 p-4 bg-gray-100 rounded">
        <h3 className="font-bold mb-2">Graph Information:</h3>
        <p>Nodes: 6</p>
        <p>Original Edges: {originalEdges.length}</p>
        <p>MST Edges: {mstEdges.length}</p>
        {showMST && (
          <p className="mt-2">
            Total MST Weight: {mstEdges.reduce((sum, edge) => sum + edge.weight, 0)}
          </p>
        )}
      </div>
    </div>
  );
};

export default PrimsVisualization;