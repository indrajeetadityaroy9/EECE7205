import React, { useState } from "react";

const DijkstraVisualization = () => {
  const [showShortestPaths, setShowShortestPaths] = useState(false);

  // Node positions
  const nodePositions = [
    { x: 300, y: 50 },   // Node 0
    { x: 500, y: 150 },  // Node 1
    { x: 400, y: 300 },  // Node 2
    { x: 200, y: 300 },  // Node 3
    { x: 100, y: 150 },  // Node 4
    { x: 200, y: 450 },  // Node 5
    { x: 400, y: 450 },  // Node 6
    { x: 500, y: 550 },  // Node 7
  ];

  const originalEdges = [
    { from: 0, to: 1, weight: 3 },
    { from: 0, to: 3, weight: 7 },
    { from: 1, to: 2, weight: 1 },
    { from: 1, to: 3, weight: 4 },
    { from: 2, to: 3, weight: 2 },
    { from: 2, to: 4, weight: 5 },
    { from: 3, to: 4, weight: 1 },
    { from: 4, to: 5, weight: 7 },
    { from: 4, to: 6, weight: 3 },
    { from: 5, to: 6, weight: 2 },
    { from: 6, to: 7, weight: 6 },
  ];

  const shortestPathEdges = [
    { from: 0, to: 1, weight: 3 },   // Shortest path from 0 to 1
    { from: 1, to: 2, weight: 1 },   // Shortest path from 1 to 2
    { from: 2, to: 3, weight: 2 },   // Shortest path from 2 to 3
    { from: 3, to: 4, weight: 1 },   // Shortest path from 3 to 4
    { from: 4, to: 6, weight: 3 },   // Shortest path from 4 to 6
    { from: 4, to: 5, weight: 7 },   // Shortest path from 4 to 5
    { from: 6, to: 7, weight: 6 },   // Shortest path from 6 to 7
  ];

  const edges = showShortestPaths ? shortestPathEdges : originalEdges;

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
          onClick={() => setShowShortestPaths(!showShortestPaths)}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          {showShortestPaths ? "Show Original Graph" : "Show Shortest Paths"}
        </button>
      </div>

      <div className="text-xl font-bold mb-4">
        {showShortestPaths ? "Shortest Paths (Dijkstra)" : "Original Graph"}
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
                stroke={showShortestPaths ? "#4CAF50" : "#666"}
                strokeWidth="2"
              />
              <circle
                cx={labelPos.x}
                cy={labelPos.y}
                r="12"
                fill="white"
                stroke={showShortestPaths ? "#4CAF50" : "#666"}
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
              fill={showShortestPaths ? "#4CAF50" : "#1E88E5"}
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
        <p>Nodes: {nodePositions.length}</p>
        <p>Original Edges: {originalEdges.length}</p>
        <p>Shortest Path Edges: {shortestPathEdges.length}</p>
        {showShortestPaths && (
          <p className="mt-2">
            Total Shortest Path Weight:{" "}
            {shortestPathEdges.reduce((sum, edge) => sum + edge.weight, 0)}
          </p>
        )}
      </div>
    </div>
  );
};

export default DijkstraVisualization;