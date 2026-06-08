import { useMemo } from 'react';

interface Connection {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

interface ConnectionLineProps {
  connections: Connection[];
  nodePositions: Record<string, { x: number; y: number }>;
  nodeDimensions: Record<string, { width: number; height: number }>;
  selected: string | null;
  onSelect: (id: string | null) => void;
  onDelete: (id: string) => void;
  tempConnection: { sx: number; sy: number; ex: number; ey: number } | null;
}

/**
 * Calculate port positions relative to the canvas.
 * Input ports are on the left side, output ports on the right.
 * Each port's Y position is offset based on its index among all ports of that type.
 */
function getPortPos(
  nodeId: string,
  portType: 'input' | 'output',
  portIndex: number,
  portsCount: number,
  positions: Record<string, { x: number; y: number }>,
  dimensions: Record<string, { width: number; height: number }>,
): { x: number; y: number } {
  const pos = positions[nodeId] || { x: 0, y: 0 };
  const dim = dimensions[nodeId] || { width: 288, height: 200 };

  const headerHeight = 48;
  const portsTop = pos.y + headerHeight + 12;
  const portSpacing = 24;
  const portY = portsTop + portIndex * portSpacing + (portsCount > 1 ? 0 : 6);

  if (portType === 'input') {
    return { x: pos.x - 8, y: portY };
  } else {
    return { x: pos.x + dim.width + 8, y: portY };
  }
}

export function ConnectionLine({
  connections,
  nodePositions,
  nodeDimensions,
  selected,
  onSelect,
  onDelete,
  tempConnection,
}: ConnectionLineProps) {
  // Compute SVG path for each connection
  const paths = useMemo(() => {
    if (connections.length === 0) return [];
    return connections.map((conn) => {
      const sourceNode = conn.source;
      const targetNode = conn.target;
      const sPos = nodePositions[sourceNode] || { x: 0, y: 0 };
      const tPos = nodePositions[targetNode] || { x: 0, y: 0 };
      const sDim = nodeDimensions[sourceNode] || { width: 288, height: 200 };
      const tDim = nodeDimensions[targetNode] || { width: 288, height: 200 };

      // Output port is on the right side of source node
      const startX = sPos.x + sDim.width;
      const startY = sPos.y + sDim.height / 2;

      // Input port is on the left side of target node
      const endX = tPos.x;
      const endY = tPos.y + tDim.height / 2;

      // Bezier control points for a smooth curve
      const dx = Math.abs(endX - startX) * 0.5;
      const cp1x = startX + dx;
      const cp1y = startY;
      const cp2x = endX - dx;
      const cp2y = endY;

      const d = `M ${startX} ${startY} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${endX} ${endY}`;
      const isSelected = selected === conn.id;

      return {
        id: conn.id,
        d,
        isSelected,
        sourceId: sourceNode,
        targetId: targetNode,
      };
    });
  }, [connections, nodePositions, nodeDimensions, selected]);

  // Temp connection path (while dragging)
  const tempPath = useMemo(() => {
    if (!tempConnection) return null;
    const { sx, sy, ex, ey } = tempConnection;
    const dx = Math.abs(ex - sx) * 0.5;
    return {
      d: `M ${sx} ${sy} C ${sx + dx} ${sy}, ${ex - dx} ${ey}, ${ex} ${ey}`,
    };
  }, [tempConnection]);

  if (connections.length === 0 && !tempConnection) {
    return (
      <svg className="absolute inset-0 pointer-events-none">
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
          </marker>
        </defs>
      </svg>
    );
  }

  return (
    <svg className="absolute inset-0 pointer-events-none" style={{ overflow: 'visible' }}>
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
        </marker>
        <marker id="arrowhead-selected" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#a78bfa" />
        </marker>
      </defs>

      {/* Existing connections */}
      {paths.map((p) => (
        <g key={p.id}>
          {/* Hit area (wider invisible path for easier clicking) */}
          <path
            d={p.d}
            fill="none"
            stroke="transparent"
            strokeWidth={12}
            className="pointer-events-auto cursor-pointer"
            onClick={(e) => { e.stopPropagation(); onSelect(p.id); }}
            onContextMenu={(e) => { e.preventDefault(); onDelete(p.id); }}
          />
          {/* Visible path */}
          <path
            d={p.d}
            fill="none"
            stroke={p.isSelected ? '#a78bfa' : '#6366f1'}
            strokeWidth={p.isSelected ? 3 : 2}
            strokeOpacity={p.isSelected ? 1 : 0.6}
            markerEnd={`url(#${p.isSelected ? 'arrowhead-selected' : 'arrowhead'})`}
            className="pointer-events-auto cursor-pointer transition-all"
            onClick={(e) => { e.stopPropagation(); onSelect(p.id); }}
            onContextMenu={(e) => { e.preventDefault(); onDelete(p.id); }}
          />
        </g>
      ))}

      {/* Temp connection while dragging */}
      {tempPath && (
        <path
          d={tempPath.d}
          fill="none"
          stroke="#818cf8"
          strokeWidth={2}
          strokeDasharray="6 3"
          strokeOpacity={0.7}
        />
      )}
    </svg>
  );
}
