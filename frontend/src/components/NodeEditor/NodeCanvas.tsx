import { useState, useCallback, useRef, useEffect } from 'react';
import { NodePalette } from './NodePalette';
import { NodeComponent, PortEvent } from './NodeComponent';
import { ConnectionLine } from './ConnectionLine';
import { NODE_DEFINITIONS } from './types';
import { useWorkflowStore, WorkflowNode } from '../../store/useWorkflowStore';
import { api } from '../../api/client';

/**
 * NodeCanvas — 工作流编辑器主画布
 *
 * 支持:
 * - 拖拽节点到画布
 * - 端口拖动连线 (output → input)
 * - 拖动画布平移
 * - 选中/删除节点和连线
 * - 运行工作流
 */
export function NodeCanvas() {
  const {
    nodes, connections, addNode, removeNode,
    updateNodeConfig, updateNodePosition, setNodes,
    addConnection, removeConnection,
    isRunning, setIsRunning,
  } = useWorkflowStore();

  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedConnection, setSelectedConnection] = useState<string | null>(null);
  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [nodeDragStart, setNodeDragStart] = useState({ x: 0, y: 0 });

  // ─── Connection drag state ─────────────────────────────────
  const [tempConnection, setTempConnection] = useState<{
    sx: number; sy: number; ex: number; ey: number;
    sourceNode: string; sourcePort: string;
  } | null>(null);

  const canvasRef = useRef<HTMLDivElement>(null);
  // Track node pixel positions on screen (for connection drawing)
  const nodeScreenPositions = useRef<Record<string, { x: number; y: number }>>({});
  const nodeDimensions = useRef<Record<string, { width: number; height: number }>>({});

  // Update node positions for connection line drawing
  const updateNodeScreenPos = useCallback((id: string) => {
    const el = canvasRef.current?.querySelector(`[data-node-id="${id}"]`) as HTMLElement;
    if (el) {
      const rect = el.getBoundingClientRect();
      const canvasRect = canvasRef.current?.getBoundingClientRect();
      if (canvasRect) {
        nodeScreenPositions.current[id] = {
          x: rect.left - canvasRect.left,
          y: rect.top - canvasRect.top,
        };
        nodeDimensions.current[id] = {
          width: rect.width,
          height: rect.height,
        };
      }
    }
  }, []);

  // Recalculate all node positions
  const recalcNodePositions = useCallback(() => {
    nodes.forEach((n) => updateNodeScreenPos(n.id));
  }, [nodes, updateNodeScreenPos]);

  useEffect(() => {
    // Small delay to ensure DOM is rendered
    const timer = setTimeout(recalcNodePositions, 50);
    return () => clearTimeout(timer);
  }, [nodes, recalcNodePositions]);

  // ─── Node drag & drop ──────────────────────────────────────
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const type = e.dataTransfer.getData('text/plain');
    const def = NODE_DEFINITIONS.find((d) => d.type === type);
    if (!def) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left - 144 + canvasOffset.x;
    const y = e.clientY - rect.top - 40 + canvasOffset.y;

    const nodeId = `${type}_${Date.now()}`;
    const defaultConfig: Record<string, any> = {};
    def.configFields.forEach((f) => {
      defaultConfig[f.name] = f.defaultValue;
    });

    addNode({
      id: nodeId,
      type,
      position: { x, y },
      config: defaultConfig,
    });

    // Deselect connection when dropping
    setSelectedConnection(null);
  }, [canvasOffset, addNode]);

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    // Only start canvas drag if clicking on canvas background
    if (target.dataset?.canvas) {
      setDragging(true);
      setDragStart({ x: e.clientX - canvasOffset.x, y: e.clientY - canvasOffset.y });
    }
    // Deselect on canvas click
    if (!target.closest('.node-component')) {
      setSelectedNode(null);
      setSelectedConnection(null);
    }
  };

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragging) {
      setCanvasOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
    if (draggingNode) {
      const dx = e.clientX - nodeDragStart.x;
      const dy = e.clientY - nodeDragStart.y;
      const node = nodes.find((n) => n.id === draggingNode);
      if (node) {
        updateNodePosition(draggingNode, {
          x: node.position.x + dx,
          y: node.position.y + dy,
        });
        setNodeDragStart({ x: e.clientX, y: e.clientY });
      }
    }
    // Update temp connection endpoint
    if (tempConnection) {
      const canvasRect = canvasRef.current?.getBoundingClientRect();
      if (canvasRect) {
        setTempConnection((prev) => prev ? {
          ...prev,
          ex: e.clientX - canvasRect.left + canvasOffset.x,
          ey: e.clientY - canvasRect.top + canvasOffset.y,
        } : null);
      }
    }
  }, [dragging, dragStart, draggingNode, nodeDragStart, nodes, updateNodePosition, tempConnection, canvasOffset]);

  const handleCanvasMouseUp = useCallback(() => {
    setDragging(false);
    setDraggingNode(null);
    // Finalize temp connection
    if (tempConnection) {
      setTempConnection(null);
    }
  }, [tempConnection]);

  // ─── Port connection handling ──────────────────────────────
  const handlePortMouseDown = useCallback((portEvent: PortEvent, e: React.MouseEvent) => {
    if (portEvent.portType === 'input') return; // Only start from outputs

    const canvasRect = canvasRef.current?.getBoundingClientRect();
    if (!canvasRect) return;

    setTempConnection({
      sx: portEvent.rect.left - canvasRect.left + canvasOffset.x,
      sy: portEvent.rect.top - canvasRect.top + canvasOffset.y + 6,
      ex: portEvent.rect.left - canvasRect.left + canvasOffset.x,
      ey: portEvent.rect.top - canvasRect.top + canvasOffset.y + 6,
      sourceNode: portEvent.nodeId,
      sourcePort: portEvent.portId,
    });
  }, [canvasOffset]);

  const handlePortMouseUp = useCallback((portEvent: PortEvent, e: React.MouseEvent) => {
    if (!tempConnection) return;
    if (portEvent.portType === 'output') return; // Only connect to inputs
    if (portEvent.nodeId === tempConnection.sourceNode) return; // No self-connect

    const connId = `${tempConnection.sourceNode}→${portEvent.nodeId}`;
    addConnection({
      id: connId,
      source: tempConnection.sourceNode,
      target: portEvent.nodeId,
      sourceHandle: tempConnection.sourcePort,
      targetHandle: portEvent.portId,
    });

    setTempConnection(null);
  }, [tempConnection, addConnection]);

  // ─── Node dragging ─────────────────────────────────────────
  const handleNodeMouseDown = useCallback((nodeId: string, e: React.MouseEvent) => {
    if (e.target instanceof HTMLElement && e.target.closest('.port-handle')) return;
    setDraggingNode(nodeId);
    setNodeDragStart({ x: e.clientX, y: e.clientY });
    setSelectedNode(nodeId);
    setSelectedConnection(null);
  }, []);

  // ─── Workflow execution ────────────────────────────────────
  const workflowAbortRef = useRef<AbortController | null>(null);

  const handleRun = async () => {
    setIsRunning(true);
    const controller = new AbortController();
    workflowAbortRef.current = controller;
    try {
      const payload = {
        nodes: nodes.map((n) => ({
          id: n.id,
          type: n.type,
          position: n.position,
          config: n.config,
        })),
        connections: connections.map((c) => ({
          id: c.id,
          source: c.source,
          target: c.target,
          source_handle: c.sourceHandle,
          target_handle: c.targetHandle,
        })),
      };
      const result = await api.runWorkflow(payload, controller.signal);
      console.log('Workflow result:', result);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        console.log('Workflow aborted');
      } else {
        console.error('Workflow error:', err);
      }
    } finally {
      setIsRunning(false);
      workflowAbortRef.current = null;
    }
  };

  const handleStop = () => {
    if (workflowAbortRef.current) {
      workflowAbortRef.current.abort();
      setIsRunning(false);
      workflowAbortRef.current = null;
    }
  };

  return (
    <div className="flex h-full" onMouseUp={() => setTempConnection(null)}>
      <NodePalette onDragStart={() => {}} />

      <div
        ref={canvasRef}
        className="flex-1 relative bg-surface-950 overflow-hidden"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={handleCanvasMouseUp}
      >
        {/* Grid background */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              'linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)',
            backgroundSize: '20px 20px',
            transform: `translate(${-canvasOffset.x % 20}px, ${-canvasOffset.y % 20}px)`,
          }}
        />

        {/* Connection layer */}
        <ConnectionLine
          connections={connections}
          nodePositions={(() => {
            const pos: Record<string, { x: number; y: number }> = {};
            nodes.forEach((n) => {
              pos[n.id] = { x: n.position.x - canvasOffset.x, y: n.position.y - canvasOffset.y };
            });
            return pos;
          })()}
          nodeDimensions={nodeDimensions.current}
          selected={selectedConnection}
          onSelect={setSelectedConnection}
          onDelete={(id) => { removeConnection(id); setSelectedConnection(null); }}
          tempConnection={tempConnection ? {
            sx: tempConnection.sx - canvasOffset.x,
            sy: tempConnection.sy - canvasOffset.y,
            ex: tempConnection.ex - canvasOffset.x,
            ey: tempConnection.ey - canvasOffset.y,
          } : null}
        />

        {/* Nodes layer */}
        <div
          data-canvas
          className="absolute inset-0"
          style={{ cursor: dragging ? 'grabbing' : 'grab' }}
        >
          {nodes.map((node) => {
            const nodeX = node.position.x - canvasOffset.x;
            const nodeY = node.position.y - canvasOffset.y;

            return (
              <div
                key={node.id}
                data-node-id={node.id}
                className="node-component"
                style={{
                  transform: `translate(${nodeX}px, ${nodeY}px)`,
                  position: 'absolute',
                  left: 0,
                  top: 0,
                }}
                onMouseDown={(e) => handleNodeMouseDown(node.id, e)}
              >
                <NodeComponent
                  id={node.id}
                  type={node.type}
                  selected={selectedNode === node.id}
                  config={node.config}
                  onConfigChange={updateNodeConfig}
                  onSelect={(id) => { setSelectedNode(id); setSelectedConnection(null); }}
                  onDelete={removeNode}
                  onPortMouseDown={handlePortMouseDown}
                />
              </div>
            );
          })}
        </div>

        {/* Port mouse up handler (global) */}
        <div
          className="absolute inset-0"
          onMouseUp={(e) => {
            // Check if we released over an input port
            const target = e.target as HTMLElement;
            const portEl = target.closest('[data-port-type="input"]');
            // Since we handle this per-port via the node component,
            // just clear temp if clicking on empty space
            if (!portEl && tempConnection) {
              setTempConnection(null);
            }
          }}
          style={{ pointerEvents: 'none' }}
        />

        {/* Empty state */}
        {nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              <div className="text-5xl mb-3">⚡</div>
              <p className="text-gray-500 text-lg">从左侧拖拽节点到此处</p>
              <p className="text-gray-600 text-sm mt-1">点击端口拖拽创建连线</p>
              <p className="text-gray-700 text-xs mt-2">支持 20+ 节点类型 · 自定义节点 · 工作流编排</p>
            </div>
          </div>
        )}

        {/* Bottom toolbar */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2 z-20">
          <button
            onClick={handleRun}
            disabled={nodes.length === 0 || isRunning}
            className="btn-primary flex items-center gap-2"
          >
            {isRunning ? (
              <span className="animate-spin">⏳</span>
            ) : (
              <span>▶</span>
            )}
            {isRunning ? '执行中...' : '运行工作流'}
          </button>
          {isRunning && (
            <button
              onClick={handleStop}
              className="btn-danger flex items-center gap-2 animate-pulse"
            >
              <span>⏹</span>
              停止
            </button>
          )}
          <button
            onClick={() => { setNodes([]); setSelectedNode(null); setSelectedConnection(null); }}
            className="btn-secondary text-sm"
            disabled={isRunning}
          >
            清空
          </button>
          {selectedConnection && (
            <button
              onClick={() => { removeConnection(selectedConnection); setSelectedConnection(null); }}
              className="btn-danger text-sm"
            >
              删除连线
            </button>
          )}
        </div>

        {/* Connection hint */}
        {tempConnection && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-primary-500/20 text-primary-300 text-xs px-3 py-1 rounded-full z-20">
            拖到目标输入端口创建连接 · 松手取消
          </div>
        )}
      </div>
    </div>
  );
}
