import { create } from 'zustand';

export interface WorkflowNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  config: Record<string, any>;
}

export interface WorkflowConnection {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

interface WorkflowState {
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  isRunning: boolean;
  addNode: (node: WorkflowNode) => void;
  removeNode: (id: string) => void;
  updateNodeConfig: (id: string, config: Record<string, any>) => void;
  updateNodePosition: (id: string, position: { x: number; y: number }) => void;
  addConnection: (conn: WorkflowConnection) => void;
  removeConnection: (id: string) => void;
  setNodes: (nodes: WorkflowNode[]) => void;
  setConnections: (connections: WorkflowConnection[]) => void;
  setIsRunning: (running: boolean) => void;
  clear: () => void;
}

export const useWorkflowStore = create<WorkflowState>((set) => ({
  nodes: [],
  connections: [],
  isRunning: false,

  addNode: (node) =>
    set((state) => ({ nodes: [...state.nodes, node] })),

  removeNode: (id) =>
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      connections: state.connections.filter(
        (c) => c.source !== id && c.target !== id,
      ),
    })),

  updateNodeConfig: (id, config) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, config: { ...n.config, ...config } } : n,
      ),
    })),

  updateNodePosition: (id, position) =>
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, position } : n,
      ),
    })),

  addConnection: (conn) =>
    set((state) => {
      // Prevent duplicate connections
      const exists = state.connections.some(
        (c) => c.source === conn.source && c.target === conn.target,
      );
      if (exists) return state;
      return { connections: [...state.connections, conn] };
    }),

  removeConnection: (id) =>
    set((state) => ({
      connections: state.connections.filter((c) => c.id !== id),
    })),

  setNodes: (nodes) => set({ nodes }),
  setConnections: (connections) => set({ connections }),
  setIsRunning: (running) => set({ isRunning: running }),
  clear: () => set({ nodes: [], connections: [], isRunning: false }),
}));
