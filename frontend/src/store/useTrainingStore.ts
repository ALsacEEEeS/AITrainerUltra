import { create } from 'zustand';

interface TrainingState {
  isRunning: boolean;
  jobId: string | null;
  modelType: string | null;
  config: any;
  metrics: Record<string, number>;
  history: { loss: number; step: number }[];
  setRunning: (running: boolean) => void;
  setJobId: (id: string | null) => void;
  setModelType: (type: string | null) => void;
  setConfig: (config: any) => void;
  updateMetrics: (metrics: Record<string, number>) => void;
  addHistory: (point: { loss: number; step: number }) => void;
  reset: () => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  isRunning: false,
  jobId: null,
  modelType: null,
  config: {
    model_type: 'llm',
    model_name: '',
    hyperparameters: {
      learning_rate: 5e-5,
      batch_size: 8,
      num_epochs: 3,
    },
  },
  metrics: {},
  history: [],
  setRunning: (running) => set({ isRunning: running }),
  setJobId: (id) => set({ jobId: id }),
  setModelType: (type) => set({ modelType: type }),
  setConfig: (config) => set({ config }),
  updateMetrics: (metrics) => set({ metrics }),
  addHistory: (point) =>
    set((state) => ({ history: [...state.history, point] })),
  reset: () =>
    set({
      isRunning: false,
      jobId: null,
      modelType: null,
      metrics: {},
      history: [],
    }),
}));
