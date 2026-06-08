import { create } from 'zustand';

interface ModelInfo {
  name: string;
  type: string;
  description: string;
  loaded: boolean;
}

interface ModelState {
  models: ModelInfo[];
  loadedModel: string | null;
  presets: Record<string, any>;
  setModels: (models: ModelInfo[]) => void;
  setLoadedModel: (name: string | null) => void;
  setPresets: (presets: Record<string, any>) => void;
  addModel: (model: ModelInfo) => void;
  removeModel: (name: string) => void;
}

export const useModelStore = create<ModelState>((set) => ({
  models: [
    { name: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', type: 'llm', description: '轻量级聊天模型', loaded: false },
    { name: 'SimianLuo/LCM_Dreamshaper_v7', type: 'lcm', description: 'LCM图像生成', loaded: false },
    { name: 'microsoft/resnet-50', type: 'cnn', description: 'ResNet图像分类', loaded: false },
  ],
  loadedModel: null,
  presets: {},
  setModels: (models) => set({ models }),
  setLoadedModel: (name) => set({ loadedModel: name }),
  setPresets: (presets) => set({ presets }),
  addModel: (model) => set((state) => ({ models: [...state.models, model] })),
  removeModel: (name) =>
    set((state) => ({
      models: state.models.filter((m) => m.name !== name),
    })),
}));
