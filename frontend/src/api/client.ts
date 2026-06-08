const BASE_URL = '/api/v1';

// Simple TTL cache for GET requests
const cache = new Map<string, { data: any; expiry: number }>();
const CACHE_TTL: Record<string, number> = {
  '/device': 300_000,        // 5 min
  '/models': 60_000,         // 1 min
  '/presets': 60_000,        // 1 min
  '/recipes': 300_000,       // 5 min
  '/dataset-presets': 300_000, // 5 min
  '/templates': 300_000,     // 5 min
  '/optimizations': 30_000,  // 30 sec
  '/training/status': 5_000, // 5 sec
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const isGet = !options || options.method === undefined || options.method === 'GET';

  // Check cache for GET requests
  if (isGet) {
    const cached = cache.get(path);
    if (cached && cached.expiry > Date.now()) {
      return cached.data as T;
    }
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Request failed' }));
    throw new Error(error.detail || error.message || `HTTP ${response.status}`);
  }
  const data = await response.json();

  // Cache GET responses
  if (isGet) {
    const ttl = CACHE_TTL[path] || 0;
    if (ttl > 0) {
      cache.set(path, { data, expiry: Date.now() + ttl });
    }
  }

  return data;
}

export const api = {
  // System
  getStatus: () => request<any>('/status'),
  listModels: () => request<any>('/models'),
  listPresets: () => request<any>('/presets'),

  // Training
  startTraining: (config: any) =>
    request<any>('/training/start', { method: 'POST', body: JSON.stringify(config) }),
  stopTraining: () =>
    request<any>('/training/stop', { method: 'POST' }),
  getTrainingStatus: () => request<any>('/training/status'),
  validateTrainingConfig: (config: any) =>
    request<any>('/training/validate', { method: 'POST', body: JSON.stringify(config) }),
  listCheckpoints: (outputDir = './output') =>
    request<any>(`/training/checkpoints?output_dir=${outputDir}`),
  deleteCheckpoint: (name: string, outputDir = './output') =>
    request<any>('/training/checkpoints/delete', {
      method: 'POST', body: JSON.stringify({ name, output_dir: outputDir }),
    }),
  restoreCheckpoint: (name: string, outputDir = './output') =>
    request<any>('/training/checkpoints/restore', {
      method: 'POST', body: JSON.stringify({ name, output_dir: outputDir }),
    }),

  // Chat
  chat: (payload: any) =>
    request<any>('/chat', { method: 'POST', body: JSON.stringify(payload) }),

  // Streaming chat (SSE)
  chatStream: (
    payload: any,
    onToken: (token: string, fullText: string) => void,
    onDone: (fullText: string) => void,
    onError: (error: string) => void,
  ) => {
    const controller = new AbortController();
    fetch('/api/v1/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    }).then(async (response) => {
      if (!response.ok) {
        onError(`HTTP ${response.status}`);
        return;
      }
      const reader = response.body?.getReader();
      if (!reader) {
        onError('No response body');
        return;
      }
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.error) {
                onError(data.full_text);
                return;
              }
              if (data.done) {
                onDone(data.full_text);
                return;
              }
              onToken(data.token, data.full_text);
            } catch { /* skip malformed */ }
          }
        }
      }
    }).catch((err) => {
      if (err.name !== 'AbortError') onError(err.message);
    });

    return controller; // Caller can call .abort() to stop
  },

  // Workflow
  runWorkflow: (workflow: any, signal?: AbortSignal) =>
    request<any>('/workflow/run', { method: 'POST', body: JSON.stringify(workflow), signal }),

  // 🔬 Experiments
  listExperiments: (status = '') =>
    request<any>(`/experiments?status=${status}`),
  createExperiment: (data: any) =>
    request<any>('/experiments/create', { method: 'POST', body: JSON.stringify(data) }),
  getExperiment: (name: string) =>
    request<any>(`/experiments/${name}`),
  deleteExperiment: (name: string) =>
    request<any>(`/experiments/${name}`, { method: 'DELETE' }),

  // 🗂️ Model Store
  listStoredModels: () => request<any>('/model-store'),
  saveToStore: (data: any) =>
    request<any>('/model-store/save', { method: 'POST', body: JSON.stringify(data) }),
  exportModel: (data: any) =>
    request<any>('/model-store/export', { method: 'POST', body: JSON.stringify(data) }),

  // 📊 Datasets
  listDatasets: () => request<any>('/datasets'),
  getDatasetStats: (name: string) => request<any>(`/datasets/${name}/stats`),
  previewDataset: (data: any) =>
    request<any>('/datasets/preview', { method: 'POST', body: JSON.stringify(data) }),

  // 📋 Templates
  listTemplates: () => request<any>('/templates'),
  getTemplate: (name: string) => request<any>(`/templates/${encodeURIComponent(name)}`),

  // 🎛️ HPO
  gridSearch: (data: any) =>
    request<any>('/hpo/grid', { method: 'POST', body: JSON.stringify(data) }),
  randomSearch: (data: any) =>
    request<any>('/hpo/random', { method: 'POST', body: JSON.stringify(data) }),

  // ⚡ Inference
  listInferenceModels: () => request<any>('/inference/models'),
  generateInference: (data: any) =>
    request<any>('/inference/generate', { method: 'POST', body: JSON.stringify(data) }),

  // 🖥️ Device
  getDevice: () => request<any>('/device'),

  // 📂 Model Loader
  getModelLoaderInfo: () => request<any>('/model-loader/info'),
  detectModel: (path: string) =>
    request<any>('/model-loader/detect', { method: 'POST', body: JSON.stringify({ path }) }),
  browseModels: (path: string) =>
    request<any>('/model-loader/browse', { method: 'POST', body: JSON.stringify({ path }) }),
  loadModelFromPath: (data: any) =>
    request<any>('/model-loader/load', { method: 'POST', body: JSON.stringify(data) }),
  loadLoraAdapter: (baseModel: string, adapterPath: string) =>
    request<any>('/model-loader/load-lora', {
      method: 'POST', body: JSON.stringify({ base_model: baseModel, adapter_path: adapterPath }),
    }),

  // 🖼️ Multimodal
  multimodalInfer: (data: any) =>
    request<any>('/multimodal/infer', { method: 'POST', body: JSON.stringify(data) }),

  // ⚡ Optimizations (WOQ / KV Offload / DMS / Variable VRAM)
  listOptimizations: () => request<any>('/optimizations'),
  applyOptimizations: (data: any) =>
    request<any>('/optimizations/apply', { method: 'POST', body: JSON.stringify(data) }),
  getOptimizationStatus: () => request<any>('/optimizations/status'),
  reportTrainStep: (data: any) =>
    request<any>('/optimizations/train-step', { method: 'POST', body: JSON.stringify(data) }),

  // 📋 Recipes
  listRecipes: (tag = '', hardware = '') =>
    request<any>(`/recipes?tag=${tag}&hardware=${hardware}`),
  getRecipe: (name: string) =>
    request<any>(`/recipes/${encodeURIComponent(name)}`),
  listDatasetPresets: () => request<any>('/dataset-presets'),

  // 📋 Jobs
  listJobs: () => request<any>('/jobs'),
  enqueueJob: (data: any) =>
    request<any>('/jobs/enqueue', { method: 'POST', body: JSON.stringify(data) }),
  getJob: (jobId: string) => request<any>(`/jobs/${jobId}`),
  cancelJob: (jobId: string) =>
    request<any>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
};
