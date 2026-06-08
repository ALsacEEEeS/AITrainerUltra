import { useState } from 'react';
import { api } from '../../api/client';
import { DeviceSelector } from '../Device/DeviceSelector';

type FormatTab = 'huggingface' | 'local' | 'gguf' | 'lora';

const FORMAT_INFO = [
  { id: 'huggingface', label: 'HuggingFace', icon: '🤗', desc: '从HuggingFace Hub加载在线模型', ext: 'model ID' },
  { id: 'local', label: '本地文件', icon: '📁', desc: '从本地路径加载 .bin / .safetensors', ext: '.bin/.safetensors' },
  { id: 'gguf', label: 'GGUF格式', icon: '🔶', desc: 'llama.cpp量化模型', ext: '.gguf' },
  { id: 'lora', label: 'LoRA适配器', icon: '🔗', desc: '加载PEFT LoRA到基础模型', ext: 'adapter_config.json' },
];

const POPULAR_MODELS = [
  { id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', desc: '1.1B参数，轻量聊天模型', size: '~2.2GB' },
  { id: 'microsoft/phi-2', desc: '2.7B参数，高性价比', size: '~5GB' },
  { id: 'Qwen/Qwen2-1.5B-Instruct', desc: '1.5B参数，中文优化', size: '~3GB' },
  { id: 'mistralai/Mistral-7B-Instruct-v0.3', desc: '7B参数，通用强', size: '~14GB' },
  { id: 'google/gemma-2-2b', desc: '2B参数，Google出品', size: '~4GB' },
  { id: 'openai/clip-vit-base-patch32', desc: 'CLIP多模态模型', size: '~600MB' },
];

export function ModelLoaderPanel() {
  const [activeTab, setActiveTab] = useState<FormatTab>('huggingface');
  const [modelPath, setModelPath] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [detected, setDetected] = useState<any>(null);
  const [localFiles, setLocalFiles] = useState<any[]>([]);
  const [loadDevice, setLoadDevice] = useState('auto');

  const handleDetect = async () => {
    if (!modelPath.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.detectModel(modelPath.trim());
      setDetected(res.data);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const handleLoad = async () => {
    if (!modelPath.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.loadModelFromPath({
        path: modelPath.trim(),
        device: loadDevice,
        load_in_8bit: false,
        load_in_4bit: false,
      });
      setResult(res.data);
      setDetected(null);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const handleBrowse = async () => {
    setLoading(true);
    try {
      const res = await api.browseModels(modelPath.trim() || '.');
      setLocalFiles(res.data?.files || []);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  const handleQuickLoad = async (modelId: string) => {
    setModelPath(modelId);
    setLoading(true);
    setError(null);
    try {
      const res = await api.loadModelFromPath({ path: modelId, device: 'auto' });
      setResult(res.data);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="flex h-full">
      {/* Format sidebar */}
      <div className="w-64 bg-surface-900 border-r border-surface-700 flex flex-col">
        <div className="p-3 border-b border-surface-700">
          <h3 className="text-sm font-medium text-gray-300">模型来源</h3>
          <p className="text-xs text-gray-500 mt-0.5">支持多种开源格式</p>
        </div>

        <div className="p-2 space-y-1">
          {FORMAT_INFO.map((fmt) => (
            <button
              key={fmt.id}
              onClick={() => setActiveTab(fmt.id as FormatTab)}
              className={`w-full p-3 rounded-lg text-left transition-all ${
                activeTab === fmt.id
                  ? 'bg-primary-600/20 border border-primary-500/30'
                  : 'hover:bg-surface-800 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-lg">{fmt.icon}</span>
                <div>
                  <div className="text-sm text-white font-medium">{fmt.label}</div>
                  <div className="text-xs text-gray-500">{fmt.ext}</div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">{fmt.desc}</p>
            </button>
          ))}
        </div>

        <div className="p-3 border-t border-surface-700 mt-auto">
          <div className="text-[10px] text-gray-600">
            自动检测格式 · 无需手动指定
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'huggingface' && (
          <div className="space-y-4 max-w-2xl">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">从 HuggingFace Hub 加载</h3>
              <p className="text-xs text-gray-400 mb-3">
                输入 HuggingFace 模型 ID 即可自动下载并加载
              </p>
              <div className="flex gap-2">
                <input
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  placeholder="例如: TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                  className="input flex-1 text-sm"
                  onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
                />
                <button onClick={handleLoad} disabled={loading || !modelPath.trim()}
                  className="btn-primary text-sm">
                  {loading ? '加载中...' : '加载'}
                </button>
              </div>
              <div className="mt-3">
                <DeviceSelector compact defaultDevice={loadDevice} onDeviceChange={setLoadDevice} />
              </div>
            </div>

            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">快速加载</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-2">
                {POPULAR_MODELS.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => handleQuickLoad(m.id)}
                    className="text-left bg-surface-900 hover:bg-surface-800 rounded-lg p-3 border border-surface-700 transition-colors"
                  >
                    <div className="text-sm text-white font-medium truncate">{m.id}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{m.desc}</div>
                    <div className="text-xs text-primary-400 mt-0.5">{m.size}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'local' && (
          <div className="space-y-4 max-w-2xl">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">从本地路径加载</h3>
              <p className="text-xs text-gray-400 mb-3">
                支持 .bin / .safetensors / .pt 格式及其目录，自动识别架构
              </p>
              <div className="flex gap-2">
                <input
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  placeholder="例如: ./models/my-model 或 C:/models/llama.gguf"
                  className="input flex-1 text-sm"
                />
                <button onClick={handleDetect} disabled={!modelPath.trim()}
                  className="btn-secondary text-sm">
                  检测
                </button>
                <button onClick={handleLoad} disabled={loading || !modelPath.trim()}
                  className="btn-primary text-sm">
                  {loading ? '加载中...' : '加载'}
                </button>
              </div>
            </div>

            {detected && (
              <div className="card border-primary-500/30">
                <h4 className="text-sm text-primary-300 mb-2">检测结果</h4>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div><span className="text-gray-500">格式:</span> {detected.format}</div>
                  <div><span className="text-gray-500">架构:</span> {detected.architecture || '未知'}</div>
                  <div><span className="text-gray-500">文件大小:</span> {detected.file_info?.size_readable}</div>
                  <div><span className="text-gray-500">文件数:</span> {detected.file_info?.num_files}</div>
                </div>
              </div>
            )}

            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-300">浏览本地文件</h3>
                <button onClick={handleBrowse} disabled={loading}
                  className="btn-secondary text-xs py-1 px-2">
                  刷新
                </button>
              </div>
              {localFiles.length > 0 ? (
                <div className="space-y-1 max-h-60 overflow-y-auto">
                  {localFiles.slice(0, 50).map((f: any) => (
                    <div key={f.path}
                      onClick={() => setModelPath(f.path)}
                      className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer text-xs transition-colors ${
                        modelPath === f.path ? 'bg-primary-600/20' : 'hover:bg-surface-800'
                      }`}
                    >
                      <span>{f.is_directory ? '📁' : f.format === 'gguf' ? '🔶' : f.format === 'safetensors' ? '🔒' : '📄'}</span>
                      <span className="text-gray-200 flex-1 truncate">{f.name}</span>
                      <span className="text-gray-500">{f.size_readable}</span>
                      {f.architecture && <span className="badge bg-surface-700 text-gray-300 text-[10px]">{f.architecture}</span>}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-500 text-sm py-4">
                  输入路径后点击"检测"或"刷新"
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'gguf' && (
          <div className="space-y-4 max-w-2xl">
            <div className="card">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-lg">🔶</span>
                <h3 className="text-sm font-medium text-gray-300">GGUF 模型加载</h3>
              </div>
              <p className="text-xs text-gray-400 mb-3">
                GGUF 是 llama.cpp 的量化格式，适合 CPU 推理。
                需要安装 llama-cpp-python: <code className="text-primary-400">pip install llama-cpp-python</code>
              </p>
              <div className="flex gap-2">
                <input
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  placeholder="例如: ./models/llama-2-7b.Q4_K_M.gguf"
                  className="input flex-1 text-sm"
                />
                <button onClick={handleLoad} disabled={loading || !modelPath.trim()}
                  className="btn-primary text-sm">
                  {loading ? '加载中...' : '加载'}
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'lora' && (
          <div className="space-y-4 max-w-2xl">
            <div className="card">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-lg">🔗</span>
                <h3 className="text-sm font-medium text-gray-300">LoRA 适配器加载</h3>
              </div>
              <p className="text-xs text-gray-400 mb-3">
                加载 PEFT LoRA 适配器到基础模型上，需要 adapter_config.json 和 adapter_model.safetensors
              </p>
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-gray-400 block mb-1">基础模型路径</label>
                  <input
                    value={modelPath}
                    onChange={(e) => setModelPath(e.target.value)}
                    placeholder="基础模型 ID 或路径"
                    className="input w-full text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">适配器路径</label>
                  <input placeholder="LoRA adapter 目录路径" className="input w-full text-sm" />
                </div>
                <button className="btn-primary text-sm w-full">
                  加载适配器
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Results display */}
        {result && (
          <div className="card bg-green-600/10 border-green-500/30 mt-4 max-w-2xl">
            <h3 className="text-sm font-medium text-green-400 mb-2">模型已加载</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div><span className="text-gray-500">模型ID:</span> <span className="text-gray-200">{result.model_id}</span></div>
              <div><span className="text-gray-500">格式:</span> <span className="text-gray-200">{result.format}</span></div>
              <div><span className="text-gray-500">架构:</span> <span className="text-gray-200">{result.architecture || '自动'}</span></div>
              {result.params && <div><span className="text-gray-500">参数量:</span> <span className="text-gray-200">{(result.params / 1e6).toFixed(1)}M</span></div>}
              <div className="col-span-2 mt-2">
                <span className="text-primary-400">现在可以到"对话交互"面板与该模型对话</span>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="card bg-red-600/10 border-red-500/30 mt-4 max-w-2xl">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
