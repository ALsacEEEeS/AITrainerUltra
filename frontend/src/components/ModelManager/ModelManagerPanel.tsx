import { useState, useEffect } from 'react';
import { api } from '../../api/client';

export function ModelManagerPanel() {
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showImport, setShowImport] = useState(false);
  const [showExport, setShowExport] = useState<string | null>(null);
  const [form, setForm] = useState({ name: '', model_type: 'llm', source_path: '', description: '' });
  const [exportForm, setExportForm] = useState({ format: 'pytorch', output_path: './exports' });

  const fetchModels = async () => {
    setLoading(true);
    try {
      const res = await api.listStoredModels();
      setModels(res.data?.models || []);
    } catch (err) { console.error(err); }
    setLoading(false);
  };

  useEffect(() => { fetchModels(); }, []);

  const handleImport = async () => {
    await api.saveToStore(form);
    setShowImport(false);
    setForm({ name: '', model_type: 'llm', source_path: '', description: '' });
    fetchModels();
  };

  const handleExport = async (name: string) => {
    await api.exportModel({ name, ...exportForm });
    setShowExport(null);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  const getTypeIcon = (type: string) => {
    const icons: Record<string, string> = { llm: '🤖', lcm: '🎨', cnn: '🖼️', lora: '🔗', qlora: '⚡' };
    return icons[type] || '📦';
  };

  return (
    <div className="flex h-full">
      <div className="w-80 bg-surface-900 border-r border-surface-700 flex flex-col">
        <div className="p-3 border-b border-surface-700 flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-300">模型仓库</h3>
          <button onClick={() => setShowImport(true)} className="btn-primary text-xs py-1 px-2">
            + 导入
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="p-4 text-center text-gray-500 text-sm">加载中...</div>
          ) : models.length === 0 ? (
            <div className="p-4 text-center">
              <div className="text-2xl mb-2">🗂️</div>
              <p className="text-gray-500 text-sm">仓库为空</p>
              <p className="text-gray-600 text-xs mt-1">导入模型开始管理</p>
            </div>
          ) : (
            models.map((m: any) => (
              <div key={m.name} className="p-3 border-b border-surface-700 hover:bg-surface-800 transition-colors">
                <div className="flex items-center gap-2">
                  <span className="text-lg">{getTypeIcon(m.model_type)}</span>
                  <span className="text-sm text-white flex-1 truncate">{m.name}</span>
                  <span className="badge bg-primary-600/20 text-primary-300 text-[10px]">{m.model_type}</span>
                </div>
                <div className="flex items-center gap-2 mt-1.5">
                  <span className="text-xs text-gray-500">v{m.version || '1.0'}</span>
                  <span className="text-xs text-gray-500">{m.framework}</span>
                  <button onClick={() => setShowExport(m.name)}
                    className="text-xs text-primary-400 hover:text-primary-300 ml-auto">
                    导出
                  </button>
                </div>
                {m.description && (
                  <p className="text-xs text-gray-500 mt-1 truncate">{m.description}</p>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {showImport ? (
          <div className="card max-w-lg mx-auto">
            <h3 className="text-base font-medium text-white mb-4">导入模型</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 block mb-1">模型名称</label>
                <input value={form.name} onChange={e => setForm({...form, name: e.target.value})}
                  className="input w-full text-sm" placeholder="my-model" />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">模型类型</label>
                <select value={form.model_type} onChange={e => setForm({...form, model_type: e.target.value})}
                  className="input w-full text-sm">
                  <option value="llm">LLM</option>
                  <option value="lcm">LCM</option>
                  <option value="cnn">CNN</option>
                  <option value="lora">LoRA</option>
                  <option value="qlora">QLoRA</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">源路径</label>
                <input value={form.source_path} onChange={e => setForm({...form, source_path: e.target.value})}
                  className="input w-full text-sm" placeholder="./output/model" />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">描述</label>
                <input value={form.description} onChange={e => setForm({...form, description: e.target.value})}
                  className="input w-full text-sm" placeholder="可选描述" />
              </div>
              <div className="flex gap-2 pt-2">
                <button onClick={handleImport} disabled={!form.name} className="btn-primary text-sm">导入</button>
                <button onClick={() => setShowImport(false)} className="btn-secondary text-sm">取消</button>
              </div>
            </div>
          </div>
        ) : showExport ? (
          <div className="card max-w-lg mx-auto">
            <h3 className="text-base font-medium text-white mb-4">导出模型: {showExport}</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 block mb-1">导出格式</label>
                <select value={exportForm.format}
                  onChange={e => setExportForm({...exportForm, format: e.target.value})}
                  className="input w-full text-sm">
                  <option value="pytorch">PyTorch (.bin)</option>
                  <option value="safetensors">SafeTensors</option>
                  <option value="onnx">ONNX</option>
                  <option value="gguf">GGUF</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">输出路径</label>
                <input value={exportForm.output_path}
                  onChange={e => setExportForm({...exportForm, output_path: e.target.value})}
                  className="input w-full text-sm" />
              </div>
              <div className="flex gap-2 pt-2">
                <button onClick={() => handleExport(showExport)} className="btn-primary text-sm">导出</button>
                <button onClick={() => setShowExport(null)} className="btn-secondary text-sm">取消</button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-4xl mb-3">🗂️</div>
              <p className="text-lg">模型仓库管理</p>
              <p className="text-sm mt-1 text-gray-600">
                导入导出 · 格式转换 · 版本管理
              </p>
              <div className="flex gap-4 mt-4 justify-center text-xs text-gray-600">
                <span>🤖 LLM</span>
                <span>🎨 LCM</span>
                <span>🖼️ CNN</span>
                <span>🔗 LoRA</span>
                <span>⚡ QLoRA</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
