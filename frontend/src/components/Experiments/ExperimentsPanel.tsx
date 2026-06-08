import { useState, useEffect } from 'react';
import { api } from '../../api/client';

interface Experiment {
  name: string;
  model_type: string;
  model_name: string;
  status: string;
  final_metrics: Record<string, number>;
  hyperparameters: Record<string, any>;
  duration: number;
  created_at: number;
  tags: string[];
  notes: string;
}

export function ExperimentsPanel() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ name: '', model_type: 'llm', model_name: '' });

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const res = await api.listExperiments();
      setExperiments(res.data?.experiments || []);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  useEffect(() => { fetchExperiments(); }, []);

  const handleCreate = async () => {
    await api.createExperiment(form);
    setShowCreate(false);
    setForm({ name: '', model_type: 'llm', model_name: '' });
    fetchExperiments();
  };

  const handleDelete = async (name: string) => {
    await api.deleteExperiment(name);
    fetchExperiments();
  };

  const toggleCompare = (name: string) => {
    setCompareIds((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name],
    );
  };

  const formatTime = (t: number) => new Date(t * 1000).toLocaleString();

  const selectedExp = experiments.find((e) => e.name === selected);
  const compareExps = experiments.filter((e) => compareIds.includes(e.name));
  const isCompareMode = compareIds.length >= 2;

  // Collect all metric keys across compared experiments
  const allMetricKeys = isCompareMode
    ? [...new Set(compareExps.flatMap((e) => Object.keys(e.final_metrics)))]
    : [];
  const allHpKeys = isCompareMode
    ? [...new Set(compareExps.flatMap((e) => Object.keys(e.hyperparameters)))]
    : [];

  const getBestValue = (key: string) => {
    const values = compareExps
      .map((e) => e.final_metrics[key])
      .filter((v) => v !== undefined);
    if (values.length === 0) return undefined;
    return key.toLowerCase().includes('loss') || key.toLowerCase().includes('error')
      ? Math.min(...values)
      : Math.max(...values);
  };

  return (
    <div className="flex h-full">
      {/* Experiment list */}
      <div className="w-80 bg-surface-900 border-r border-surface-700 flex flex-col">
        <div className="p-3 border-b border-surface-700 flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-300">实验列表</h3>
          <div className="flex gap-1">
            {isCompareMode && (
              <button onClick={() => setCompareIds([])} className="text-xs text-gray-400 hover:text-white px-2 py-1">
                ✕ 清除
              </button>
            )}
            <button onClick={() => setShowCreate(true)} className="btn-primary text-xs py-1 px-2">
              + 新建
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="p-4 text-center text-gray-500 text-sm">加载中...</div>
          ) : experiments.length === 0 ? (
            <div className="p-4 text-center">
              <div className="text-2xl mb-2">🔬</div>
              <p className="text-gray-500 text-sm">暂无实验记录</p>
              <p className="text-gray-600 text-xs mt-1">点击"新建"创建第一个实验</p>
            </div>
          ) : (
            experiments.map((exp) => {
              const isCompared = compareIds.includes(exp.name);
              return (
                <div
                  key={exp.name}
                  onClick={() => {
                    if (isCompareMode) {
                      toggleCompare(exp.name);
                    } else {
                      setSelected(exp.name);
                    }
                  }}
                  className={`p-3 border-b border-surface-700 cursor-pointer transition-colors ${
                    selected === exp.name && !isCompareMode
                      ? 'bg-primary-600/10 border-l-2 border-l-primary-500'
                      : isCompared
                        ? 'bg-purple-600/10 border-l-2 border-l-purple-500'
                        : 'hover:bg-surface-800'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={isCompared}
                      onChange={() => toggleCompare(exp.name)}
                      className="accent-primary-500 shrink-0"
                      onClick={(e) => e.stopPropagation()}
                    />
                    <span className={`w-2 h-2 rounded-full shrink-0 ${
                      exp.status === 'completed' ? 'bg-green-500' :
                      exp.status === 'running' ? 'bg-yellow-500 animate-pulse' :
                      exp.status === 'failed' ? 'bg-red-500' : 'bg-gray-500'
                    }`} />
                    <span className="text-sm text-white truncate flex-1">{exp.name}</span>
                    <span className="badge bg-surface-700 text-gray-300 text-[10px]">{exp.model_type}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1 ml-5">
                    {formatTime(exp.created_at)}
                  </div>
                  {exp.status === 'completed' && exp.final_metrics && (
                    <div className="text-xs text-primary-400 mt-1 ml-5">
                      {Object.entries(exp.final_metrics).slice(0, 2).map(([k, v]) => (
                        <span key={k} className="mr-2">{k}: {typeof v === 'number' ? v.toFixed(4) : v}</span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {isCompareMode && (
          <div className="p-2 border-t border-surface-700 bg-purple-600/5">
            <p className="text-xs text-purple-400 text-center">
              对比 {compareIds.length} 个实验 · 点击复选框调整
            </p>
          </div>
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-4">
        {showCreate ? (
          <div className="card max-w-lg mx-auto">
            <h3 className="text-base font-medium text-white mb-4">新建实验</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-400 block mb-1">实验名称</label>
                <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
                  className="input w-full text-sm" placeholder="my-experiment" />
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">模型类型</label>
                <select value={form.model_type} onChange={(e) => setForm({ ...form, model_type: e.target.value })}
                  className="input w-full text-sm">
                  <option value="llm">LLM</option>
                  <option value="lcm">LCM</option>
                  <option value="cnn">CNN</option>
                  <option value="lora">LoRA</option>
                  <option value="qlora">QLoRA</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">模型名称</label>
                <input value={form.model_name} onChange={(e) => setForm({ ...form, model_name: e.target.value })}
                  className="input w-full text-sm" placeholder="model-name" />
              </div>
              <div className="flex gap-2 pt-2">
                <button onClick={handleCreate} disabled={!form.name} className="btn-primary text-sm">创建</button>
                <button onClick={() => setShowCreate(false)} className="btn-secondary text-sm">取消</button>
              </div>
            </div>
          </div>
        ) : isCompareMode ? (
          <div className="space-y-4">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-4">
                📊 实验对比 · {compareExps.length} 个实验
              </h3>

              {/* Final metrics comparison table */}
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="text-gray-500 border-b border-surface-700">
                      <th className="pb-2 pr-4 font-medium">指标</th>
                      {compareExps.map((exp) => (
                        <th key={exp.name} className="pb-2 pr-4 font-medium text-gray-300">{exp.name}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {/* Status row */}
                    <tr className="border-b border-surface-800">
                      <td className="py-2 pr-4 text-gray-400">Status</td>
                      {compareExps.map((exp) => (
                        <td key={exp.name} className="py-2 pr-4">
                          <span className={`capitalize ${
                            exp.status === 'completed' ? 'text-green-400' :
                            exp.status === 'running' ? 'text-yellow-400' :
                            exp.status === 'failed' ? 'text-red-400' : 'text-gray-400'
                          }`}>{exp.status}</span>
                        </td>
                      ))}
                    </tr>
                    {/* Duration row */}
                    <tr className="border-b border-surface-800">
                      <td className="py-2 pr-4 text-gray-400">Duration</td>
                      {compareExps.map((exp) => (
                        <td key={exp.name} className="py-2 pr-4 text-gray-200">{exp.duration.toFixed(1)}s</td>
                      ))}
                    </tr>
                    {/* Metric rows */}
                    {allMetricKeys.map((key) => {
                      const best = getBestValue(key);
                      return (
                        <tr key={key} className="border-b border-surface-800">
                          <td className="py-2 pr-4 text-gray-400 font-medium">{key}</td>
                          {compareExps.map((exp) => {
                            const val = exp.final_metrics[key];
                            const isBest = val !== undefined && best !== undefined && val === best;
                            return (
                              <td key={exp.name} className={`py-2 pr-4 font-mono ${
                                isBest ? 'text-green-400 font-bold' : 'text-gray-200'
                              }`}>
                                {val !== undefined ? (typeof val === 'number' ? val.toFixed(6) : val) : '-'}
                                {isBest && <span className="ml-1 text-[10px]">★</span>}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Hyperparameters comparison */}
            {allHpKeys.length > 0 && (
              <div className="card">
                <h3 className="text-sm font-medium text-gray-300 mb-3">超参数对比</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead>
                      <tr className="text-gray-500 border-b border-surface-700">
                        <th className="pb-2 pr-4">参数</th>
                        {compareExps.map((exp) => (
                          <th key={exp.name} className="pb-2 pr-4 text-gray-300">{exp.name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {allHpKeys.map((key) => (
                        <tr key={key} className="border-b border-surface-800">
                          <td className="py-2 pr-4 text-gray-400">{key}</td>
                          {compareExps.map((exp) => (
                            <td key={exp.name} className="py-2 pr-4 text-gray-200">
                              {String(exp.hyperparameters[key] ?? '-')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Metrics overlay chart */}
            {allMetricKeys.length > 0 && (
              <div className="card">
                <h3 className="text-sm font-medium text-gray-300 mb-3">指标对比</h3>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  {allMetricKeys.slice(0, 4).map((key) => {
                    const vals = compareExps.map((e) => e.final_metrics[key]).filter((v) => v !== undefined);
                    if (vals.length === 0) return null;
                    const maxVal = Math.max(...vals);
                    const minVal = Math.min(...vals);
                    const range = maxVal - minVal || 1;
                    const barHeight = 80;
                    return (
                      <div key={key} className="bg-surface-900 rounded-lg p-3">
                        <div className="text-xs text-gray-500 mb-2">{key}</div>
                        <div className="flex items-end gap-2" style={{ height: barHeight }}>
                          {compareExps.map((exp, i) => {
                            const v = exp.final_metrics[key];
                            if (v === undefined) return null;
                            const h = ((v - minVal) / range) * barHeight;
                            return (
                              <div key={exp.name} className="flex-1 flex flex-col items-center">
                                <div
                                  className="w-full rounded-t"
                                  style={{
                                    height: `${Math.max(h, 4)}px`,
                                    backgroundColor: ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#a855f7'][i % 5],
                                  }}
                                  title={`${exp.name}: ${v.toFixed(4)}`}
                                />
                                <div className="text-[8px] text-gray-500 mt-1 truncate w-full text-center">
                                  {exp.name.length > 8 ? exp.name.slice(0, 6) + '..' : exp.name}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ) : selectedExp ? (
          <div className="space-y-4">
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-base font-medium text-white">{selectedExp.name}</h3>
                <button onClick={() => handleDelete(selectedExp.name)}
                  className="text-xs text-red-400 hover:text-red-300">删除</button>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                <div>
                  <span className="text-gray-500 text-xs">状态</span>
                  <p className="text-white capitalize">{selectedExp.status}</p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs">模型类型</span>
                  <p className="text-white">{selectedExp.model_type}</p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs">模型</span>
                  <p className="text-white truncate">{selectedExp.model_name || '-'}</p>
                </div>
                <div>
                  <span className="text-gray-500 text-xs">耗时</span>
                  <p className="text-white">{selectedExp.duration.toFixed(1)}s</p>
                </div>
              </div>
            </div>

            {Object.keys(selectedExp.final_metrics).length > 0 && (
              <div className="card">
                <h4 className="text-sm font-medium text-gray-300 mb-3">最终指标</h4>
                <div className="grid grid-cols-3 gap-3">
                  {Object.entries(selectedExp.final_metrics).map(([k, v]) => (
                    <div key={k} className="bg-surface-900 rounded-lg p-3 text-center">
                      <div className="text-xs text-gray-500">{k}</div>
                      <div className="text-lg font-mono text-primary-400">
                        {typeof v === 'number' ? v.toFixed(4) : v}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {Object.keys(selectedExp.hyperparameters).length > 0 && (
              <div className="card">
                <h4 className="text-sm font-medium text-gray-300 mb-3">超参数</h4>
                <div className="grid grid-cols-3 gap-3 text-xs text-gray-400">
                  {Object.entries(selectedExp.hyperparameters).map(([k, v]) => (
                    <div key={k}><span className="text-gray-500">{k}: </span>{String(v)}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-4xl mb-3">🔬</div>
              <p className="text-lg">选择一个实验查看详情</p>
              <p className="text-sm mt-1">勾选 2+ 个实验进入对比模式</p>
              <p className="text-xs text-gray-600 mt-1">或点击"新建"创建实验</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
