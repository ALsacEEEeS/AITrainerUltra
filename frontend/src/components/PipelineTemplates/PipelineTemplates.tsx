import { useState, useEffect } from 'react';
import { api } from '../../api/client';
import { useWorkflowStore } from '../../store/useWorkflowStore';
import type { Tab } from '../../App';

interface TemplateInfo {
  name: string;
  description: string;
  category: string;
  icon: string;
}

const colorMap: Record<string, string> = {
  llm: 'from-blue-500 to-blue-700',
  lora: 'from-amber-500 to-orange-700',
  qlora: 'from-green-500 to-emerald-700',
  lcm: 'from-purple-500 to-violet-700',
  cnn: 'from-cyan-500 to-teal-700',
  multimodal: 'from-pink-500 to-rose-700',
  gpt: 'from-indigo-500 to-blue-700',
  bert: 'from-emerald-500 to-teal-700',
  rnn: 'from-orange-500 to-red-700',
  lstm: 'from-violet-500 to-purple-700',
  scratch: 'from-green-500 to-lime-700',
  moe: 'from-fuchsia-500 to-pink-700',
  tpu: 'from-violet-500 to-purple-700',
  npu: 'from-amber-500 to-orange-700',
};

const stepIcons: Record<string, string> = {
  load_model: '🤖',
  dataset: '📊',
  train: '🎯',
  lora_config: '🔗',
  evaluate: '📈',
  save: '💾',
  chat: '💬',
};

interface PipelineTemplatesProps {
  onNavigate?: (tab: Tab) => void;
}

export function PipelineTemplates({ onNavigate }: PipelineTemplatesProps) {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<TemplateInfo | null>(null);
  const [workflow, setWorkflow] = useState<any>(null);
  const [applying, setApplying] = useState(false);
  const setNodes = useWorkflowStore((s) => s.setNodes);
  const setConnections = useWorkflowStore((s) => s.setConnections);

  useEffect(() => {
    api.listTemplates().then((res) => {
      setTemplates(res.data?.templates || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const handleSelect = async (template: TemplateInfo) => {
    setSelected(template);
    try {
      const res = await api.getTemplate(template.name);
      setWorkflow(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleApply = () => {
    if (!workflow) return;
    setApplying(true);
    setNodes(workflow.nodes || []);
    setConnections(workflow.connections || []);
    onNavigate?.('workflow');
    setTimeout(() => setApplying(false), 500);
  };

  const getCategoryCount = (cat: string) => templates.filter((t) => t.category === cat).length;

  return (
    <div className="flex h-full">
      {/* Template list */}
      <div className="w-72 bg-surface-900 border-r border-surface-700 flex flex-col">
        <div className="p-3 border-b border-surface-700">
          <h3 className="text-sm font-medium text-gray-300">流水线模板</h3>
          <p className="text-xs text-gray-500 mt-0.5">预置训练工作流</p>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {loading ? (
            <div className="text-center text-gray-500 text-sm p-4">加载中...</div>
          ) : (
            ['llm', 'lora', 'qlora', 'lcm', 'cnn', 'multimodal', 'gpt', 'bert', 'rnn', 'lstm', 'scratch', 'moe', 'tpu', 'npu'].map((cat) => {
              const count = getCategoryCount(cat);
              if (count === 0) return null;
              const catTemplates = templates.filter((t) => t.category === cat);
              return (
                <div key={cat} className="mb-3">
                  <h4 className="text-xs text-gray-500 uppercase px-2 mb-1">{cat} ({count})</h4>
                  {catTemplates.map((t) => (
                    <div
                      key={t.name}
                      onClick={() => handleSelect(t)}
                      className={`p-3 rounded-lg cursor-pointer transition-all mb-1 ${
                        selected?.name === t.name
                          ? 'bg-primary-600/20 border border-primary-500/30'
                          : 'hover:bg-surface-800 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{t.icon}</span>
                        <span className="text-sm text-white font-medium">{t.name}</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1 line-clamp-2">{t.description}</p>
                    </div>
                  ))}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Template detail */}
      <div className="flex-1 overflow-y-auto p-4">
        {selected && workflow ? (
          <div className="max-w-3xl mx-auto space-y-4">
            <div className="card">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-3xl">{selected.icon}</span>
                  <div>
                    <h2 className="text-lg font-medium text-white">{selected.name}</h2>
                    <p className="text-sm text-gray-400">{selected.description}</p>
                  </div>
                </div>
                <button
                  onClick={handleApply}
                  disabled={applying}
                  className="btn-primary"
                >
                  {applying ? '⏳ 应用中...' : '🚀 应用到工作流'}
                </button>
              </div>
            </div>

            {/* Pipeline visualization */}
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-4">流水线步骤</h3>
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                {workflow.nodes?.map((node: any, i: number) => (
                  <div key={node.id} className="flex items-center gap-2 shrink-0">
                    <div className={`bg-gradient-to-br ${colorMap[selected.category] || 'from-primary-500 to-primary-700'} rounded-xl p-3 text-center min-w-[100px]`}>
                      <div className="text-xl mb-1">{stepIcons[node.type] || '⚙️'}</div>
                      <div className="text-xs text-white font-medium">{node.type.replace('_', ' ')}</div>
                      {node.config?.model_name && (
                        <div className="text-[10px] text-white/70 truncate max-w-[100px]">{node.config.model_name.split('/').pop()}</div>
                      )}
                    </div>
                    {i < workflow.nodes.length - 1 && (
                      <div className="text-gray-600 text-lg">→</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Node details */}
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">节点配置详情</h3>
              <div className="space-y-2">
                {workflow.nodes?.map((node: any) => (
                  <details key={node.id} className="bg-surface-900 rounded-lg">
                    <summary className="p-3 cursor-pointer text-sm text-gray-200 hover:text-white flex items-center gap-2">
                      <span>{stepIcons[node.type] || '⚙️'}</span>
                      <span className="capitalize">{node.type.replace('_', ' ')}</span>
                      <span className="text-xs text-gray-500 ml-auto">
                        {Object.keys(node.config).length} 参数
                      </span>
                    </summary>
                    <div className="px-3 pb-3 grid grid-cols-2 gap-2 text-xs">
                      {Object.entries(node.config).map(([k, v]) => (
                        <div key={k} className="text-gray-400">
                          <span className="text-gray-500">{k}: </span>
                          <span className="text-gray-200">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  </details>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-4xl mb-3">📋</div>
              <p className="text-lg">选择一个流水线模板</p>
              <p className="text-sm mt-1 text-gray-600">
                预置了 LLM微调 / LoRA / QLoRA / LCM / CNN 等常用模板
              </p>
              <div className="flex gap-2 mt-4 justify-center text-xs text-gray-600">
                {templates.map((t) => (
                  <span key={t.name} className="bg-surface-800 px-2 py-1 rounded">{t.icon} {t.name}</span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
