import { DeviceSelector } from '../Device/DeviceSelector';
import { useLanguage } from '../../i18n/LanguageProvider';
import type { Tab } from '../../App';

interface TopBarProps {
  status: any;
  tab?: Tab;
}

const tabLabels: Record<Tab, { zh: string; en: string }> = {
  workflow: { zh: '节点式工作流编辑器', en: 'Node Workflow Editor' },
  chat: { zh: '对话式模型交互', en: 'Model Chat' },
  training: { zh: '训练管理与监控', en: 'Training Manager' },
  optimization: { zh: '模型优化引擎', en: 'Optimization Engine' },
  loader: { zh: '模型加载器', en: 'Model Loader' },
  recipes: { zh: '训练食谱', en: 'Training Recipes' },
  multimodal: { zh: '多模态模型训练', en: 'Multimodal Training' },
  experiments: { zh: '实验追踪与对比', en: 'Experiment Tracking' },
  models: { zh: '模型仓库管理', en: 'Model Store' },
  templates: { zh: '预置流水线模板', en: 'Pipeline Templates' },
  help: { zh: '帮助文档', en: 'Help & Docs' },
};

const tabDescriptions: Record<Tab, { zh: string; en: string }> = {
  workflow: { zh: '拖拽节点构建AI训练工作流', en: 'Build AI training workflows via drag & drop' },
  chat: { zh: '与加载的模型进行对话交互', en: 'Chat with loaded models' },
  training: { zh: '配置超参数并开始训练', en: 'Configure hyperparameters and train' },
  optimization: { zh: 'WOQ量化 / KV Cache卸载 / 动态内存压缩 / 自适应显存', en: 'WOQ / KV Offload / DMS / Variable VRAM' },
  loader: { zh: '浏览和加载开源模型文件', en: 'Browse and load open-source models' },
  recipes: { zh: '预配置的训练参数模板', en: 'Pre-configured training templates' },
  multimodal: { zh: '视觉-语言多模态模型训练推理', en: 'Vision-language model training & inference' },
  experiments: { zh: '记录和比较不同实验', en: 'Record and compare experiments' },
  models: { zh: '导入/导出/转换模型格式', en: 'Import/export/convert model formats' },
  templates: { zh: '从预置模板快速开始', en: 'Quick start with preset templates' },
  help: { zh: '使用指南和功能说明', en: 'User guide and feature documentation' },
};

const deviceIcons: Record<string, string> = {
  cuda: '🟢', mps: '🔵', rocm: '🔴',
  tpu: '🟣', npu: '🟠', xpu: '🔷', cpu: '🟡',
};

export function TopBar({ status, tab = 'workflow' }: TopBarProps) {
  const { lang } = useLanguage();

  return (
    <header className="h-12 bg-surface-900 border-b border-surface-700 flex items-center justify-between px-4 shrink-0">
      <div className="flex items-center gap-4">
        <div>
          <span className="text-sm font-medium text-white">
            {tabLabels[tab]?.[lang] || tabLabels[tab]?.zh || ''}
          </span>
          <span className="text-xs text-gray-500 ml-2 hidden md:inline">
            {tabDescriptions[tab]?.[lang] || tabDescriptions[tab]?.zh || ''}
          </span>
        </div>
        {status?.engine?.running && (
          <span className="badge bg-yellow-500/20 text-yellow-400">
            <span className="animate-pulse mr-1">●</span>
            {status.engine.model_type}
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        <DeviceSelector compact defaultDevice="auto" />
        <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer"
           className="text-xs text-primary-400 hover:text-primary-300 transition-colors">API</a>
      </div>
    </header>
  );
}
