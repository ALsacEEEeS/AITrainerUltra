export type Lang = 'zh' | 'en';

export const translations: Record<string, { zh: string; en: string }> = {
  // Nav
  'nav.workflow': { zh: '节点编辑', en: 'Workflow' },
  'nav.chat': { zh: '对话交互', en: 'Chat' },
  'nav.training': { zh: '训练管理', en: 'Training' },
  'nav.optimization': { zh: '模型优化', en: 'Optimization' },
  'nav.multimodal': { zh: '多模态模型', en: 'Multimodal' },
  'nav.experiments': { zh: '实验追踪', en: 'Experiments' },
  'nav.loader': { zh: '加载模型', en: 'Load Model' },
  'nav.models': { zh: '模型管理', en: 'Models' },
  'nav.recipes': { zh: '训练食谱', en: 'Recipes' },
  'nav.templates': { zh: '流水线模板', en: 'Templates' },
  'nav.help': { zh: '帮助文档', en: 'Help' },
  'nav.ready': { zh: '就绪', en: 'Ready' },
  'nav.training_in_progress': { zh: '训练中', en: 'Training' },

  // TopBar
  'topbar.system_status': { zh: '系统状态', en: 'System' },
  'topbar.running': { zh: '运行中', en: 'Running' },
  'topbar.api_docs': { zh: 'API 文档', en: 'API Docs' },

  // Device
  'device.title': { zh: '计算设备', en: 'Compute Device' },
  'device.auto': { zh: '自动检测', en: 'Auto Detect' },
  'device.cuda': { zh: 'NVIDIA CUDA', en: 'NVIDIA CUDA' },
  'device.mps': { zh: 'Apple MPS (M1/M2/M3)', en: 'Apple MPS (M1/M2/M3)' },
  'device.rocm': { zh: 'AMD ROCm', en: 'AMD ROCm' },
  'device.tpu': { zh: 'Google TPU', en: 'Google TPU' },
  'device.npu': { zh: '华为昇腾 NPU', en: 'Huawei Ascend NPU' },
  'device.xpu': { zh: 'Intel XPU', en: 'Intel XPU' },
  'device.cpu': { zh: 'CPU', en: 'CPU' },
  'device.memory': { zh: '显存', en: 'VRAM' },
  'device.used': { zh: '已使用', en: 'Used' },
  'device.available': { zh: '可用', en: 'Available' },
  'device.total': { zh: '总量', en: 'Total' },

  // Training
  'training.config': { zh: '训练任务配置', en: 'Training Config' },
  'training.model_type': { zh: '模型类型', en: 'Model Type' },
  'training.model_name': { zh: '模型名称', en: 'Model Name' },
  'training.task_type': { zh: '任务类型', en: 'Task Type' },
  'training.output_dir': { zh: '输出目录', en: 'Output Dir' },
  'training.hyperparams': { zh: '超参数设置', en: 'Hyperparameters' },
  'training.learning_rate': { zh: '学习率', en: 'Learning Rate' },
  'training.batch_size': { zh: '批次大小', en: 'Batch Size' },
  'training.epochs': { zh: '训练轮数', en: 'Epochs' },
  'training.start': { zh: '开始训练', en: 'Start Training' },
  'training.stop': { zh: '停止', en: 'Stop' },
  'training.log': { zh: '训练日志', en: 'Training Log' },
  'training.progress': { zh: '训练进度', en: 'Progress' },
  'training.loss': { zh: '损失', en: 'Loss' },

  // Chat
  'chat.placeholder': { zh: '输入消息... (Enter发送, Shift+Enter换行)', en: 'Type a message... (Enter to send, Shift+Enter for new line)' },
  'chat.send': { zh: '发送', en: 'Send' },
  'chat.clear': { zh: '清空对话', en: 'Clear Chat' },
  'chat.select_model': { zh: '选择模型', en: 'Select Model' },
  'chat.temperature': { zh: '温度', en: 'Temperature' },
  'chat.max_tokens': { zh: '最大Token', en: 'Max Tokens' },
  'chat.welcome': { zh: '选择一个模型即可开始对话', en: 'Select a model to start chatting' },

  // Model Loader
  'loader.title': { zh: '模型来源', en: 'Model Source' },
  'loader.huggingface': { zh: 'HuggingFace Hub', en: 'HuggingFace Hub' },
  'loader.local': { zh: '本地文件', en: 'Local Files' },
  'loader.gguf': { zh: 'GGUF格式', en: 'GGUF Format' },
  'loader.lora': { zh: 'LoRA适配器', en: 'LoRA Adapter' },
  'loader.load': { zh: '加载', en: 'Load' },
  'loader.detect': { zh: '检测', en: 'Detect' },
  'loader.browse': { zh: '浏览', en: 'Browse' },
  'loader.quick_load': { zh: '快速加载', en: 'Quick Load' },

  // Optimization
  'opt.title': { zh: '优化预设', en: 'Optimization Presets' },
  'opt.woq': { zh: 'WOQ量化', en: 'WOQ Quant' },
  'opt.kv_offload': { zh: 'KV卸载', en: 'KV Offload' },
  'opt.dms': { zh: 'DMS压缩', en: 'DMS Compress' },
  'opt.vram': { zh: '自适应VRAM', en: 'Adaptive VRAM' },
  'opt.apply': { zh: '应用', en: 'Apply' },
  'opt.current': { zh: '当前', en: 'Current' },
  'opt.none': { zh: '无优化', en: 'No Optimizations' },

  // Common
  'common.save': { zh: '保存', en: 'Save' },
  'common.cancel': { zh: '取消', en: 'Cancel' },
  'common.delete': { zh: '删除', en: 'Delete' },
  'common.search': { zh: '搜索', en: 'Search' },
  'common.loading': { zh: '加载中...', en: 'Loading...' },
  'common.error': { zh: '错误', en: 'Error' },
  'common.success': { zh: '成功', en: 'Success' },
  'common.no_data': { zh: '暂无数据', en: 'No Data' },
};

export function t(key: string, lang: Lang): string {
  const entry = translations[key];
  if (!entry) return key;
  return entry[lang];
}
