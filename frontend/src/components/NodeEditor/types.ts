export interface NodeDefinition {
  type: string;
  label: string;
  category: 'input' | 'model' | 'training' | 'data' | 'output' | 'processing' | 'custom';
  icon: string;
  color: string;
  inputs: { id: string; label: string; type: string }[];
  outputs: { id: string; label: string; type: string }[];
  configFields: ConfigField[];
}

export interface ConfigField {
  name: string;
  label: string;
  type: 'text' | 'number' | 'select' | 'boolean' | 'json' | 'code';
  defaultValue: any;
  options?: { label: string; value: any }[];
  placeholder?: string;
}

export const NODE_DEFINITIONS: NodeDefinition[] = [
  // ─── 模型节点 ───────────────────────────────────────────────
  {
    type: 'load_model',
    label: '加载模型',
    category: 'model',
    icon: '🤖',
    color: '#6366f1',
    inputs: [],
    outputs: [{ id: 'model', label: '模型', type: 'model' }],
    configFields: [
      { name: 'model_name', label: '模型名称', type: 'text', defaultValue: '', placeholder: '如: gpt2, bert-base-uncased' },
      { name: 'model_type', label: '模型类型', type: 'select', defaultValue: 'llm', options: [
        { label: 'LLM', value: 'llm' },
        { label: 'GPT', value: 'gpt' },
        { label: 'BERT', value: 'bert' },
        { label: 'T5 (Text-to-Text)', value: 't5' },
        { label: 'Phi (微软高效LLM)', value: 'phi' },
        { label: 'RNN', value: 'rnn' },
        { label: 'LSTM', value: 'lstm' },
        { label: '多模态(VLM)', value: 'multimodal' },
        { label: 'CLIP', value: 'clip' },
        { label: 'BLIP', value: 'blip' },
        { label: '🎤 Whisper(语音)', value: 'whisper' },
        { label: '🖼️ Diffusion(文生图)', value: 'diffusion' },
        { label: '✨ Flux(文生图)', value: 'flux' },
        { label: '🎬 SVD(视频生成)', value: 'video-diffusion' },
        { label: '🎬 I2VGenXL(文生视频)', value: 'i2vgen-xl' },
        { label: '🎬 插帧(帧率提升)', value: 'frame-interpolation' },
        { label: '🎯 DETR(目标检测)', value: 'detr' },
        { label: '🗺️ SAM(图像分割)', value: 'sam' },
        { label: '🔤 Embedding(文本向量)', value: 'embedding' },
        { label: 'LCM', value: 'lcm' },
        { label: 'CNN', value: 'cnn' },
        { label: 'MoE', value: 'moe' },
        { label: 'LoRA', value: 'lora' },
        { label: 'QLoRA', value: 'qlora' },
        { label: '从零训练', value: 'scratch' },
      ]},
      { name: 'load_in_8bit', label: '8-bit量化', type: 'boolean', defaultValue: false },
      { name: 'load_in_4bit', label: '4-bit量化', type: 'boolean', defaultValue: false },
      { name: 'device', label: '设备', type: 'select', defaultValue: 'auto', options: [
        { label: '自动', value: 'auto' },
        { label: 'CUDA', value: 'cuda' },
        { label: 'MPS', value: 'mps' },
        { label: 'CPU', value: 'cpu' },
      ]},
    ],
  },
  {
    type: 'lora_config',
    label: 'LoRA配置',
    category: 'model',
    icon: '🔗',
    color: '#f59e0b',
    inputs: [{ id: 'model', label: '模型', type: 'model' }],
    outputs: [{ id: 'model', label: '带LoRA模型', type: 'model' }],
    configFields: [
      { name: 'r', label: 'LoRA秩(r)', type: 'number', defaultValue: 16 },
      { name: 'alpha', label: 'Alpha', type: 'number', defaultValue: 32 },
      { name: 'dropout', label: 'Dropout', type: 'number', defaultValue: 0.05 },
    ],
  },
  {
    type: 'qlora_config',
    label: 'QLoRA配置',
    category: 'model',
    icon: '💎',
    color: '#8b5cf6',
    inputs: [{ id: 'model', label: '模型', type: 'model' }],
    outputs: [{ id: 'model', label: '量化模型', type: 'model' }],
    configFields: [
      { name: 'bits', label: '量化位数', type: 'select', defaultValue: '4', options: [
        { label: '4-bit', value: '4' }, { label: '8-bit', value: '8' },
      ]},
      { name: 'double_quant', label: '双重量化', type: 'boolean', defaultValue: true },
    ],
  },
  {
    type: 'merge_models',
    label: '模型合并',
    category: 'model',
    icon: '🔀',
    color: '#ec4899',
    inputs: [
      { id: 'model_a', label: '模型A', type: 'model' },
      { id: 'model_b', label: '模型B', type: 'model' },
    ],
    outputs: [{ id: 'model', label: '合并模型', type: 'model' }],
    configFields: [
      { name: 'method', label: '合并方法', type: 'select', defaultValue: 'linear', options: [
        { label: '线性合并', value: 'linear' },
        { label: 'SLERP', value: 'slerp' },
        { label: 'TIES', value: 'ties' },
        { label: 'DARE', value: 'dare' },
      ]},
      { name: 'weight', label: '权重', type: 'number', defaultValue: 0.5 },
    ],
  },

  // ─── 数据节点 ───────────────────────────────────────────────
  {
    type: 'dataset',
    label: '数据集',
    category: 'data',
    icon: '📊',
    color: '#22c55e',
    inputs: [],
    outputs: [{ id: 'data', label: '数据', type: 'data' }],
    configFields: [
      { name: 'source', label: '数据源', type: 'text', defaultValue: '', placeholder: 'HuggingFace ID 或本地路径' },
      { name: 'split', label: '划分', type: 'select', defaultValue: 'train', options: [
        { label: '训练集', value: 'train' },
        { label: '验证集', value: 'validation' },
        { label: '测试集', value: 'test' },
      ]},
      { name: 'max_samples', label: '最大样本数', type: 'number', defaultValue: 1000 },
    ],
  },
  {
    type: 'preprocess',
    label: '数据预处理',
    category: 'data',
    icon: '🧹',
    color: '#10b981',
    inputs: [{ id: 'data', label: '原始数据', type: 'data' }],
    outputs: [{ id: 'data', label: '处理后数据', type: 'data' }],
    configFields: [
      { name: 'normalize', label: '归一化', type: 'boolean', defaultValue: true },
      { name: 'remove_stopwords', label: '去停用词', type: 'boolean', defaultValue: false },
      { name: 'lowercase', label: '转小写', type: 'boolean', defaultValue: true },
      { name: 'max_length', label: '截断长度', type: 'number', defaultValue: 512 },
    ],
  },
  {
    type: 'augment',
    label: '数据增强',
    category: 'data',
    icon: '🔬',
    color: '#34d399',
    inputs: [{ id: 'data', label: '数据', type: 'data' }],
    outputs: [{ id: 'data', label: '增强数据', type: 'data' }],
    configFields: [
      { name: 'method', label: '增强方法', type: 'select', defaultValue: 'mixup', options: [
        { label: 'MixUp', value: 'mixup' },
        { label: 'CutMix', value: 'cutmix' },
        { label: '噪声注入', value: 'noise' },
        { label: '回译', value: 'backtranslate' },
        { label: 'EDA', value: 'eda' },
      ]},
      { name: 'factor', label: '增强系数', type: 'number', defaultValue: 0.3 },
    ],
  },
  {
    type: 'filter',
    label: '数据过滤',
    category: 'data',
    icon: '📏',
    color: '#6ee7b7',
    inputs: [{ id: 'data', label: '数据', type: 'data' }],
    outputs: [{ id: 'data', label: '过滤后数据', type: 'data' }],
    configFields: [
      { name: 'min_length', label: '最小长度', type: 'number', defaultValue: 10 },
      { name: 'max_length', label: '最大长度', type: 'number', defaultValue: 8192 },
      { name: 'deduplicate', label: '去重', type: 'boolean', defaultValue: true },
    ],
  },
  {
    type: 'split_data',
    label: '数据拆分',
    category: 'data',
    icon: '✂️',
    color: '#a7f3d0',
    inputs: [{ id: 'data', label: '数据', type: 'data' }],
    outputs: [
      { id: 'train', label: '训练集', type: 'data' },
      { id: 'eval', label: '验证集', type: 'data' },
    ],
    configFields: [
      { name: 'ratio', label: '训练集比例', type: 'number', defaultValue: 0.8 },
      { name: 'shuffle', label: '打乱', type: 'boolean', defaultValue: true },
      { name: 'seed', label: '随机种子', type: 'number', defaultValue: 42 },
    ],
  },

  // ─── 训练节点 ───────────────────────────────────────────────
  {
    type: 'train',
    label: '训练',
    category: 'training',
    icon: '🎯',
    color: '#ef4444',
    inputs: [
      { id: 'model', label: '模型', type: 'model' },
      { id: 'data', label: '数据', type: 'data' },
    ],
    outputs: [{ id: 'result', label: '训练结果', type: 'model' }],
    configFields: [
      { name: 'learning_rate', label: '学习率', type: 'number', defaultValue: 5e-5 },
      { name: 'batch_size', label: '批次大小', type: 'number', defaultValue: 8 },
      { name: 'num_epochs', label: '训练轮数', type: 'number', defaultValue: 3 },
      { name: 'output_dir', label: '输出目录', type: 'text', defaultValue: './output' },
      { name: 'optimizer', label: '优化器', type: 'select', defaultValue: 'adamw', options: [
        { label: 'AdamW', value: 'adamw' },
        { label: 'Adam', value: 'adam' },
        { label: 'SGD', value: 'sgd' },
        { label: 'Lion', value: 'lion' },
      ]},
      { name: 'gradient_accumulation', label: '梯度累积', type: 'number', defaultValue: 1 },
      { name: 'fp16', label: '混合精度(fp16)', type: 'boolean', defaultValue: false },
      { name: 'bf16', label: 'bfloat16', type: 'boolean', defaultValue: false },
    ],
  },
  {
    type: 'evaluate',
    label: '评估',
    category: 'training',
    icon: '📈',
    color: '#06b6d4',
    inputs: [
      { id: 'model', label: '模型', type: 'model' },
      { id: 'data', label: '测试数据', type: 'data' },
    ],
    outputs: [{ id: 'metrics', label: '指标', type: 'metrics' }],
    configFields: [
      { name: 'metric', label: '评估指标', type: 'select', defaultValue: 'accuracy', options: [
        { label: '准确率', value: 'accuracy' },
        { label: 'F1分数', value: 'f1' },
        { label: '精确率', value: 'precision' },
        { label: '召回率', value: 'recall' },
        { label: '困惑度', value: 'perplexity' },
        { label: 'ROUGE', value: 'rouge' },
        { label: 'BLEU', value: 'bleu' },
      ]},
    ],
  },
  {
    type: 'hyperparameter_opt',
    label: '超参搜索',
    category: 'training',
    icon: '🔍',
    color: '#fb923c',
    inputs: [],
    outputs: [{ id: 'params', label: '最佳参数', type: 'params' }],
    configFields: [
      { name: 'method', label: '搜索方法', type: 'select', defaultValue: 'grid', options: [
        { label: '网格搜索', value: 'grid' },
        { label: '随机搜索', value: 'random' },
        { label: '贝叶斯优化', value: 'bayesian' },
      ]},
      { name: 'n_trials', label: '试验次数', type: 'number', defaultValue: 20 },
    ],
  },
  {
    type: 'schedule',
    label: '调度器',
    category: 'training',
    icon: '⏱️',
    color: '#f97316',
    inputs: [{ id: 'optimizer', label: '优化器', type: 'optimizer' }],
    outputs: [{ id: 'scheduler', label: '调度器', type: 'scheduler' }],
    configFields: [
      { name: 'scheduler_type', label: '调度类型', type: 'select', defaultValue: 'cosine', options: [
        { label: '余弦退火', value: 'cosine' },
        { label: '线性', value: 'linear' },
        { label: '阶梯', value: 'step' },
        { label: '多项式', value: 'polynomial' },
        { label: '恒定', value: 'constant' },
      ]},
      { name: 'warmup_ratio', label: '预热比例', type: 'number', defaultValue: 0.1 },
    ],
  },

  // ─── 处理节点 ───────────────────────────────────────────────
  {
    type: 'transform',
    label: '格式转换',
    category: 'processing',
    icon: '🔄',
    color: '#14b8a6',
    inputs: [{ id: 'input', label: '输入', type: 'any' }],
    outputs: [{ id: 'output', label: '输出', type: 'any' }],
    configFields: [
      { name: 'target_format', label: '目标格式', type: 'select', defaultValue: 'safetensors', options: [
        { label: 'SafeTensors', value: 'safetensors' },
        { label: 'PyTorch', value: 'pytorch' },
        { label: 'GGUF', value: 'gguf' },
        { label: 'ONNX', value: 'onnx' },
        { label: 'TensorFlow', value: 'tensorflow' },
        { label: 'MLX', value: 'mlx' },
      ]},
    ],
  },
  {
    type: 'visualize',
    label: '可视化',
    category: 'processing',
    icon: '📊',
    color: '#2dd4bf',
    inputs: [{ id: 'metrics', label: '指标数据', type: 'metrics' }],
    outputs: [{ id: 'chart', label: '图表', type: 'chart' }],
    configFields: [
      { name: 'chart_type', label: '图表类型', type: 'select', defaultValue: 'line', options: [
        { label: '折线图', value: 'line' },
        { label: '柱状图', value: 'bar' },
        { label: '散点图', value: 'scatter' },
        { label: '热力图', value: 'heatmap' },
      ]},
      { name: 'title', label: '标题', type: 'text', defaultValue: '训练指标' },
    ],
  },
  {
    type: 'notebook',
    label: '代码执行',
    category: 'processing',
    icon: '📓',
    color: '#5eead4',
    inputs: [{ id: 'input', label: '输入数据', type: 'any' }],
    outputs: [{ id: 'output', label: '输出结果', type: 'any' }],
    configFields: [
      { name: 'code', label: 'Python代码', type: 'code', defaultValue: '# 在此编写Python代码\nresult = input_data\n', placeholder: '输入Python代码...' },
    ],
  },
  {
    type: 'export',
    label: '导出模型',
    category: 'output',
    icon: '📤',
    color: '#94a3b8',
    inputs: [{ id: 'model', label: '模型', type: 'model' }],
    outputs: [],
    configFields: [
      { name: 'path', label: '导出路径', type: 'text', defaultValue: './exports/' },
      { name: 'format', label: '导出格式', type: 'select', defaultValue: 'safetensors', options: [
        { label: 'SafeTensors', value: 'safetensors' },
        { label: 'PyTorch', value: 'pytorch' },
        { label: 'GGUF', value: 'gguf' },
        { label: 'ONNX', value: 'onnx' },
        { label: 'MLX', value: 'mlx' },
      ]},
      { name: 'quantize', label: '导出时量化', type: 'select', defaultValue: 'none', options: [
        { label: '不量化', value: 'none' },
        { label: 'INT8', value: 'int8' },
        { label: 'INT4', value: 'int4' },
      ]},
    ],
  },

  // ─── 输出节点 ───────────────────────────────────────────────
  {
    type: 'chat',
    label: '对话',
    category: 'output',
    icon: '💬',
    color: '#a855f7',
    inputs: [{ id: 'model', label: '模型', type: 'model' }],
    outputs: [{ id: 'response', label: '回复', type: 'text' }],
    configFields: [
      { name: 'prompt', label: '提示词', type: 'text', defaultValue: '', placeholder: '输入对话提示...' },
      { name: 'max_tokens', label: '最大Token数', type: 'number', defaultValue: 512 },
      { name: 'temperature', label: '温度', type: 'number', defaultValue: 0.7 },
      { name: 'system_prompt', label: '系统提示', type: 'text', defaultValue: '' },
    ],
  },
  {
    type: 'save',
    label: '保存模型',
    category: 'output',
    icon: '💾',
    color: '#64748b',
    inputs: [{ id: 'model', label: '模型', type: 'model' }],
    outputs: [],
    configFields: [
      { name: 'path', label: '保存路径', type: 'text', defaultValue: './output/final' },
      { name: 'format', label: '格式', type: 'select', defaultValue: 'safetensors', options: [
        { label: 'SafeTensors', value: 'safetensors' },
        { label: 'PyTorch', value: 'pytorch' },
        { label: 'GGUF', value: 'gguf' },
      ]},
      { name: 'save_tokenizer', label: '保存Tokenizr', type: 'boolean', defaultValue: true },
    ],
  },

  // ─── 自定义节点 ─────────────────────────────────────────────
  {
    type: 'custom',
    label: '自定义节点',
    category: 'custom',
    icon: '🧩',
    color: '#e2e8f0',
    inputs: [
      { id: 'in_1', label: '输入1', type: 'any' },
      { id: 'in_2', label: '输入2', type: 'any' },
    ],
    outputs: [
      { id: 'out_1', label: '输出1', type: 'any' },
      { id: 'out_2', label: '输出2', type: 'any' },
    ],
    configFields: [
      { name: 'node_name', label: '节点名称', type: 'text', defaultValue: '我的自定义节点' },
      { name: 'code', label: 'Python代码', type: 'code', defaultValue: '# 输入: in_1, in_2\n# 输出: out_1, out_2\nresult = in_1 if in_1 else in_2\nout_1 = result', placeholder: '编写自定义处理逻辑...' },
      { name: 'description', label: '描述', type: 'text', defaultValue: '' },
    ],
  },
];
