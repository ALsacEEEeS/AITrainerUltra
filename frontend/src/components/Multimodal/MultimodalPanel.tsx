import { useState, useRef } from 'react';
import { api } from '../../api/client';
import { useTrainingStore } from '../../store/useTrainingStore';

interface MultimodalModel {
  id: string;
  name: string;
  type: 'clip' | 'blip' | 'vlm' | 'custom' | 'composite';
  description: string;
  modalities: string[];
  icon: string;
}

const MODELS: MultimodalModel[] = [
  { id: 'clip-vit', name: 'CLIP ViT-Base', type: 'clip',
    description: '对比式视觉-语言预训练模型，支持零样本分类和图文匹配',
    modalities: ['图像', '文本'], icon: '🔄' },
  { id: 'blip-vqa', name: 'BLIP VQA', type: 'blip',
    description: '视觉问答模型，理解图像内容并回答自然语言问题',
    modalities: ['图像', '文本', '问答'], icon: '❓' },
  { id: 'llava', name: 'LLaVA 1.5', type: 'vlm',
    description: '大型视觉-语言助手，支持多轮图文对话',
    modalities: ['图像', '文本', '对话'], icon: '💬' },
  { id: 'custom-vlm', name: '自定义VLM', type: 'custom',
    description: '自定义视觉编码器+文本编码器的融合模型',
    modalities: ['图像', '文本', '分类'], icon: '🔧' },
  { id: 'composite-mm', name: '合成多模态模型', type: 'composite',
    description: '自由组合任意视觉+文本+音频编码器为多模态模型',
    modalities: ['图像', '文本', '音频', '视频'], icon: '🧩' },
];

const ARCHITECTURES = [
  { name: 'ViT + BERT 融合', desc: 'Vision Transformer + BERT 编码器融合', dims: '768+768→512' },
  { name: 'CNN + LSTM', desc: 'CNN视觉特征 + LSTM文本编码', dims: '512+256→256' },
  { name: 'CLIP对比学习', desc: '图像-文本对比学习框架', dims: '512×512' },
  { name: '编码器-解码器', desc: '统一多模态编码-生成架构', dims: '768→768' },
];

const VISION_ENCODERS = [
  { id: 'vit-base', name: 'ViT-Base', dim: 768, model: 'google/vit-base-patch16-224' },
  { id: 'vit-large', name: 'ViT-Large', dim: 1024, model: 'google/vit-large-patch16-224' },
  { id: 'resnet-50', name: 'ResNet-50', dim: 2048, model: 'microsoft/resnet-50' },
  { id: 'clip-vit', name: 'CLIP ViT', dim: 512, model: 'openai/clip-vit-base-patch32' },
  { id: 'dinov2', name: 'DINOv2', dim: 768, model: 'facebook/dinov2-base' },
];

const TEXT_ENCODERS = [
  { id: 'bert-base', name: 'BERT-Base', dim: 768, model: 'bert-base-uncased' },
  { id: 'bert-large', name: 'BERT-Large', dim: 1024, model: 'bert-large-uncased' },
  { id: 'llm-tiny', name: 'TinyLLaMA', dim: 2048, model: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' },
  { id: 't5-small', name: 'T5-Small', dim: 512, model: 't5-small' },
  { id: 'phi-2', name: 'Phi-2', dim: 2560, model: 'microsoft/phi-2' },
];

const AUDIO_ENCODERS = [
  { id: 'whisper-tiny', name: 'Whisper-Tiny', dim: 384, model: 'openai/whisper-tiny' },
  { id: 'hubert-base', name: 'HuBERT-Base', dim: 768, model: 'facebook/hubert-base-ls960' },
];

const FUSION_METHODS = [
  { id: 'concat', name: '拼接融合 (Concat)', desc: '特征向量拼接后投影', params: '最简单, 适合分类' },
  { id: 'cross-attn', name: '交叉注意力', desc: '用文本query关注视觉特征', params: '最强, 适合VQA' },
  { id: 'weighted-sum', name: '加权求和', desc: '可学习权重融合多模态特征', params: '轻量, 适合对比学习' },
  { id: 'mlp-gate', name: 'MLP门控融合', desc: '可学习的门控机制选择模态', params: '灵活, 适合多模态' },
];

export function MultimodalPanel() {
  const [selected, setSelected] = useState<string | null>(null);
  const [inputType, setInputType] = useState<'image+text' | 'image' | 'text'>('image+text');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ─── Custom Composition State ──────────────────────────────────────────
  const [composeVision, setComposeVision] = useState('vit-base');
  const [composeText, setComposeText] = useState('bert-base');
  const [composeAudio, setComposeAudio] = useState<string>('');
  const [composeFusion, setComposeFusion] = useState('concat');
  const [composeProjectionDim, setComposeProjectionDim] = useState(512);
  const [composeName, setComposeName] = useState('我的合成多模态模型');
  const [composeStatus, setComposeStatus] = useState<string | null>(null);
  const setConfig = useTrainingStore((s) => s.setConfig);

  const selectedModel = MODELS.find(m => m.id === selected);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setImagePreview(result);
      setImageBase64(result.split(',')[1]); // Remove data:image/...;base64, prefix
    };
    reader.readAsDataURL(file);
  };

  const handleRun = async () => {
    if (!selected) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const res = await api.multimodalInfer({
        model_type: selectedModel?.type || 'clip',
        text: prompt,
        image_base64: imageBase64,
      });
      setResult(res.data);
    } catch (err: any) {
      setError(err.message || 'Inference failed');
    }
    setLoading(false);
  };

  return (
    <div className="flex h-full">
      {/* Model selection sidebar */}
      <div className="w-72 bg-surface-900 border-r border-surface-700 flex flex-col">
        <div className="p-3 border-b border-surface-700">
          <h3 className="text-sm font-medium text-gray-300">多模态模型</h3>
          <p className="text-xs text-gray-500 mt-0.5">视觉 + 语言融合模型</p>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {MODELS.map(model => (
            <div
              key={model.id}
              onClick={() => setSelected(model.id)}
              className={`p-3 rounded-lg cursor-pointer mb-1.5 transition-all ${
                selected === model.id
                  ? 'bg-primary-600/20 border border-primary-500/30'
                  : 'hover:bg-surface-800 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-xl">{model.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-white font-medium truncate">{model.name}</div>
                  <div className="flex gap-1 mt-1">
                    {model.modalities.map(m => (
                      <span key={m} className="badge bg-surface-700 text-gray-300 text-[10px]">{m}</span>
                    ))}
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1 line-clamp-2">{model.description}</p>
            </div>
          ))}
        </div>

        <div className="p-3 border-t border-surface-700">
          <div className="text-[10px] text-gray-600">
            支持架构: ViT, BERT, CLIP, BLIP, LLaVA
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-4">
        {!selected ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500 max-w-md">
              <div className="text-4xl mb-3">🖼️</div>
              <p className="text-lg mb-2">多模态模型训练与推理</p>
              <p className="text-sm text-gray-600">
                左侧选择一个模型开始。支持 CLIP 对比学习、BLIP 视觉问答、
                LLaVA 视觉对话，以及自定义 VLM 架构训练。
              </p>
              <div className="grid grid-cols-2 gap-2 mt-4 text-xs text-gray-600">
                {ARCHITECTURES.map(a => (
                  <div key={a.name} className="bg-surface-800 rounded-lg p-2 text-left">
                    <div className="text-gray-300 font-medium">{a.name}</div>
                    <div className="text-gray-500 mt-0.5">{a.desc}</div>
                    <div className="text-primary-400 mt-0.5">{a.dims}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto space-y-4">
            {/* Model header */}
            <div className="card">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-2xl">{selectedModel?.icon}</span>
                <div>
                  <h2 className="text-lg font-medium text-white">{selectedModel?.name}</h2>
                  <p className="text-sm text-gray-400">{selectedModel?.description}</p>
                </div>
              </div>
              <div className="flex gap-2">
                {selectedModel?.modalities.map(m => (
                  <span key={m} className="badge bg-primary-600/20 text-primary-300">{m}</span>
                ))}
              </div>
            </div>

            {/* Input type selector */}
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">输入配置</h3>
              <div className="flex gap-2 mb-4">
                {(['image+text', 'image', 'text'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => setInputType(t)}
                    className={`px-3 py-1.5 rounded-lg text-xs transition-colors ${
                      inputType === t
                        ? 'bg-primary-600 text-white'
                        : 'bg-surface-700 text-gray-400 hover:text-white'
                    }`}
                  >
                    {t === 'image+text' ? '图像+文本' : t === 'image' ? '仅图像' : '仅文本'}
                  </button>
                ))}
              </div>

              <div className="space-y-3">
                {(inputType === 'image+text' || inputType === 'image') && (
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">图像输入</label>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleImageSelect}
                      className="hidden"
                    />
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      className="border-2 border-dashed border-surface-700 rounded-xl p-4 text-center hover:border-primary-500/50 transition-colors cursor-pointer"
                    >
                      {imagePreview ? (
                        <div className="relative">
                          <img src={imagePreview} alt="preview" className="max-h-48 mx-auto rounded-lg" />
                          <p className="text-xs text-gray-500 mt-1">点击更换图像</p>
                        </div>
                      ) : (
                        <>
                          <div className="text-2xl mb-1">🖼️</div>
                          <p className="text-xs text-gray-500">点击选择图像</p>
                          <p className="text-[10px] text-gray-600 mt-1">支持 JPG / PNG / WEBP</p>
                        </>
                      )}
                    </div>
                  </div>
                )}
                {(inputType === 'image+text' || inputType === 'text') && (
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">文本输入</label>
                    <textarea
                      className="input w-full text-sm resize-none"
                      rows={3}
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder={selected?.includes('clip') ? '输入图像描述文本...' : '输入问题...'}
                    />
                  </div>
                )}
              </div>

              <button onClick={handleRun} disabled={loading || !selected} className="btn-primary w-full mt-4">
                {loading ? '⏳ 推理中...' : '🚀 运行推理'}
              </button>
            </div>

            {/* 🧩 Custom Composite Multimodal Model Builder */}
            {selected === 'composite-mm' && (
              <div className="card border-amber-500/30 bg-amber-600/5">
                <h3 className="text-sm font-medium text-amber-300 mb-3">🧩 合成多模态模型构建器</h3>
                <p className="text-xs text-gray-400 mb-4">
                  自由组合视觉、文本、音频编码器，构建自定义多模态模型。选择编码器后系统会自动融合。
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Vision Encoder */}
                  <div>
                    <label className="text-xs text-amber-400 block mb-1">🖼️ 视觉编码器</label>
                    <select value={composeVision} onChange={(e) => setComposeVision(e.target.value)}
                      className="input text-sm w-full">
                      {VISION_ENCODERS.map(e => (
                        <option key={e.id} value={e.id}>{e.name} (dim={e.dim})</option>
                      ))}
                    </select>
                  </div>

                  {/* Text Encoder */}
                  <div>
                    <label className="text-xs text-amber-400 block mb-1">📝 文本编码器</label>
                    <select value={composeText} onChange={(e) => setComposeText(e.target.value)}
                      className="input text-sm w-full">
                      {TEXT_ENCODERS.map(e => (
                        <option key={e.id} value={e.id}>{e.name} (dim={e.dim})</option>
                      ))}
                    </select>
                  </div>

                  {/* Audio Encoder (optional) */}
                  <div>
                    <label className="text-xs text-amber-400 block mb-1">🎵 音频编码器 (可选)</label>
                    <select value={composeAudio} onChange={(e) => setComposeAudio(e.target.value)}
                      className="input text-sm w-full">
                      <option value="">— 不使用音频 —</option>
                      {AUDIO_ENCODERS.map(e => (
                        <option key={e.id} value={e.id}>{e.name} (dim={e.dim})</option>
                      ))}
                    </select>
                  </div>

                  {/* Fusion Method */}
                  <div>
                    <label className="text-xs text-amber-400 block mb-1">🔗 融合方式</label>
                    <select value={composeFusion} onChange={(e) => setComposeFusion(e.target.value)}
                      className="input text-sm w-full">
                      {FUSION_METHODS.map(m => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                    <p className="text-[10px] text-gray-500 mt-1">
                      {FUSION_METHODS.find(m => m.id === composeFusion)?.params}
                    </p>
                  </div>

                  {/* Projection Dim */}
                  <div>
                    <label className="text-xs text-amber-400 block mb-1">投影维度</label>
                    <input type="number" value={composeProjectionDim}
                      onChange={(e) => setComposeProjectionDim(parseInt(e.target.value) || 512)}
                      className="input text-sm w-full" min={64} max={4096} />
                  </div>

                  {/* Model Name */}
                  <div>
                    <label className="text-xs text-amber-400 block mb-1">模型名称</label>
                    <input type="text" value={composeName}
                      onChange={(e) => setComposeName(e.target.value)}
                      className="input text-sm w-full" placeholder="给模型起个名字..." />
                  </div>
                </div>

                {/* Architecture preview */}
                <div className="mt-3 p-3 bg-surface-900 rounded-lg">
                  <div className="text-xs text-gray-400 mb-1">架构预览:</div>
                  <div className="text-xs text-amber-300/80 font-mono">
                    {composeAudio ? (
                      <span>[{VISION_ENCODERS.find(e => e.id === composeVision)?.name}] → [{TEXT_ENCODERS.find(e => e.id === composeText)?.name}] ← [{AUDIO_ENCODERS.find(e => e.id === composeAudio)?.name}]</span>
                    ) : (
                      <span>[{VISION_ENCODERS.find(e => e.id === composeVision)?.name}] ⨯ [{TEXT_ENCODERS.find(e => e.id === composeText)?.name}]</span>
                    )}
                    {' → '}{FUSION_METHODS.find(m => m.id === composeFusion)?.name}
                    {' → Linear({composeProjectionDim})'}
                  </div>
                </div>

                <div className="flex gap-2 mt-4">
                  <button onClick={async () => {
                    setComposeStatus('⏳ 正在构建合成多模态模型...');
                    try {
                      const res = await api.multimodalInfer({
                        action: 'compose',
                        model_type: 'composite',
                        compose_config: {
                          name: composeName,
                          vision_encoder: VISION_ENCODERS.find(e => e.id === composeVision),
                          text_encoder: TEXT_ENCODERS.find(e => e.id === composeText),
                          audio_encoder: composeAudio ? AUDIO_ENCODERS.find(e => e.id === composeAudio) : null,
                          fusion_method: composeFusion,
                          projection_dim: composeProjectionDim,
                        },
                      });
                      setComposeStatus('✅ 模型构建成功! 可在训练中使用');
                      setResult(res.data);
                    } catch (err: any) {
                      setComposeStatus('❌ 构建失败: ' + err.message);
                    }
                  }} className="btn-primary text-sm">
                    🛠️ 构建模型
                  </button>

                  <button onClick={() => {
                    const visionEnc = VISION_ENCODERS.find(e => e.id === composeVision);
                    const textEnc = TEXT_ENCODERS.find(e => e.id === composeText);
                    setConfig({
                      model_type: 'multimodal',
                      model_name: composeName,
                      task: 'multimodal',
                      hyperparameters: { learning_rate: 5e-5, batch_size: 8, num_epochs: 5 },
                      scratch_config: {
                        vision_encoder: visionEnc?.model,
                        text_encoder: textEnc?.model,
                        fusion: composeFusion,
                        projection_dim: composeProjectionDim,
                      },
                    });
                    window.location.hash = '#training';
                  }} className="btn-secondary text-sm">
                    🚀 开始训练
                  </button>
                </div>

                {composeStatus && (
                  <div className="mt-3 p-2 bg-surface-900 rounded-lg text-xs"
                    style={{ color: composeStatus.includes('✅') ? '#4ade80' : composeStatus.includes('❌') ? '#f87171' : '#fbbf24' }}>
                    {composeStatus}
                  </div>
                )}
              </div>
            )}

            {/* Training config */}
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">训练配置</h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-400 block mb-1">学习率</label>
                  <input type="number" defaultValue={5e-5} className="input w-full text-sm" step="1e-6" />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">批次大小</label>
                  <input type="number" defaultValue={8} className="input w-full text-sm" />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">训练轮数</label>
                  <input type="number" defaultValue={5} className="input w-full text-sm" />
                </div>
                <div>
                  <label className="text-xs text-gray-400 block mb-1">图像大小</label>
                  <select className="input w-full text-sm">
                    <option value={224}>224×224</option>
                    <option value={384}>384×384</option>
                    <option value={512}>512×512</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Results */}
            {result && (
              <div className="card bg-primary-600/10 border-primary-500/30">
                <h3 className="text-sm font-medium text-primary-300 mb-2">推理结果</h3>
                <div className="space-y-2 text-sm">
                  <div className="text-gray-400">
                    <span className="text-gray-500">模型:</span>{' '}
                    <span className="text-gray-200">{result.model || 'N/A'}</span>
                  </div>
                  {result.answer && (
                    <div className="text-gray-200 bg-surface-900 rounded-lg p-3">{result.answer}</div>
                  )}
                  {result.response && (
                    <div className="text-gray-200 bg-surface-900 rounded-lg p-3">{result.response}</div>
                  )}
                  {result.predictions && result.predictions.map((p: any, i: number) => (
                    <div key={i} className="flex items-center gap-2 bg-surface-900 rounded-lg p-2">
                      <span className="text-xs text-gray-500">类别{p.class}:</span>
                      <div className="flex-1 bg-surface-700 rounded-full h-2">
                        <div className="bg-primary-500 h-2 rounded-full" style={{ width: `${p.score * 100}%` }} />
                      </div>
                      <span className="text-xs font-mono text-primary-400">{(p.score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                  {result.embedding_dim && (
                    <div className="text-xs text-gray-400">
                      嵌入维度: {result.embedding_dim} · L2范数: {result.norm}
                    </div>
                  )}
                  {result.error && (
                    <div className="text-red-400 bg-red-600/10 rounded-lg p-2">{result.error}</div>
                  )}
                </div>
              </div>
            )}

            {error && (
              <div className="card bg-red-600/10 border-red-500/30">
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}

            {/* Architecture details */}
            <details className="card">
              <summary className="text-sm font-medium text-gray-300 cursor-pointer">
                模型架构详情
              </summary>
              <div className="mt-3 text-xs text-gray-400 space-y-1">
                {selected === 'clip-vit' && (
                  <>
                    <p>• 视觉编码器: ViT-Base/32 (12层, 768维)</p>
                    <p>• 文本编码器: GPT-2 (12层, 768维)</p>
                    <p>• 对比学习: InfoNCE Loss, 温度系数 0.07</p>
                    <p>• 参数量: ~150M</p>
                  </>
                )}
                {selected === 'blip-vqa' && (
                  <>
                    <p>• 视觉编码器: ViT-Base/16</p>
                    <p>• 文本编码器: BERT-Base</p>
                    <p>• 融合方式: 交叉注意力 (Cross-Attention)</p>
                    <p>• 训练目标: LM Loss + ITC Loss</p>
                  </>
                )}
                {selected === 'llava' && (
                  <>
                    <p>• 视觉编码器: CLIP ViT-L/14</p>
                    <p>• 语言模型: Vicuna-7B / LLaMA-2-7B</p>
                    <p>• 连接方式: 线性投影层 (MLP)</p>
                    <p>• 训练: 视觉指令微调</p>
                  </>
                )}
                {selected === 'custom-vlm' && (
                  <>
                    <p>• 视觉编码器: 可选 ViT / ResNet / CNN</p>
                    <p>• 文本编码器: 可选 BERT / LSTM / Transformer</p>
                    <p>• 融合方式: 投影层 + 特征相加</p>
                    <p>• 分类头: Linear(768, num_classes)</p>
                  </>
                )}
              </div>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}
