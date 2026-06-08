import { useState, useEffect, useRef, useCallback } from 'react';
import { HyperparameterForm } from './HyperparameterForm';
import { MetricsChart } from './MetricsChart';
import { ProgressBar } from './ProgressBar';
import { CheckpointList } from './CheckpointList';
import { DeviceSelector } from '../Device/DeviceSelector';
import { useTrainingStore } from '../../store/useTrainingStore';
import { useLanguage } from '../../i18n/LanguageProvider';
import { api } from '../../api/client';
import { wsClient } from '../../api/websocket';

export function TrainingPanel() {
  const { isRunning, config, setConfig, setRunning, setJobId, history, addHistory, updateMetrics, reset } = useTrainingStore();
  const { t } = useLanguage();
  const [log, setLog] = useState<string[]>([
    '[System] AITrainerUltra training engine ready',
    '[System] Supported: LLM, GPT, BERT, Multimodal, CNN, RNN, LSTM, MoE, LoRA, QLoRA, Whisper, Diffusion, Flux, T5, Phi, DETR, Embedding, SAM, SVD, I2VGen-XL, Frame Interpolation',
  ]);
  const [device, setDevice] = useState('auto');
  const [jobProgress, setJobProgress] = useState(0);
  const logRef = useRef<string[]>([]);

  // Scratch architecture config
  const [scratchSize, setScratchSize] = useState('tiny');
  const [scratchVocab, setScratchVocab] = useState('32000');
  const [scratchSeqLen, setScratchSeqLen] = useState('128');
  const [scratchHeads, setScratchHeads] = useState('8');

  // ─── Validation ────────────────────────────────────────────────────────
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const validateConfig = (): boolean => {
    const errors: string[] = [];
    if (!config.model_type) errors.push('请选择模型类型');
    if (!config.model_type.startsWith('scratch') && !config.model_type.startsWith('moe-') && config.model_type !== 'moe') {
      if (!config.model_name) errors.push('请输入模型名称 (HuggingFace ID 或本地路径)');
      else if (config.model_name.length < 2) errors.push('模型名称至少2个字符');
    }
    const hp = config.hyperparameters || {};
    if (hp.learning_rate !== undefined && hp.learning_rate <= 0) errors.push('学习率必须大于0');
    if (hp.batch_size !== undefined && hp.batch_size < 1) errors.push('批次大小必须≥1');
    if (hp.num_epochs !== undefined && hp.num_epochs < 1) errors.push('训练轮数必须≥1');
    if (hp.fp16 && hp.bf16) errors.push('fp16和bf16不能同时启用');
    setValidationErrors(errors);
    return errors.length === 0;
  };

  const addLog = useCallback((msg: string) => {
    const entry = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logRef.current = [...logRef.current, entry];
    setLog((prev) => [...prev, entry]);
  }, []);

  // Connect WebSocket for real-time training events
  useEffect(() => {
    wsClient.connect();

    const unsub1 = wsClient.on('training:start', (data: any) => {
      setRunning(true);
      setJobId(data.job_id);
      addLog(`Training started: ${data.model_type} - ${data.model_name || 'default'}`);
    });

    const unsub2 = wsClient.on('metrics:update', (data: any) => {
      if (data.metrics) {
        updateMetrics(data.metrics);
        if (data.step !== undefined && data.metrics.loss !== undefined) {
          addHistory({ loss: data.metrics.loss, step: data.step });
          setJobProgress(data.step);
        }
      }
    });

    const unsub3 = wsClient.on('training:end', (data: any) => {
      setRunning(false);
      addLog(`Training complete! Final loss: ${data.result?.final_loss?.toFixed(4) || data.result?.loss?.toFixed(4) || 'N/A'}`);
    });

    const unsub4 = wsClient.on('training:error', (data: any) => {
      setRunning(false);
      setJobId(null);
      addLog(`Error: ${data.error}`);
    });

    const unsub5 = wsClient.on('log:message', (data: any) => {
      if (data.message) addLog(data.message);
    });

    return () => {
      unsub1(); unsub2(); unsub3(); unsub4(); unsub5();
    };
  }, [addLog, setRunning, setJobId, updateMetrics, addHistory]);

  const handleStart = async () => {
    if (!validateConfig()) {
      addLog('❌ 配置验证失败，请修正红色提示的错误');
      return;
    }
    setRunning(true);
    reset();
    logRef.current = [];
    setLog([]);
    addLog(`Training started: ${config.model_type} - ${config.model_name || 'default'} (device: ${device})`);

    // Merge scratch config if applicable
    let fullConfig = { ...config, device_strategy: device };
    if (config.model_type.startsWith('scratch')) {
      const sizeMap: Record<string, any> = {
        tiny: { num_layers: 4, d_model: 256, num_heads: 4, d_ff: 1024 },
        small: { num_layers: 6, d_model: 384, num_heads: 6, d_ff: 1536 },
        medium: { num_layers: 8, d_model: 512, num_heads: 8, d_ff: 2048 },
      };
      fullConfig = {
        ...fullConfig,
        scratch_config: {
          ...sizeMap[scratchSize],
          vocab_size: parseInt(scratchVocab),
          max_seq_len: parseInt(scratchSeqLen),
          num_heads: parseInt(scratchHeads),
        },
      };
    }

    try {
      const res = await api.startTraining(fullConfig);
      const jobId = res.data?.job_id;
      if (jobId) {
        setJobId(jobId);
        addLog(`Job submitted: ${jobId}`);
      }
    } catch (err: any) {
      addLog(`Error: ${err.message}`);
      setRunning(false);
    }
  };

  const handleStop = async () => {
    try { await api.stopTraining(); } catch {}
    setRunning(false);
    setJobId(null);
    addLog('Training stopped by user');
  };

  return (
    <div className="flex h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Config */}
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-3">{t('training.config')}</h3>
          {validationErrors.length > 0 && (
            <div className="mb-3 p-2 bg-red-600/10 border border-red-500/30 rounded-lg">
              {validationErrors.map((err, i) => (
                <div key={i} className="text-xs text-red-400 flex items-center gap-1">
                  <span>⚠️</span> {err}
                </div>
              ))}
            </div>
          )}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div>
              <label className="text-xs text-gray-400 block mb-1">{t('training.model_type')}</label>
              <select value={config.model_type} onChange={(e) => setConfig({ ...config, model_type: e.target.value })}
                className="input text-sm w-full">
                <option value="llm">LLM</option><option value="gpt">GPT</option>
                <option value="bert">BERT</option><option value="t5">T5 (Text-to-Text)</option>
                <option value="phi">Phi (微软高效LLM)</option>
                <option value="multimodal">多模态 VLM</option>
                <option value="clip">CLIP</option><option value="blip">BLIP</option>
                <option value="whisper">🎤 Whisper (语音识别)</option>
                <option value="diffusion">🖼️ Stable Diffusion (文生图)</option>
                <option value="flux">✨ Flux (文生图)</option>
                <option value="video-diffusion">🎬 Stable Video Diffusion (视频)</option>
                <option value="i2vgen-xl">🎬 I2VGen-XL (文生视频)</option>
                <option value="frame-interpolation">🎬 Frame Interpolation (插帧)</option>
                <option value="detr">🎯 DETR (目标检测)</option>
                <option value="sam">🗺️ SAM (图像分割)</option>
                <option value="embedding">🔤 Embedding (文本向量)</option>
                <option value="lcm">LCM</option><option value="cnn">CNN</option>
                <option value="rnn">RNN</option><option value="lstm">LSTM</option>
                <option value="moe">MoE</option><option value="lora">LoRA</option>
                <option value="qlora">QLoRA</option>
                <option disabled>──────────</option>
                <option value="scratch-transformer">Scratch Transformer</option>
                <option value="scratch-cnn">Scratch CNN</option>
                <option value="scratch-lstm">Scratch LSTM</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">{t('training.model_name')}</label>
              <input type="text" value={config.model_name}
                onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                placeholder={config.model_type.startsWith('scratch') ? 'Scratch training, no model needed' : 'HuggingFace model ID'}
                className="input text-sm w-full" disabled={config.model_type.startsWith('scratch')} />
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">{t('training.task_type')}</label>
              <select value={config.task} onChange={(e) => setConfig({ ...config, task: e.target.value })}
                className="input text-sm w-full">
                <option value="text-generation">Text Generation</option>
                <option value="image-generation">Image Generation</option>
                <option value="video-generation">Video Generation</option>
                <option value="image-classification">Image Classification</option>
                <option value="image-segmentation">Image Segmentation</option>
                <option value="object-detection">Object Detection</option>
                <option value="sequence-classification">Sequence Classification</option>
                <option value="text2text-generation">Text-to-Text</option>
                <option value="speech-recognition">Speech Recognition</option>
                <option value="feature-extraction">Feature Extraction</option>
                <option value="language-modeling">Language Modeling</option>
                <option value="multimodal">Multimodal</option>
                <option value="question-answering">Question Answering</option>
                <option value="summarization">Summarization</option>
                <option value="translation">Translation</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-400 block mb-1">{t('training.output_dir')}</label>
              <input type="text" value={config.output_dir || './output'}
                onChange={(e) => setConfig({ ...config, output_dir: e.target.value })}
                className="input text-sm w-full" />
            </div>
          </div>
        </div>

        {/* Device Selector */}
        <DeviceSelector onDeviceChange={setDevice} defaultDevice={device} />

        {/* Hyperparams */}
        <div className="card"><HyperparameterForm /></div>

        {/* Metrics */}
        <MetricsChart />

        {/* Progress */}
        <div className="card">
          <ProgressBar value={jobProgress} max={50} label={t('training.progress')}
            color="bg-gradient-to-r from-primary-500 to-purple-500" />
          {isRunning && <div className="text-xs text-gray-500 mt-1">Step {jobProgress} / 50</div>}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <button onClick={handleStart} disabled={isRunning} className="btn-primary flex-1">
            {isRunning ? '⏳ ...' : `🚀 ${t('training.start')}`}
          </button>
          <button onClick={handleStop} disabled={!isRunning} className="btn-secondary">
            ⏹ {t('training.stop')}
          </button>
        </div>

        {/* Scratch config */}
        {config.model_type.startsWith('scratch') && (
          <div className="card border-primary-500/30 bg-primary-600/5">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-lg">🏗️</span>
              <h4 className="text-sm font-medium text-gray-300">Scratch Architecture Config</h4>
              <span className="badge bg-green-500/20 text-green-400 text-[10px] ml-auto">Random Init</span>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div>
                <label className="text-xs text-gray-400 block mb-1">Model Size</label>
                <select className="input text-sm w-full" value={scratchSize} onChange={(e) => setScratchSize(e.target.value)}>
                  <option value="tiny">Tiny (4L, 256d) ~5M</option>
                  <option value="small">Small (6L, 384d) ~20M</option>
                  <option value="medium">Medium (8L, 512d) ~50M</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Vocab</label>
                <select className="input text-sm w-full" value={scratchVocab} onChange={(e) => setScratchVocab(e.target.value)}>
                  <option value={5000}>5K</option>
                  <option value={10000}>10K</option>
                  <option value={32000}>32K</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Seq Length</label>
                <select className="input text-sm w-full" value={scratchSeqLen} onChange={(e) => setScratchSeqLen(e.target.value)}>
                  <option value={64}>64</option><option value={128}>128</option>
                  <option value={256}>256</option><option value={512}>512</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-400 block mb-1">Heads</label>
                <select className="input text-sm w-full" value={scratchHeads} onChange={(e) => setScratchHeads(e.target.value)}>
                  <option value={4}>4</option><option value={8}>8</option>
                  <option value={12}>12</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Right panel: Log + Checkpoints */}
      <div className="w-80 bg-surface-900 border-l border-surface-700 flex flex-col">
        {/* Log section */}
        <div className="flex flex-col flex-1 min-h-0">
          <div className="p-3 border-b border-surface-700">
            <h3 className="text-sm font-medium text-gray-300">{t('training.log')}</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-1">
            {log.map((entry, i) => (
              <div key={i} className={
                entry.includes('Error') ? 'text-red-400' :
                entry.includes('complete') ? 'text-green-400' :
                entry.includes('Training started') ? 'text-primary-400' : 'text-gray-400'
              }>{entry}</div>
            ))}
          </div>
        </div>

        {/* Checkpoints section */}
        <div className="border-t border-surface-700 flex flex-col max-h-48">
          <div className="p-3 border-b border-surface-700">
            <h3 className="text-sm font-medium text-gray-300">📦 检查点 (Checkpoints)</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            <CheckpointList outputDir={config.output_dir || './output'} />
          </div>
        </div>
      </div>
    </div>
  );
}
