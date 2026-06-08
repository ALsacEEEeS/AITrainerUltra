import { useState, useEffect } from 'react';
import { api } from '../../api/client';

interface PresetInfo {
  name: string;
  description: string;
  woq_bits: number;
  woq_desc: string;
  kv_offload: boolean;
  dms: boolean;
  variable_vram: boolean;
  flash_attn: boolean;
  flash_attn_backend?: string;
  is_compatible: boolean;
  incompatibility_reason: string;
}

const presetIcons: Record<string, string> = {
  none: '🔄', memory_save: '💾', extreme: '🔥', quality: '🎯',
  tpu_mode: '🟣', mps_mode: '🍎', m4_optimized: '🍎', m5_optimized: '🍎',
  nvidia_rtx50: '🟢', nvidia_rtx40: '🟢', nvidia_rtx30: '🟢',
  nvidia_rtx20: '🔵', nvidia_pro: '💎', nvidia_hopper: '⚡',
};

const presetColors: Record<string, string> = {
  none: 'bg-gray-600', memory_save: 'bg-blue-600', extreme: 'bg-red-600',
  quality: 'bg-green-600', tpu_mode: 'bg-purple-600',
  mps_mode: 'bg-sky-600', m4_optimized: 'bg-sky-500', m5_optimized: 'bg-sky-400',
  nvidia_rtx50: 'bg-emerald-500', nvidia_rtx40: 'bg-emerald-600',
  nvidia_rtx30: 'bg-teal-600', nvidia_rtx20: 'bg-cyan-600',
  nvidia_pro: 'bg-amber-500', nvidia_hopper: 'bg-red-500',
};

export function OptimizationPanel() {
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [current, setCurrent] = useState<any>(null);
  const [selectedPreset, setSelectedPreset] = useState('none');
  const [applying, setApplying] = useState(false);
  const [vram, setVram] = useState<any>(null);
  const [flashAvail, setFlashAvail] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'woq' | 'kv' | 'dms' | 'vram' | 'flash'>('overview');

  const fetchData = async () => {
    try {
      const res = await api.listOptimizations();
      setPresets(res.data?.presets || []);
      setCurrent(res.data?.current);
      setVram(res.data?.current?.vram);
      setFlashAvail(res.data?.flash_attention_available || false);
    } catch (err) { console.error(err); }
  };

  useEffect(() => { fetchData(); }, []);

  const handleApply = async (name: string) => {
    setApplying(true);
    setSelectedPreset(name);
    try {
      const res = await api.applyOptimizations({ preset: name });
      setCurrent(res.data);
      setVram(res.data?.vram);
    } catch (err) { console.error(err); }
    setTimeout(() => setApplying(false), 500);
  };

  return (
    <div className="flex h-full">
      {/* Preset sidebar */}
      <div className="w-72 bg-surface-900 border-r border-surface-700 flex flex-col">
        <div className="p-3 border-b border-surface-700">
          <h3 className="text-sm font-medium text-gray-300">优化预设</h3>
          <p className="text-xs text-gray-500 mt-0.5">点击选择预设 · 自动检测兼容性</p>
          <button
            onClick={() => handleApply('auto')}
            className="mt-2 text-xs px-2 py-1 rounded bg-primary-600/20 text-primary-400 hover:bg-primary-600/30 transition-colors w-full"
          >
            ⚡ 自动检测最优预设
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {presets.map((p) => {
            const isSelected = selectedPreset === p.name;
            return (
              <div
                key={p.name}
                onClick={() => p.is_compatible && handleApply(p.name)}
                className={`p-3 rounded-lg cursor-pointer mb-1.5 transition-all ${
                  isSelected
                    ? 'bg-primary-600/20 border border-primary-500/30'
                    : 'hover:bg-surface-800 border border-transparent'
                } ${!p.is_compatible ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                <div className="flex items-center gap-2">
                  <span>{presetIcons[p.name] || '⚙️'}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-white font-medium truncate">{p.name}</span>
                      <span className={`w-2 h-2 rounded-full ${presetColors[p.name] || 'bg-gray-500'}`} />
                      {p.is_compatible ? (
                        <span className="text-green-400 text-xs" title="当前硬件支持">✓</span>
                      ) : (
                        <span className="text-red-400 text-xs" title={p.incompatibility_reason}>✗</span>
                      )}
                    </div>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-1">{p.description}</p>
                {!p.is_compatible && p.incompatibility_reason && (
                  <p className="text-[10px] text-red-400 mt-1">⚠ {p.incompatibility_reason}</p>
                )}
                <div className="flex gap-1.5 mt-1.5 flex-wrap">
                  {p.flash_attn && (
                    <span className="badge bg-purple-600/20 text-purple-300 text-[10px]">
                      ⚡FlashAttn
                    </span>
                  )}
                  {p.woq_bits < 32 && (
                    <span className="badge bg-primary-600/20 text-primary-300 text-[10px]">
                      WOQ {p.woq_bits}bit
                    </span>
                  )}
                  {p.kv_offload && (
                    <span className="badge bg-amber-600/20 text-amber-300 text-[10px]">KV卸载</span>
                  )}
                  {p.dms && (
                    <span className="badge bg-cyan-600/20 text-cyan-300 text-[10px]">DMS</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="p-3 border-t border-surface-700 text-[10px] text-gray-600">
          <div className="flex items-center gap-1 mb-1">
            {flashAvail ? (
              <span className="text-purple-400">⚡ Flash Attention 可用</span>
            ) : (
              <span className="text-gray-500">⚡ Flash Attention 不可用</span>
            )}
          </div>
          {current?.applied_optimizations?.length > 0 ? (
            <div>已应用: {current.applied_optimizations.join(' · ')}</div>
          ) : (
            <div>当前: 无优化</div>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Tab navigation */}
        <div className="flex gap-2 mb-4 flex-wrap">
          {(['overview', 'woq', 'kv', 'dms', 'vram', 'flash'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-3 py-1.5 rounded-lg text-xs transition-colors ${
                activeTab === tab
                  ? 'bg-primary-600 text-white'
                  : 'bg-surface-800 text-gray-400 hover:text-white border border-surface-700'
              }`}
            >
              {tab === 'overview' ? '📊 总览' :
               tab === 'woq' ? '🔢 WOQ' :
               tab === 'kv' ? '💾 KV卸载' :
               tab === 'dms' ? '📦 DMS' :
               tab === 'vram' ? '🎛️ VRAM' :
               tab === 'flash' ? '⚡ FlashAttn' : tab}
            </button>
          ))}
        </div>

        {/* Overview tab */}
        {activeTab === 'overview' && (
          <div className="space-y-4">
            {/* VRAM Status */}
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">显存/内存状态</h3>
              {vram ? (
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="bg-surface-900 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500">已使用</div>
                    <div className="text-lg font-mono text-yellow-400">{vram.used_gb || 0} GB</div>
                  </div>
                  <div className="bg-surface-900 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500">可用</div>
                    <div className="text-lg font-mono text-green-400">{vram.available_gb || 'N/A'} GB</div>
                  </div>
                  <div className="bg-surface-900 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500">总容量</div>
                    <div className="text-lg font-mono text-primary-400">{vram.total_gb || 'N/A'} GB</div>
                  </div>
                  <div className="bg-surface-900 rounded-lg p-3 text-center">
                    <div className="text-xs text-gray-500">OOM次数</div>
                    <div className="text-lg font-mono text-red-400">{vram.oom_count || 0}</div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500 text-sm py-4">
                  未检测到GPU显存（当前运行在CPU模式）
                </div>
              )}
            </div>

            {/* Apple/NVIDIA chip info */}
            {current?.apple_silicon && (
              <div className="card">
                <h3 className="text-sm font-medium text-gray-300 mb-3">🍎 Apple Silicon</h3>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                  <div><span className="text-gray-500">芯片:</span> <span className="text-white">{current.apple_silicon.chip_name}</span></div>
                  <div><span className="text-gray-500">GPU核心:</span> <span className="text-white">{current.apple_silicon.gpu_cores}</span></div>
                  <div><span className="text-gray-500">ANE:</span> <span className="text-white">{current.apple_silicon.neural_engine_tops} TOPS</span></div>
                  <div><span className="text-gray-500">内存带宽:</span> <span className="text-white">{current.apple_silicon.memory_bandwidth_gb_s} GB/s</span></div>
                </div>
              </div>
            )}

            {/* Applied optimizations */}
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">已应用的优化</h3>
              {current?.applied_optimizations?.length > 0 ? (
                <div className="space-y-2">
                  {current.applied_optimizations.map((opt: string) => (
                    <div key={opt} className="flex items-center gap-2 bg-surface-900 rounded-lg p-2">
                      <span className="text-green-400">✓</span>
                      <span className="text-sm text-gray-200">{opt}</span>
                    </div>
                  ))}
                  <div className="mt-2 p-2 bg-primary-600/10 rounded-lg text-xs text-primary-400">
                    {current.estimated_vram_savings || ''}
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500 text-sm py-4">
                  当前未应用任何优化。从左侧选择一个预设开始。
                </div>
              )}
            </div>

            {/* Preset comparison table */}
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">预设对比</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead>
                    <tr className="text-gray-500 border-b border-surface-700">
                      <th className="pb-2 pr-4">预设</th>
                      <th className="pb-2 pr-4">兼容</th>
                      <th className="pb-2 pr-4">FlashAttn</th>
                      <th className="pb-2 pr-4">WOQ</th>
                      <th className="pb-2 pr-4">KV卸载</th>
                      <th className="pb-2 pr-4">DMS</th>
                      <th className="pb-2">VRAM</th>
                    </tr>
                  </thead>
                  <tbody>
                    {presets.map((p) => (
                      <tr key={p.name} className="border-b border-surface-800 text-gray-300">
                        <td className="py-2 pr-4 font-medium">{p.name}</td>
                        <td className="py-2 pr-4">{p.is_compatible ? '✅' : '❌'}</td>
                        <td className="py-2 pr-4">{p.flash_attn ? '⚡' : '—'}</td>
                        <td className="py-2 pr-4">{p.woq_bits < 32 ? `${p.woq_desc}` : '-'}</td>
                        <td className="py-2 pr-4">{p.kv_offload ? '✅' : '❌'}</td>
                        <td className="py-2 pr-4">{p.dms ? '✅' : '❌'}</td>
                        <td className="py-2">{p.variable_vram ? '✅' : '❌'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* WOQ tab */}
        {activeTab === 'woq' && (
          <div className="space-y-4">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">WOQ — Weight-Only Quantization</h3>
              <p className="text-xs text-gray-400 mb-4">
                仅量化模型权重，不量化激活值。可大幅降低模型显存占用。
                当前预设: {selectedPreset}
              </p>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                {[
                  { method: 'none', bits: 32, desc: '无量化', mem: '1x', color: 'bg-gray-600' },
                  { method: 'int8', bits: 8, desc: '8-bit bitsandbytes', mem: '4x', color: 'bg-blue-600' },
                  { method: 'nf4', bits: 4, desc: 'NF4 (QLoRA)', mem: '8x', color: 'bg-green-600' },
                  { method: 'int4', bits: 4, desc: '4-bit bitsandbytes', mem: '8x', color: 'bg-teal-600' },
                  { method: 'gptq', bits: 4, desc: 'GPTQ 4-bit', mem: '8x', color: 'bg-purple-600' },
                  { method: 'awq', bits: 4, desc: 'AWQ 4-bit', mem: '8x', color: 'bg-pink-600' },
                  { method: 'gguf_q8', bits: 8, desc: 'GGUF Q8_0', mem: '4x', color: 'bg-amber-600' },
                  { method: 'int2', bits: 2, desc: '2-bit 极限', mem: '16x', color: 'bg-red-600' },
                  { method: 'ane_int8', bits: 8, desc: 'Apple ANE int8', mem: '4x', color: 'bg-sky-600' },
                  { method: 'mlx_fp16', bits: 16, desc: 'MLX float16', mem: '2x', color: 'bg-sky-400' },
                ].map((q) => (
                  <div key={q.method} className="bg-surface-900 rounded-lg p-3 text-center border border-surface-700">
                    <div className={`w-8 h-8 rounded-full ${q.color} mx-auto mb-2 flex items-center justify-center text-white text-xs font-bold`}>
                      {q.bits}
                    </div>
                    <div className="text-sm text-white font-medium">{q.bits}-bit</div>
                    <div className="text-xs text-gray-400 mt-0.5">{q.desc}</div>
                    <div className="text-xs text-primary-400 mt-1">~{q.mem} 显存</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* KV Offload tab */}
        {activeTab === 'kv' && (
          <div className="space-y-4">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">KV Cache Offloading</h3>
              <p className="text-xs text-gray-400 mb-4">
                将KV缓存卸载到CPU内存或磁盘，释放GPU显存用于长序列生成。
                对 Apple Silicon (统一内存) 效果有限—因为CPU/GPU共享内存。
              </p>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
                {[
                  { icon: '💾', title: 'CPU卸载', desc: 'KV缓存 → CPU内存', items: ['Pinned Memory加速', '异步卸载', '可配置阈值'] },
                  { icon: '💿', title: '磁盘卸载', desc: 'KV缓存 → 磁盘文件', items: ['极低内存占用', '持久化缓存', '适合超大上下文'] },
                  { icon: '🔧', title: '压缩传输', desc: '卸载时自动压缩', items: ['FP16压缩 (2x)', 'INT8压缩 (4x)', 'INT4极限 (8x)'] },
                ].map((m) => (
                  <div key={m.title} className="bg-surface-900 rounded-lg p-4 border border-surface-700">
                    <div className="text-lg mb-1">{m.icon}</div>
                    <h4 className="text-sm text-white font-medium">{m.title}</h4>
                    <p className="text-xs text-gray-400 mt-1">{m.desc}</p>
                    <ul className="text-xs text-gray-500 mt-2 space-y-1">
                      {m.items.map((item) => <li key={item}>✓ {item}</li>)}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* DMS tab */}
        {activeTab === 'dms' && (
          <div className="space-y-4">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">DMS — Dynamic Memory Compression</h3>
              <p className="text-xs text-gray-400 mb-4">动态压缩KV缓存和中间激活值。</p>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                {[
                  { method: 'avg_pool', desc: '自适应平均池化', effect: 'KV缓存减50%', icon: '📉' },
                  { method: 'topk', desc: 'Top-K头部选择', effect: '保留最重要头部', icon: '🎯' },
                  { method: 'quantize', desc: 'FP16量化', effect: 'KV缓存减半', icon: '🔢' },
                  { method: 'pruning', desc: '激活值剪枝', effect: '稀疏化中间值', icon: '✂️' },
                ].map((m) => (
                  <div key={m.method} className="bg-surface-900 rounded-lg p-3 border border-surface-700">
                    <div className="text-xl mb-1">{m.icon}</div>
                    <div className="text-sm text-white font-medium">{m.desc}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{m.effect}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* VRAM tab */}
        {activeTab === 'vram' && (
          <div className="space-y-4">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Variable VRAM — 自适应显存管理</h3>
              <p className="text-xs text-gray-400 mb-4">
                根据当前显存使用情况自动调整参数。
                Apple Silicon 使用统一内存，VRAM管理策略自动适配。
              </p>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { name: '自动批次', desc: 'VRAM高时减半batch，低时翻倍' },
                  { name: '自适应序列', desc: '根据显存调整序列长度，M5支持128K' },
                  { name: '精度切换', desc: 'fp32 → fp16 → int8 自动降级' },
                  { name: '梯度检查点', desc: '用计算换显存，节省~50%' },
                  { name: '优化器卸载', desc: '优化器状态 → CPU' },
                  { name: '安全阈值', desc: '可配置显存阈值触发保护' },
                ].map((f) => (
                  <div key={f.name} className="bg-surface-900 rounded-lg p-3 border border-surface-700">
                    <div className="text-sm text-white font-medium">{f.name}</div>
                    <div className="text-xs text-gray-400 mt-0.5">{f.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Flash Attention tab */}
        {activeTab === 'flash' && (
          <div className="space-y-4">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">⚡ Flash Attention</h3>
              <div className={`p-3 rounded-lg mb-4 ${flashAvail ? 'bg-green-600/10 text-green-400' : 'bg-amber-600/10 text-amber-400'} text-sm`}>
                {flashAvail
                  ? '✅ 当前硬件支持 Flash Attention (NVIDIA Turing+ / PyTorch 2.0+)'
                  : '⚠️ 当前硬件不支持 Flash Attention (需要 NVIDIA Turing+ 或 PyTorch 2.0+)'}
              </div>
              <p className="text-xs text-gray-400 mb-4">
                Flash Attention 是一种内存高效的注意力机制实现。它通过分块计算和重计算，
                将注意力机制的时间和空间复杂度从 O(n²) 降低到 O(n)，特别适合长序列训练和推理。
              </p>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
                {[
                  { backend: 'SDPA', icon: '🔷', desc: 'PyTorch 2.0+ 原生', supports: '所有设备', speedup: '1.5-3x' },
                  { backend: 'xformers', icon: '🔶', desc: 'Meta xformers库', supports: 'NVIDIA GPU', speedup: '2-4x' },
                  { backend: 'Triton', icon: '🔷', desc: 'OpenAI Triton', supports: 'NVIDIA GPU', speedup: '2-3x' },
                ].map((b) => (
                  <div key={b.backend} className="bg-surface-900 rounded-lg p-4 border border-surface-700">
                    <div className="text-xl mb-1">{b.icon}</div>
                    <h4 className="text-sm text-white font-medium">{b.backend}</h4>
                    <p className="text-xs text-gray-400 mt-1">{b.desc}</p>
                    <div className="flex gap-2 mt-2 text-xs">
                      <span className="text-gray-500">支持: {b.supports}</span>
                      <span className="text-green-400">加速: {b.speedup}</span>
                    </div>
                  </div>
                ))}
              </div>
              {current?.flash_attn && (
                <div className="mt-4 p-3 bg-purple-600/10 rounded-lg">
                  <span className="text-xs text-purple-400">
                    ⚡ 当前配置: backend={current.flash_attn.backend || 'auto'}
                    {current.flash_attn.use_fp8 ? ', FP8模式' : ''}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
