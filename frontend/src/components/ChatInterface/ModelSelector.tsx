import { useState } from 'react';
import { useModelStore } from '../../store/useModelStore';
import { api } from '../../api/client';

interface ModelSelectorProps {
  selected: string | null;
  onSelect: (name: string) => void;
}

export function ModelSelector({ selected, onSelect }: ModelSelectorProps) {
  const { models, presets } = useModelStore();
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const handleLoadModel = async () => {
    if (!selected) return;
    setLoading(true);
    setLoadError(null);
    try {
      await api.loadModelFromPath({ path: selected, device: 'auto' });
    } catch (err: any) {
      setLoadError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="p-3 border-b border-surface-700">
      <label className="text-xs text-gray-400 block mb-1.5">选择模型</label>
      <select
        value={selected || ''}
        onChange={(e) => onSelect(e.target.value)}
        className="input text-sm w-full"
      >
        <option value="" disabled>-- 选择模型加载对话 --</option>
        <optgroup label="预设模型">
          {Object.entries(presets).map(([key, val]) => (
            <option key={key} value={val.model_name || key}>
              {key} - {val.model_type}
            </option>
          ))}
        </optgroup>
        <optgroup label="🏗️ 从零训练模型">
          <option value="scratch-transformer">Scratch Transformer (语言模型)</option>
          <option value="scratch-lstm">Scratch LSTM (文本生成)</option>
        </optgroup>
        <optgroup label="已注册模型">
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name.split('/').pop()} ({m.type})
            </option>
          ))}
        </optgroup>
      </select>

      {selected && (
        <button
          onClick={handleLoadModel}
          disabled={loading}
          className="btn-primary text-xs py-1 px-2 mt-2 w-full"
        >
          {loading ? '⏳ 加载中...' : '加载模型到对话'}
        </button>
      )}
      {loadError && (
        <p className="text-xs text-red-400 mt-1">{loadError}</p>
      )}
    </div>
  );
}
