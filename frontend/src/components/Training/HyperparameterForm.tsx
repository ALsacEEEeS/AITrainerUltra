import { useTrainingStore } from '../../store/useTrainingStore';

export function HyperparameterForm() {
  const { config, setConfig } = useTrainingStore();
  const hp = config.hyperparameters || {};

  const updateHP = (key: string, value: any) => {
    setConfig({
      ...config,
      hyperparameters: { ...hp, [key]: value },
    });
  };

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-gray-300">超参数设置</h4>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-gray-400 block mb-1">学习率</label>
          <input
            type="number"
            value={hp.learning_rate ?? 5e-5}
            onChange={(e) => updateHP('learning_rate', parseFloat(e.target.value))}
            className="input text-sm w-full"
            step="1e-6"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">批次大小</label>
          <input
            type="number"
            value={hp.batch_size ?? 8}
            onChange={(e) => updateHP('batch_size', parseInt(e.target.value))}
            className="input text-sm w-full"
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">训练轮数</label>
          <input
            type="number"
            value={hp.num_epochs ?? 3}
            onChange={(e) => updateHP('num_epochs', parseInt(e.target.value))}
            className="input text-sm w-full"
            min={1}
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">预热步数</label>
          <input
            type="number"
            value={hp.warmup_steps ?? 100}
            onChange={(e) => updateHP('warmup_steps', parseInt(e.target.value))}
            className="input text-sm w-full"
            min={0}
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">权重衰减</label>
          <input
            type="number"
            value={hp.weight_decay ?? 0.01}
            onChange={(e) => updateHP('weight_decay', parseFloat(e.target.value))}
            className="input text-sm w-full"
            step="0.001"
          />
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">梯度裁剪</label>
          <input
            type="number"
            value={hp.max_grad_norm ?? 1.0}
            onChange={(e) => updateHP('max_grad_norm', parseFloat(e.target.value))}
            className="input text-sm w-full"
            step="0.1"
          />
        </div>
      </div>
    </div>
  );
}
