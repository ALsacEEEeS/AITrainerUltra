import { useState, useEffect } from 'react';
import { api } from '../../api/client';
import { useLanguage } from '../../i18n/LanguageProvider';

interface DeviceInfo {
  primary: string;
  type: string;
  available_devices: Record<string, boolean>;
  gpu_count: number;
  gpu_name: string;
  memory_gb: number;
  apple_chip?: {
    chip_name: string;
    chip_type: string;
    gpu_cores: number;
    neural_engine_tops: number;
    memory_bandwidth_gb_s: number;
    has_ray_tracing: boolean;
    has_av1_decode: boolean;
    is_m4_or_newer: boolean;
    is_m5_or_newer: boolean;
  };
}

const DEVICE_LABELS: Record<string, string> = {
  auto: 'device.auto',
  cuda: 'device.cuda',
  mps: 'device.mps',
  rocm: 'device.rocm',
  tpu: 'device.tpu',
  npu: 'device.npu',
  xpu: 'device.xpu',
  cpu: 'device.cpu',
};

const DEVICE_COLORS: Record<string, string> = {
  cuda: '#00dc82',
  mps: '#0071e3',
  rocm: '#ed1c24',
  tpu: '#8b5cf6',
  npu: '#f97316',
  xpu: '#0096d6',
  cpu: '#64748b',
};

interface DeviceSelectorProps {
  onDeviceChange?: (device: string) => void;
  defaultDevice?: string;
  showLabel?: boolean;
  compact?: boolean;
}

export function DeviceSelector({
  onDeviceChange,
  defaultDevice = 'auto',
  showLabel = true,
  compact = false,
}: DeviceSelectorProps) {
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [selected, setSelected] = useState(defaultDevice);
  const { t } = useLanguage();

  useEffect(() => {
    api.getStatus().then((res) => {
      setDeviceInfo(res.data?.device);
    }).catch(() => {});
  }, []);

  const handleChange = (value: string) => {
    setSelected(value);
    onDeviceChange?.(value);
  };

  const device = deviceInfo;
  const devType = device?.type || 'cpu';
  const color = DEVICE_COLORS[devType] || '#64748b';

  if (compact) {
    const chipName = device?.apple_chip?.chip_name;
    return (
      <div className="flex items-center gap-2" title={chipName ? `Apple ${chipName}` : device?.primary}>
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-xs text-gray-400 hidden md:inline">
          {chipName ? `🍎 ${chipName}` : device?.type}
        </span>
        <select
          value={selected}
          onChange={(e) => handleChange(e.target.value)}
          className="input text-xs py-1 px-2"
        >
          <option value="auto">{t('device.auto')}</option>
          {device?.available_devices && Object.entries(device.available_devices).map(([k, v]) =>
            v ? <option key={k} value={k}>{DEVICE_LABELS[k] ? t(DEVICE_LABELS[k]) : k}</option> : null
          )}
          <option value="cpu">{t('device.cpu')}</option>
        </select>
      </div>
    );
  }

  return (
    <div className="card">
      {showLabel && (
        <h3 className="text-sm font-medium text-gray-300 mb-3">{t('device.title')}</h3>
      )}

      <div className="flex items-center gap-3 mb-3">
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
        <div className="flex-1">
          <div className="text-sm text-white">{device?.primary || 'Detecting...'}</div>
          {device?.gpu_name && (
            <div className="text-xs text-gray-400">{device.gpu_name}</div>
          )}
        </div>
      </div>

      <select
        value={selected}
        onChange={(e) => handleChange(e.target.value)}
        className="input w-full text-sm mb-3"
      >
        <option value="auto">{t('device.auto')}</option>
        {[
          { value: 'cuda', label: t('device.cuda') },
          { value: 'mps', label: t('device.mps') },
          { value: 'rocm', label: t('device.rocm') },
          { value: 'tpu', label: t('device.tpu') },
          { value: 'npu', label: t('device.npu') },
          { value: 'xpu', label: t('device.xpu') },
          { value: 'cpu', label: t('device.cpu') },
        ].map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>

      {device && (
        <>
          {/* Apple Silicon chip details */}
          {device.apple_chip && (
            <div className="mb-3 bg-surface-900 rounded p-2">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-blue-400 font-medium">Apple {device.apple_chip.chip_name}</span>
                <span className="text-gray-400">{device.apple_chip.is_m5_or_newer ? 'M5 优化' : device.apple_chip.is_m4_or_newer ? 'M4 优化' : 'MPS'}</span>
              </div>
              <div className="grid grid-cols-4 gap-1 text-[10px] text-gray-400">
                <div>
                  <div className="text-gray-500">GPU</div>
                  <div className="text-white">{device.apple_chip.gpu_cores}核</div>
                </div>
                <div>
                  <div className="text-gray-500">ANE</div>
                  <div className="text-white">{device.apple_chip.neural_engine_tops} TOPS</div>
                </div>
                <div>
                  <div className="text-gray-500">内存带宽</div>
                  <div className="text-white">{device.apple_chip.memory_bandwidth_gb_s}GB/s</div>
                </div>
                <div>
                  <div className="text-gray-500">光追</div>
                  <div className="text-white">{device.apple_chip.has_ray_tracing ? '✓' : '✗'}</div>
                </div>
              </div>
            </div>
          )}
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-surface-900 rounded p-2">
              <div className="text-[10px] text-gray-500">{t('device.used')}</div>
              <div className="text-sm font-mono text-yellow-400">{device.memory_gb > 0 ? 'N/A' : '-'}</div>
            </div>
            <div className="bg-surface-900 rounded p-2">
              <div className="text-[10px] text-gray-500">{t('device.available')}</div>
              <div className="text-sm font-mono text-green-400">
                {device.type === 'mps' && device.apple_chip ? `${device.apple_chip.gpu_cores}` : device.gpu_count > 0 ? `${device.gpu_count}` : '-'}
              </div>
            </div>
            <div className="bg-surface-900 rounded p-2">
              <div className="text-[10px] text-gray-500">{t('device.total')}</div>
              <div className="text-sm font-mono text-primary-400">{device.memory_gb > 0 ? `${device.memory_gb.toFixed(1)}GB` : '-'}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
