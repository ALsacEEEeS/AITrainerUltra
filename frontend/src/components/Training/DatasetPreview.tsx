import { useState } from 'react';
import { api } from '../../api/client';

interface DatasetPreviewProps {
  datasetPath: string;
  modelType: string;
  onClose?: () => void;
}

export function DatasetPreview({ datasetPath, modelType, onClose }: DatasetPreviewProps) {
  const [preview, setPreview] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePreview = async () => {
    if (!datasetPath.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.previewDataset({
        path: datasetPath,
        model_type: modelType,
        split: 'train',
        max_samples: 5,
      });
      setPreview(res.data);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  if (!datasetPath) return null;

  return (
    <div className="card border-surface-700">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-medium text-gray-300">
          📊 数据集预览: <span className="text-primary-400">{datasetPath.split('/').pop()}</span>
        </h4>
        <div className="flex gap-1">
          <button onClick={handlePreview} disabled={loading}
            className="text-xs px-2 py-0.5 rounded bg-primary-600/20 text-primary-300 hover:bg-primary-600/30">
            {loading ? '加载中...' : '预览'}
          </button>
          {onClose && (
            <button onClick={onClose} className="text-xs text-gray-500 hover:text-white">✕</button>
          )}
        </div>
      </div>

      {error && <p className="text-xs text-red-400 mb-2">{error}</p>}

      {preview && (
        <div className="space-y-1">
          {preview.total_rows > 0 && (
            <p className="text-[10px] text-gray-500">
              共 {preview.total_rows.toLocaleString()} 条记录 · {preview.columns?.join(', ')}
            </p>
          )}
          {preview.samples?.map((row: any, i: number) => (
            <div key={i} className="bg-surface-900 rounded p-2 text-xs">
              {preview.columns?.map((col: string) => (
                <div key={col} className="text-gray-400">
                  <span className="text-gray-600">{col}: </span>
                  <span className="text-gray-200">{String(row[col] ?? '')}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
