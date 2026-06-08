import { useState, useEffect } from 'react';
import { api } from '../../api/client';

interface Checkpoint {
  name: string;
  path: string;
  size_bytes: number;
  size_readable: string;
  modified: string;
}

export function CheckpointList({ outputDir }: { outputDir: string }) {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetch = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listCheckpoints(outputDir);
      setCheckpoints(res.data?.checkpoints || []);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  useEffect(() => { fetch(); }, [outputDir]);

  const handleDelete = async (name: string) => {
    if (!confirm(`确定删除检查点 ${name}？`)) return;
    try {
      await api.deleteCheckpoint(name, outputDir);
      setCheckpoints((prev) => prev.filter((c) => c.name !== name));
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleRestore = async (name: string) => {
    setRestoring(name);
    setError(null);
    try {
      await api.restoreCheckpoint(name, outputDir);
    } catch (err: any) {
      setError(err.message);
    }
    setRestoring(null);
  };

  if (loading) {
    return <div className="text-center text-gray-500 text-xs py-4">加载中...</div>;
  }

  if (error) {
    return (
      <div className="text-center">
        <p className="text-xs text-red-400">{error}</p>
        <button onClick={fetch} className="text-xs text-primary-400 mt-1">重试</button>
      </div>
    );
  }

  if (checkpoints.length === 0) {
    return (
      <div className="text-center text-gray-500 text-xs py-4">
        暂无检查点<br />
        <span className="text-[10px] text-gray-600">训练完成后自动生成</span>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {checkpoints.map((ckpt) => (
        <div key={ckpt.name} className="bg-surface-800 rounded-lg p-2 flex items-center gap-2">
          <span className="text-xs text-gray-400 shrink-0">📄</span>
          <div className="flex-1 min-w-0">
            <div className="text-xs text-gray-200 truncate">{ckpt.name}</div>
            <div className="text-[10px] text-gray-500">{ckpt.size_readable}</div>
          </div>
          <button
            onClick={() => handleRestore(ckpt.name)}
            disabled={restoring === ckpt.name}
            className="text-[10px] px-1.5 py-0.5 rounded bg-primary-600/20 text-primary-300 hover:bg-primary-600/30"
          >
            {restoring === ckpt.name ? '...' : '恢复'}
          </button>
          <button
            onClick={() => handleDelete(ckpt.name)}
            className="text-[10px] px-1.5 py-0.5 rounded bg-red-600/20 text-red-300 hover:bg-red-600/30"
          >
            删除
          </button>
        </div>
      ))}
    </div>
  );
}
