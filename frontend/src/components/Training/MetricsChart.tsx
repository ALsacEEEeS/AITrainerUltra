import { useTrainingStore } from '../../store/useTrainingStore';

export function MetricsChart() {
  const { history, metrics } = useTrainingStore();

  if (history.length === 0) {
    return (
      <div className="card h-48 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-2xl mb-2">📊</div>
          <p className="text-sm">等待训练数据...</p>
          <p className="text-xs mt-1">开始训练后将显示实时指标</p>
        </div>
      </div>
    );
  }

  const maxLoss = Math.max(...history.map((h) => h.loss), 0.1);
  const minLoss = Math.min(...history.map((h) => h.loss), 0);
  const range = maxLoss - minLoss || 1;
  const width = 600;
  const height = 160;
  const padding = { top: 10, right: 10, bottom: 20, left: 40 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const points = history.map((h, i) => {
    const x = padding.left + (i / Math.max(history.length - 1, 1)) * chartW;
    const y = padding.top + (1 - (h.loss - minLoss) / range) * chartH;
    return `${x},${y}`;
  });

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-300">训练损失</h4>
        <div className="flex gap-3 text-xs">
          {Object.entries(metrics).slice(0, 3).map(([key, val]) => (
            <span key={key} className="text-gray-400">
              {key}: <span className="text-primary-400 font-mono">{typeof val === 'number' ? val.toFixed(4) : val}</span>
            </span>
          ))}
        </div>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-40">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((frac) => {
          const y = padding.top + frac * chartH;
          const val = maxLoss - frac * range;
          return (
            <g key={frac}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y}
                stroke="#334155" strokeWidth="0.5" />
              <text x={padding.left - 5} y={y + 3} textAnchor="end"
                className="fill-gray-600 text-[8px]">
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Loss line */}
        {points.length > 1 && (
          <polyline
            points={points.join(' ')}
            fill="none"
            stroke="#6366f1"
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}

        {/* Data points */}
        {points.map((pt, i) => {
          const [x, y] = pt.split(',').map(Number);
          return (
            <circle key={i} cx={x} cy={y} r="2" fill="#818cf8" />
          );
        })}
      </svg>

      <div className="flex justify-between text-[10px] text-gray-600 mt-1">
        <span>Step 0</span>
        <span>Step {Math.max(history.length - 1, 0)}</span>
      </div>
    </div>
  );
}
