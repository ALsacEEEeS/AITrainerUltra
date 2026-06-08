import { NODE_DEFINITIONS, NodeDefinition } from './types';

interface NodePaletteProps {
  onDragStart: (def: NodeDefinition) => void;
}

const categoryLabels: Record<string, string> = {
  model: '模型节点',
  data: '数据节点',
  training: '训练节点',
  processing: '处理节点',
  output: '输出节点',
  custom: '自定义节点',
};

export function NodePalette({ onDragStart }: NodePaletteProps) {
  const categories = ['model', 'data', 'training', 'processing', 'output', 'custom'] as const;

  return (
    <div className="w-60 bg-surface-900 border-r border-surface-700 overflow-y-auto shrink-0 flex flex-col">
      <div className="p-3 border-b border-surface-700">
        <h3 className="text-sm font-medium text-gray-300">节点工具箱</h3>
        <p className="text-xs text-gray-500 mt-0.5">拖拽到画布使用 · 端口可连线</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {categories.map((key) => {
          const nodes = NODE_DEFINITIONS.filter((n) => n.category === key);
          if (nodes.length === 0) return null;
          return (
            <div key={key} className="p-2">
              <h4 className="text-xs text-gray-500 uppercase tracking-wider px-2 py-1 flex items-center gap-1">
                {categoryLabels[key] || key}
                <span className="text-[10px] text-gray-600">({nodes.length})</span>
              </h4>
              {nodes.map((def) => (
                <div
                  key={def.type}
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData('text/plain', def.type);
                    onDragStart(def);
                  }}
                  className="flex items-center gap-2 p-2 rounded-lg cursor-grab active:cursor-grabbing
                    hover:bg-surface-800 transition-colors group"
                >
                  <span className="text-lg shrink-0">{def.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-gray-200 group-hover:text-white truncate flex items-center gap-1">
                      {def.label}
                      {def.type === 'custom' && (
                        <span className="text-[10px] px-1 py-0.5 rounded bg-primary-500/20 text-primary-400">NEW</span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500 flex gap-2">
                      {def.inputs.length > 0 && <span>⬅{def.inputs.length}</span>}
                      {def.outputs.length > 0 && <span>➡{def.outputs.length}</span>}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>

      {/* Hint */}
      <div className="p-3 border-t border-surface-700">
        <p className="text-[10px] text-gray-600 leading-relaxed">
          提示: 点击端口拖拽到目标端口创建连线<br />
          右键点击连线删除 · 点击连线选中
        </p>
      </div>
    </div>
  );
}
