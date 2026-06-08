import { useCallback, useRef } from 'react';
import { NODE_DEFINITIONS } from './types';

export interface PortEvent {
  nodeId: string;
  portId: string;
  portType: 'input' | 'output';
  rect: DOMRect;
}

interface NodeComponentProps {
  id: string;
  type: string;
  selected: boolean;
  config: Record<string, any>;
  onConfigChange: (id: string, config: Record<string, any>) => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onPortMouseDown: (event: PortEvent, e: React.MouseEvent) => void;
}

export function NodeComponent({
  id,
  type,
  selected,
  config,
  onConfigChange,
  onSelect,
  onDelete,
  onPortMouseDown,
}: NodeComponentProps) {
  const def = NODE_DEFINITIONS.find((d) => d.type === type);
  const nodeRef = useRef<HTMLDivElement>(null);
  if (!def) return null;

  const handlePortMouseDown = useCallback(
    (portId: string, portType: 'input' | 'output', e: React.MouseEvent) => {
      e.stopPropagation();
      const portEl = e.currentTarget as HTMLElement;
      const rect = portEl.getBoundingClientRect();
      onPortMouseDown({ nodeId: id, portId, portType, rect }, e);
    },
    [id, onPortMouseDown],
  );

  const renderField = (field: typeof def.configFields[0]) => {
    const value = config[field.name] ?? field.defaultValue;
    switch (field.type) {
      case 'number':
        return (
          <input
            type="number"
            value={value}
            onChange={(e) => onConfigChange(id, { ...config, [field.name]: parseFloat(e.target.value) })}
            className="input text-xs py-1 flex-1"
            onClick={(e) => e.stopPropagation()}
          />
        );
      case 'select':
        return (
          <select
            value={value}
            onChange={(e) => onConfigChange(id, { ...config, [field.name]: e.target.value })}
            className="input text-xs py-1 flex-1"
            onClick={(e) => e.stopPropagation()}
          >
            {(field.options || []).map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        );
      case 'boolean':
        return (
          <input
            type="checkbox"
            checked={!!value}
            onChange={(e) => onConfigChange(id, { ...config, [field.name]: e.target.checked })}
            className="toggle"
            onClick={(e) => e.stopPropagation()}
          />
        );
      case 'code':
        return (
          <textarea
            value={value}
            onChange={(e) => onConfigChange(id, { ...config, [field.name]: e.target.value })}
            className="input text-xs py-1 flex-1 font-mono"
            rows={4}
            onClick={(e) => e.stopPropagation()}
            placeholder={field.placeholder}
          />
        );
      default:
        return (
          <input
            type="text"
            value={value}
            onChange={(e) => onConfigChange(id, { ...config, [field.name]: e.target.value })}
            className="input text-xs py-1 flex-1"
            onClick={(e) => e.stopPropagation()}
            placeholder={field.placeholder}
          />
        );
    }
  };

  return (
    <div
      ref={nodeRef}
      className={`absolute bg-surface-800 rounded-xl border-2 shadow-xl w-72 ${
        selected ? 'border-primary-500 shadow-primary-500/20' : 'border-surface-700'
      }`}
      style={{ zIndex: selected ? 10 : 1 }}
      onClick={() => onSelect(id)}
    >
      {/* Header */}
      <div
        className="flex items-center gap-2 p-3 rounded-t-xl cursor-move"
        style={{ backgroundColor: `${def.color}20` }}
      >
        <span className="text-lg">{def.icon}</span>
        <span className="text-sm font-medium text-white flex-1 truncate">{def.label}</span>
        {type === 'custom' && config.node_name && (
          <span className="text-xs text-gray-400 truncate max-w-[80px]">{config.node_name}</span>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(id); }}
          className="text-gray-500 hover:text-red-400 transition-colors text-lg leading-none ml-1"
          title="删除节点"
        >
          ×
        </button>
      </div>

      {/* Input/output ports — interactive */}
      <div className="flex justify-between px-3 py-2">
        <div className="flex flex-col gap-2">
          {def.inputs.map((input, idx) => (
            <div key={input.id} className="flex items-center gap-1.5 group relative">
              <div
                className="w-3 h-3 rounded-full bg-primary-500 border-2 border-surface-900 cursor-crosshair
                  hover:scale-125 transition-transform"
                onMouseDown={(e) => handlePortMouseDown(input.id, 'input', e)}
                title={`输入: ${input.label}`}
              />
              <span className="text-xs text-gray-400">{input.label}</span>
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-2 items-end">
          {def.outputs.map((output, idx) => (
            <div key={output.id} className="flex items-center gap-1.5 group relative">
              <span className="text-xs text-gray-400">{output.label}</span>
              <div
                className="w-3 h-3 rounded-full bg-green-500 border-2 border-surface-900 cursor-crosshair
                  hover:scale-125 transition-transform"
                onMouseDown={(e) => handlePortMouseDown(output.id, 'output', e)}
                title={`输出: ${output.label}`}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Config fields */}
      {def.configFields.length > 0 && (
        <div className="px-3 pb-3 space-y-2 max-h-60 overflow-y-auto">
          <div className="h-px bg-surface-700" />
          {def.configFields.map((field) => (
            <div key={field.name} className="flex items-center gap-2">
              <label className="text-xs text-gray-400 w-24 shrink-0 truncate" title={field.label}>
                {field.label}
              </label>
              {renderField(field)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
