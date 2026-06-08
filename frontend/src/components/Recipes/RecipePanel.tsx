import { useState, useEffect } from 'react';
import { api } from '../../api/client';
import { useTrainingStore } from '../../store/useTrainingStore';

interface Recipe {
  name: string;
  description: string;
  model_type: string;
  model_name: string;
  hardware: string;
  estimated_time: string;
  tags: string[];
  hyperparameters: Record<string, any>;
  apple_silicon_note?: string;
}

const TAG_COLORS: Record<string, string> = {
  llm: 'bg-blue-600',
  gpt: 'bg-indigo-600',
  bert: 'bg-emerald-600',
  moe: 'bg-fuchsia-600',
  cnn: 'bg-cyan-600',
  scratch: 'bg-green-600',
  clip: 'bg-pink-600',
  lora: 'bg-amber-600',
  qlora: 'bg-orange-600',
  '4-bit': 'bg-red-600',
  beginner: 'bg-teal-600',
  efficient: 'bg-violet-600',
  nlp: 'bg-sky-600',
  vision: 'bg-rose-600',
};

export function RecipePanel({ onNavigate }: { onNavigate?: (tab: string) => void }) {
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [selected, setSelected] = useState<Recipe | null>(null);
  const [filterTag, setFilterTag] = useState('');
  const [detail, setDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const setConfig = useTrainingStore((s) => s.setConfig);

  useEffect(() => {
    api.listRecipes(filterTag).then((res) => {
      setRecipes(res.data?.recipes || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [filterTag]);

  const handleSelect = async (recipe: Recipe) => {
    setSelected(recipe);
    try {
      const res = await api.getRecipe(recipe.name);
      setDetail(res.data);
    } catch { setDetail(null); }
  };

  const tags = [...new Set(recipes.flatMap(r => r.tags))];

  return (
    <div className="flex h-full">
      <div className="w-80 bg-surface-900 border-r border-surface-700 flex flex-col">
        <div className="p-3 border-b border-surface-700">
          <h3 className="text-sm font-medium text-gray-300">Training Recipes</h3>
          <p className="text-xs text-gray-500 mt-0.5">Pre-configured training configs</p>
        </div>

        {/* Tag filter */}
        <div className="p-2 border-b border-surface-700 flex flex-wrap gap-1">
          <button onClick={() => setFilterTag('')}
            className={`text-xs px-2 py-0.5 rounded ${!filterTag ? 'bg-primary-600 text-white' : 'bg-surface-800 text-gray-400'}`}>
            All
          </button>
          {tags.map(tag => (
            <button key={tag} onClick={() => setFilterTag(tag)}
              className={`text-xs px-2 py-0.5 rounded ${filterTag === tag ? 'bg-primary-600 text-white' : 'bg-surface-800 text-gray-400'}`}>
              {tag}
            </button>
          ))}
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {loading ? (
            <div className="text-center text-gray-500 text-sm p-4">Loading...</div>
          ) : recipes.map(r => (
            <div key={r.name} onClick={() => handleSelect(r)}
              className={`p-3 rounded-lg cursor-pointer mb-1.5 transition-all ${
                selected?.name === r.name
                  ? 'bg-primary-600/20 border border-primary-500/30'
                  : 'hover:bg-surface-800 border border-transparent'
              }`}>
              <div className="text-sm text-white font-medium">{r.name}</div>
              <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{r.description}</p>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="text-[10px] text-gray-500">{r.model_type}</span>
                <span className="text-[10px] text-gray-500">|</span>
                <span className="text-[10px] text-primary-400">{r.estimated_time}</span>
              </div>
              <div className="flex gap-1 mt-1.5 flex-wrap">
                {r.tags.map(tag => (
                  <span key={tag} className={`badge ${TAG_COLORS[tag] || 'bg-surface-700'} text-white text-[10px]`}>
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detail */}
      <div className="flex-1 overflow-y-auto p-4">
        {!selected ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-4xl mb-3">📋</div>
              <p className="text-lg">Select a training recipe</p>
              <p className="text-sm mt-1">Pre-configured for different models and hardware</p>
            </div>
          </div>
        ) : (
          <div className="max-w-2xl mx-auto space-y-4">
            <div className="card">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-lg font-medium text-white">{selected.name}</h2>
                  <p className="text-sm text-gray-400 mt-1">{selected.description}</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mt-3">
                {selected.tags.map(tag => (
                  <span key={tag} className={`badge ${TAG_COLORS[tag] || 'bg-surface-700'} text-white`}>{tag}</span>
                ))}
              </div>
            </div>

            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Config</h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="text-xs text-gray-500">Model Type</span>
                  <p className="text-sm text-white">{selected.model_type}</p>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Model Name</span>
                  <p className="text-sm text-white font-mono">{selected.model_name || '(from scratch)'}</p>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Hardware</span>
                  <p className="text-sm text-primary-400">{selected.hardware}</p>
                </div>
                <div>
                  <span className="text-xs text-gray-500">Est. Time</span>
                  <p className="text-sm text-primary-400">{selected.estimated_time}</p>
                </div>
              </div>
            </div>

            {selected.apple_silicon_note && (
              <div className="card border-sky-500/30 bg-sky-600/5">
                <h3 className="text-sm font-medium text-sky-300 mb-2">🍎 Apple Silicon 优化</h3>
                <p className="text-xs text-sky-200/80">{selected.apple_silicon_note}</p>
              </div>
            )}

            <div className="card">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Hyperparameters</h3>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
                {Object.entries(selected.hyperparameters).map(([k, v]) => (
                  <div key={k} className="bg-surface-900 rounded p-2">
                    <div className="text-[10px] text-gray-500">{k}</div>
                    <div className="text-sm font-mono text-white">{String(v)}</div>
                  </div>
                ))}
              </div>
            </div>

            {detail?.config && (
              <div className="card">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Full Config</h3>
                <pre className="text-xs text-gray-400 bg-surface-900 rounded-lg p-3 overflow-x-auto">
                  {JSON.stringify(detail.config, null, 2)}
                </pre>
              </div>
            )}

            <button className="btn-primary w-full" onClick={() => {
              // Load recipe config into training store and navigate
              setConfig({
                model_type: selected.model_type,
                model_name: selected.model_name,
                task: detail?.config?.task || 'text-generation',
                output_dir: detail?.config?.output_dir || './output',
                hyperparameters: selected.hyperparameters,
                dataset: detail?.config?.dataset || {},
              });
              if (onNavigate) {
                onNavigate('training');
              } else {
                window.location.hash = '#training';
              }
            }}>
              🚀 Use This Recipe
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
