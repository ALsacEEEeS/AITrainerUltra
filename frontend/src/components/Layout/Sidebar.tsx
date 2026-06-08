import { useTrainingStore } from '../../store/useTrainingStore';
import { useLanguage } from '../../i18n/LanguageProvider';
import type { Tab } from '../../App';

interface SidebarProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
}

const navItems: { id: Tab; labelKey: string; icon: string }[] = [
  { id: 'workflow', labelKey: 'nav.workflow', icon: '⚡' },
  { id: 'chat', labelKey: 'nav.chat', icon: '💬' },
  { id: 'training', labelKey: 'nav.training', icon: '🎯' },
  { id: 'optimization', labelKey: 'nav.optimization', icon: '⚡' },
  { id: 'multimodal', labelKey: 'nav.multimodal', icon: '🖼️' },
  { id: 'loader', labelKey: 'nav.loader', icon: '📂' },
  { id: 'experiments', labelKey: 'nav.experiments', icon: '🔬' },
  { id: 'models', labelKey: 'nav.models', icon: '🗂️' },
  { id: 'recipes', labelKey: 'nav.recipes', icon: '📋' },
  { id: 'templates', labelKey: 'nav.templates', icon: '📋' },
  { id: 'help', labelKey: 'nav.help', icon: '❓' },
];

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  const isRunning = useTrainingStore((s) => s.isRunning);
  const { t, lang, setLang } = useLanguage();

  return (
    <aside className="w-16 lg:w-56 bg-surface-900 border-r border-surface-700 flex flex-col shrink-0">
      {/* Logo */}
      <div className="p-3 lg:p-4 border-b border-surface-700">
        <h1 className="hidden lg:block text-lg font-bold bg-gradient-to-r from-primary-400 to-purple-400 bg-clip-text text-transparent">
          AITrainerUltra
        </h1>
        <span className="lg:hidden text-xl text-center block">🧠</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onTabChange(item.id)}
            className={`w-full flex items-center gap-3 px-3 lg:px-4 py-3 transition-colors ${
              activeTab === item.id
                ? 'bg-primary-600/20 text-primary-400 border-r-2 border-primary-500'
                : 'text-gray-400 hover:bg-surface-800 hover:text-white'
            }`}
          >
            <span className="text-xl">{item.icon}</span>
            <span className="hidden lg:block text-sm font-medium">{t(item.labelKey)}</span>
          </button>
        ))}
      </nav>

      {/* Status + Language Toggle */}
      <div className="p-3 lg:p-4 border-t border-surface-700 space-y-2">
        {/* Training status */}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
            }`}
          />
          <span className="hidden lg:block text-xs text-gray-400">
            {isRunning ? t('nav.training_in_progress') : t('nav.ready')}
          </span>
        </div>

        {/* Language toggle */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setLang('zh')}
            className={`text-xs px-2 py-0.5 rounded transition-colors ${
              lang === 'zh' ? 'bg-primary-600 text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            中文
          </button>
          <button
            onClick={() => setLang('en')}
            className={`text-xs px-2 py-0.5 rounded transition-colors ${
              lang === 'en' ? 'bg-primary-600 text-white' : 'text-gray-500 hover:text-gray-300'
            }`}
          >
            EN
          </button>
        </div>
      </div>
    </aside>
  );
}
