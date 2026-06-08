import { useState, useEffect, lazy, Suspense } from 'react';
import { LanguageProvider } from './i18n/LanguageProvider';
import { Sidebar } from './components/Layout/Sidebar';
import { TopBar } from './components/Layout/TopBar';
import { ErrorBoundary } from './components/Common/ErrorBoundary';
import { api } from './api/client';

// Lazy-loaded panels for code splitting
const NodeCanvas = lazy(() => import('./components/NodeEditor/NodeCanvas').then(m => ({ default: m.NodeCanvas })));
const ChatPanel = lazy(() => import('./components/ChatInterface/ChatPanel').then(m => ({ default: m.ChatPanel })));
const TrainingPanel = lazy(() => import('./components/Training/TrainingPanel').then(m => ({ default: m.TrainingPanel })));
const ExperimentsPanel = lazy(() => import('./components/Experiments/ExperimentsPanel').then(m => ({ default: m.ExperimentsPanel })));
const ModelManagerPanel = lazy(() => import('./components/ModelManager/ModelManagerPanel').then(m => ({ default: m.ModelManagerPanel })));
const PipelineTemplates = lazy(() => import('./components/PipelineTemplates/PipelineTemplates').then(m => ({ default: m.PipelineTemplates })));
const MultimodalPanel = lazy(() => import('./components/Multimodal/MultimodalPanel').then(m => ({ default: m.MultimodalPanel })));
const OptimizationPanel = lazy(() => import('./components/Optimization/OptimizationPanel').then(m => ({ default: m.OptimizationPanel })));
const ModelLoaderPanel = lazy(() => import('./components/ModelLoader/ModelLoaderPanel').then(m => ({ default: m.ModelLoaderPanel })));
const RecipePanel = lazy(() => import('./components/Recipes/RecipePanel').then(m => ({ default: m.RecipePanel })));
const HelpPanel = lazy(() => import('./components/Help/HelpPanel').then(m => ({ default: m.HelpPanel })));

function PanelLoader({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-full bg-surface-950">
        <div className="text-center">
          <div className="animate-spin text-2xl mb-2">⏳</div>
          <p className="text-sm text-gray-500">加载中...</p>
        </div>
      </div>
    }>
      {children}
    </Suspense>
  );
}

export type Tab = 'workflow' | 'chat' | 'training' | 'experiments' | 'models' | 'templates' | 'multimodal' | 'optimization' | 'loader' | 'recipes' | 'help';

function AppContent() {
  const [activeTab, setActiveTab] = useState<Tab>('workflow');
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    api.getStatus().then((res) => setStatus(res.data)).catch(console.error);
  }, []);

  const renderContent = () => {
    const panel = (() => {
      switch (activeTab) {
        case 'workflow': return <NodeCanvas />;
        case 'chat': return <ChatPanel />;
        case 'training': return <TrainingPanel />;
        case 'experiments': return <ExperimentsPanel />;
        case 'models': return <ModelManagerPanel />;
        case 'templates': return <PipelineTemplates onNavigate={setActiveTab} />;
        case 'multimodal': return <MultimodalPanel />;
        case 'optimization': return <OptimizationPanel />;
        case 'loader': return <ModelLoaderPanel />;
        case 'recipes': return <RecipePanel onNavigate={(tab) => setActiveTab(tab as Tab)} />;
        case 'help': return <HelpPanel />;
        default: return <NodeCanvas />;
      }
    })();
    return <ErrorBoundary key={activeTab}><PanelLoader>{panel}</PanelLoader></ErrorBoundary>;
  };

  return (
    <div className="flex h-screen bg-surface-950 text-white overflow-hidden">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 flex flex-col">
        <TopBar status={status} tab={activeTab} />
        <main className="flex-1 overflow-hidden">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}

export default App;
