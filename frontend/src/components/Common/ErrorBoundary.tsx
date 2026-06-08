import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="flex items-center justify-center h-full bg-surface-950 p-8">
            <div className="text-center max-w-md">
              <div className="text-4xl mb-4">💥</div>
              <h2 className="text-lg font-medium text-white mb-2">面板发生错误</h2>
              <p className="text-sm text-gray-400 mb-4 font-mono bg-surface-900 rounded-lg p-3 text-left">
                {this.state.error?.message || 'Unknown error'}
              </p>
              <button
                onClick={this.handleReset}
                className="btn-primary text-sm"
              >
                重试
              </button>
            </div>
          </div>
        )
      );
    }

    return this.props.children;
  }
}
