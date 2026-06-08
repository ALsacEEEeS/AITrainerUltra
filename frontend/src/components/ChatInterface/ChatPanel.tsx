import { useState, useEffect, useRef } from 'react';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { ModelSelector } from './ModelSelector';
import { DeviceSelector } from '../Device/DeviceSelector';
import { api } from '../../api/client';
import { useModelStore } from '../../store/useModelStore';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'system',
      content: 'Welcome to AITrainerUltra Chat! Select a model and device, then start chatting.',
      timestamp: Date.now(),
    },
  ]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [inferenceDevice, setInferenceDevice] = useState('auto');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [streamingContent, setStreamingContent] = useState('');

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  useEffect(() => {
    api.listPresets().then((res) => {
      if (res.data) useModelStore.getState().setPresets(res.data);
    }).catch(() => {});
  }, []);

  const handleSend = async (content: string) => {
    const userMsg: Message = { role: 'user', content, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setIsStreaming(true);
    setStreamingContent('');

    const modelId = selectedModel || 'TinyLlama/TinyLlama-1.1B-Chat-v1.0';
    const payload = {
      model_id: modelId,
      device: inferenceDevice,
      temperature,
      max_tokens: maxTokens,
      messages: [...messages, userMsg].map((m) => ({ role: m.role, content: m.content })),
    };

    // Try streaming first
    try {
      const controller = api.chatStream(
        payload,
        (token, fullText) => {
          setStreamingContent(fullText);
        },
        (fullText) => {
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: fullText, timestamp: Date.now() },
          ]);
          setStreamingContent('');
          setIsLoading(false);
          setIsStreaming(false);
        },
        (error) => {
          // Fall back to non-streaming on stream error
          fallbackChat(payload, userMsg);
        },
      );
      abortRef.current = controller;
    } catch {
      fallbackChat(payload, userMsg);
    }
  };

  const fallbackChat = async (payload: any, userMsg: Message) => {
    setIsStreaming(false);
    try {
      const res = await api.chat(payload);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.data?.response || '(empty response)',
          timestamp: Date.now(),
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          content: `Error: ${err.message}. Try loading a real model first.`,
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleStop = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    // Save whatever was streamed so far
    if (streamingContent) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: streamingContent + ' ✋ (stopped)', timestamp: Date.now() },
      ]);
      setStreamingContent('');
    }
    setIsLoading(false);
    setIsStreaming(false);
  };

  const handleClear = () => {
    setMessages([{ role: 'system', content: 'Chat cleared.', timestamp: Date.now() }]);
  };

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-1">
          {messages.map((msg, i) => <MessageBubble key={i} message={msg} />)}
          {/* Streaming content */}
          {isStreaming && streamingContent && (
            <div className="flex gap-3 mb-4">
              <div className="w-8 h-8 rounded-full bg-primary-600 flex items-center justify-center text-sm shrink-0">AI</div>
              <div className="bg-surface-800 rounded-2xl rounded-tl-sm px-4 py-3 border border-surface-700 max-w-[80%]">
                <p className="text-sm text-gray-200 whitespace-pre-wrap">{streamingContent}</p>
                <span className="inline-block w-2 h-4 bg-primary-400 animate-pulse ml-0.5" />
              </div>
            </div>
          )}
          {isLoading && !streamingContent && (
            <div className="flex gap-3 mb-4">
              <div className="w-8 h-8 rounded-full bg-surface-700 flex items-center justify-center text-sm shrink-0">AI</div>
              <div className="bg-surface-800 rounded-2xl rounded-tl-sm px-4 py-3 border border-surface-700">
                <div className="flex gap-1.5">
                  <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="relative">
          {isStreaming && (
            <div className="absolute -top-8 left-1/2 -translate-x-1/2 z-10">
              <button onClick={handleStop} className="bg-red-600 hover:bg-red-700 text-white text-xs py-1 px-3 rounded-full shadow-lg transition-colors">
                ⏹ 停止生成
              </button>
            </div>
          )}
          <ChatInput onSend={handleSend} disabled={isLoading && !isStreaming} />
        </div>
      </div>

      {/* Side panel */}
      <div className="w-64 bg-surface-900 border-l border-surface-700 flex flex-col">
        <ModelSelector selected={selectedModel} onSelect={setSelectedModel} />

        {/* Inference Device Selector */}
        <div className="p-3 border-b border-surface-700">
          <DeviceSelector
            compact={false}
            defaultDevice={inferenceDevice}
            onDeviceChange={setInferenceDevice}
          />
        </div>

        <div className="p-3 border-b border-surface-700">
          <label className="text-xs text-gray-400 block mb-1.5">Sampling</label>
          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Temperature</span>
                <span className="text-gray-300">{temperature.toFixed(1)}</span>
              </div>
              <input type="range" min="0" max="2" step="0.1" value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full accent-primary-500" />
            </div>
            <div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Max Tokens</span>
                <span className="text-gray-300">{maxTokens}</span>
              </div>
              <input type="range" min="64" max="2048" step="64" value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full accent-primary-500" />
            </div>
          </div>
        </div>

        <div className="p-3 mt-auto">
          <button onClick={handleClear} className="btn-secondary text-xs w-full py-1.5">Clear Chat</button>
          <p className="text-[10px] text-gray-600 text-center mt-2">
            {messages.length} msgs · {selectedModel?.split('/').pop() || 'no model'}
          </p>
        </div>
      </div>
    </div>
  );
}
