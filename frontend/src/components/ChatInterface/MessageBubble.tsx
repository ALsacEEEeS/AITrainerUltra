interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: number;
}

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''} mb-4`}>
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center text-sm shrink-0 ${
          isUser
            ? 'bg-primary-600 text-white'
            : isSystem
              ? 'bg-yellow-500/20 text-yellow-400'
              : 'bg-surface-700 text-gray-300'
        }`}
      >
        {isUser ? 'U' : isSystem ? 'S' : 'AI'}
      </div>

      {/* Content */}
      <div className={`max-w-[80%] ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div
          className={`rounded-2xl px-4 py-2.5 ${
            isUser
              ? 'bg-primary-600 text-white rounded-tr-sm'
              : isSystem
                ? 'bg-yellow-500/10 text-yellow-300 border border-yellow-500/20 rounded-tl-sm'
                : 'bg-surface-800 text-gray-100 border border-surface-700 rounded-tl-sm'
          }`}
        >
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
        <span className="text-[10px] text-gray-600 mt-1">
          {message.timestamp
            ? new Date(message.timestamp).toLocaleTimeString()
            : '刚刚'}
        </span>
      </div>
    </div>
  );
}
