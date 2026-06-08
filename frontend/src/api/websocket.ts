type EventHandler = (data: any) => void;

export class WSClient {
  private ws: WebSocket | null = null;
  private handlers = new Map<string, Set<EventHandler>>();
  private reconnectTimer: number | null = null;
  private reconnectAttempts = 0;
  private url: string;
  private maxReconnectDelay = 60000;

  constructor(url = `ws://${window.location.hostname}:8000/ws`) {
    this.url = url;
  }

  private getReconnectDelay(): number {
    // Exponential backoff with jitter: 3s, 6s, 12s, 24s, 48s, max 60s
    const delay = Math.min(
      3000 * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay,
    );
    const jitter = delay * (0.5 + Math.random() * 0.5); // 50-100% of delay
    return Math.round(jitter);
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        const { event: eventName, data } = payload;
        const handlers = this.handlers.get(eventName);
        if (handlers) {
          handlers.forEach((handler) => handler(data));
        }
        const wildcardHandlers = this.handlers.get('*');
        if (wildcardHandlers) {
          wildcardHandlers.forEach((handler) => handler(payload));
        }
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    this.ws.onclose = () => {
      const delay = this.getReconnectDelay();
      console.log(`WebSocket disconnected, reconnecting in ${delay}ms (attempt ${this.reconnectAttempts + 1})...`);
      this.reconnectAttempts++;
      this.reconnectTimer = window.setTimeout(() => this.connect(), delay);
    };

    this.ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      this.ws?.close();
    };
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  on(event: string, handler: EventHandler) {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
    return () => this.handlers.get(event)?.delete(handler);
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}

export const wsClient = new WSClient();
