/** 进程内观察接口：不向请求体或模型输出添加字段。 */
export interface LlmResponseObservation {
  kind: 'sse_event' | 'decode_input' | 'decoded' | 'body' | 'parse_end' | 'tool_assembly';
  streamId: string;
  captureToken: string;
  eventSeq?: number;
  byteStart?: number;
  byteEnd?: number;
  parent?: unknown;
  value?: unknown;
}

export interface LlmResponseObserver {
  streamId: string;
  active(): string | undefined;
  observe(event: LlmResponseObservation): unknown;
}

const observers = new WeakMap<object, LlmResponseObserver>();
const sources = new WeakMap<object, unknown>();

export function attachLlmResponseObserver<T extends object>(response: T, observer: LlmResponseObserver): T {
  observers.set(response, observer);
  return response;
}

export function getLlmResponseObserver(response: object): LlmResponseObserver | undefined {
  return observers.get(response);
}

export function getLlmObservation(value: unknown): unknown {
  return value !== null && typeof value === 'object' ? sources.get(value) : undefined;
}

export function carryLlmObservation<T>(from: unknown, to: T): T {
  const source = getLlmObservation(from);
  if (source !== undefined && to !== null && typeof to === 'object') sources.set(to, source);
  return to;
}

export function observeLlmObject<T>(
  value: T,
  observer: LlmResponseObserver | undefined,
  event: () => Omit<LlmResponseObservation, 'streamId' | 'captureToken'>,
): T {
  try {
    const captureToken = observer?.active();
    if (observer && captureToken) {
      if (value !== null && typeof value === 'object') observers.set(value, observer);
      const source = observer.observe({ ...event(), streamId: observer.streamId, captureToken });
      if (source !== undefined && value !== null && typeof value === 'object') sources.set(value, source);
    }
  } catch {
    // 观察失败不影响解析和模型结果。
  }
  return value;
}

export function observeLlmDerived(parent: object | undefined, event: (captureToken: string) => Pick<LlmResponseObservation, 'kind' | 'value'>): void {
  try {
    const observer = parent ? observers.get(parent) : undefined;
    const captureToken = observer?.active();
    if (observer && captureToken) observer.observe({ ...event(captureToken), streamId: observer.streamId, captureToken, parent: getLlmObservation(parent) });
  } catch { /* 观察不改变解码。 */ }
}

/** 跟随真实解析器逐行前进，不建立第二份事件队列或累计正文。 */
export class LlmSseObservation {
  private offset = 0;
  private readStart = 0;
  private bytes: Uint8Array = new Uint8Array();
  private handledLines = 0;
  private scannedLines = 0;
  private cursor = -1;
  private lineEnd: number | undefined;
  private eventStart: number | undefined;
  private token: string | undefined;
  private eventSeq = 0;

  public constructor(private readonly observer: LlmResponseObserver) {
    this.token = this.active();
    if (this.token) this.eventStart = 0;
  }

  public read(bytes: Uint8Array): void {
    this.readStart = this.offset;
    this.offset += bytes.byteLength;
    this.bytes = bytes;
    this.handledLines = this.scannedLines = 0;
    this.cursor = -1;
  }

  public line(): void {
    this.handledLines += 1;
    const token = this.active();
    if (token !== this.token) {
      this.token = token;
      this.eventStart = undefined;
    }
    if (!token) {
      this.lineEnd = undefined;
      return;
    }
    while (this.scannedLines < this.handledLines) {
      this.cursor = this.bytes.indexOf(10, this.cursor + 1);
      this.scannedLines += 1;
      if (this.cursor < 0) break;
    }
    this.lineEnd = this.cursor < 0 ? this.offset : this.readStart + this.cursor + 1;
  }

  public end(): void { this.lineEnd = this.offset; }

  public dispatch<T>(value: T, data: string, event?: string): T {
    this.eventSeq += 1;
    const byteStart = this.eventStart;
    const byteEnd = this.lineEnd;
    const eventSeq = this.eventSeq;
    this.eventStart = byteEnd;
    if (!data) return value;
    return observeLlmObject(value, this.observer, () => ({
      kind: 'sse_event', eventSeq, byteStart, byteEnd, value: { event, data },
    }));
  }

  private active(): string | undefined {
    try { return this.observer.active(); } catch { return undefined; }
  }
}
