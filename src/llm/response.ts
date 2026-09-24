/**
 * 响应后处理模块
 *
 * 统一处理流式和非流式响应。
 * 内部使用 FormatAdapter 做格式解码，内置 SSE 解析处理流式数据。
 */

import type { LLMRawErrorInfo, LLMResponse, LLMStreamChunk } from '../types.js';
import type { FormatAdapter } from './formats/types.js';
import { getLlmResponseObserver, getLlmObservation, observeLlmObject, LlmSseObservation } from './observation.js';

// ============ 通用错误透传 ============

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function headersToRecord(headers: Headers): Record<string, string> {
  const result: Record<string, string> = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}

function stringifyError(err: unknown): string {
  return err instanceof Error ? `${err.name}: ${err.message}` : String(err);
}

function tryParseJson(text: string): { ok: true; value: unknown } | { ok: false } {
  if (!text.trim()) return { ok: false };
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch {
    return { ok: false };
  }
}

async function readResponseBody(res: Response): Promise<{ bodyText: string; rawBody?: unknown }> {
  const bodyText = await res.text();
  const parsed = tryParseJson(bodyText);
  observeLlmObject(parsed, getLlmResponseObserver(res), () => ({ kind: 'body', value: bodyText }));
  return parsed.ok ? { bodyText, rawBody: parsed.value } : { bodyText };
}

function createErrorResponse(error: LLMRawErrorInfo): LLMResponse {
  return {
    content: { role: 'model', parts: [{ text: '' }] },
    error,
    rawResponse: error.rawBody ?? error.bodyText,
  };
}

function createErrorStreamChunk(error: LLMRawErrorInfo): LLMStreamChunk {
  return {
    error,
    rawChunk: error.rawChunk ?? error.rawBody ?? error.bodyText ?? error.data,
  };
}

function stringField(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function hasErrorLikeEvent(event: unknown): boolean {
  if (typeof event !== 'string') return false;
  const normalized = event.toLowerCase();
  return normalized.includes('error')
    || normalized.includes('failed')
    || normalized.includes('incomplete');
}

function hasErrorLikeType(type: unknown): boolean {
  if (typeof type !== 'string') return false;
  const normalized = type.toLowerCase();
  return normalized === 'error'
    || normalized.endsWith('_error')
    || normalized.includes('error')
    || normalized.includes('failed')
    || normalized.includes('incomplete');
}

function hasErrorLikeStatus(status: unknown): boolean {
  if (typeof status !== 'string') return false;
  const normalized = status.toLowerCase();
  return normalized === 'error'
    || normalized === 'failed'
    || normalized === 'incomplete'
    || normalized === 'cancelled';
}

function hasNonNullErrorField(payload: Record<string, unknown>): boolean {
  return 'error' in payload && payload.error !== null && payload.error !== undefined;
}

function isProviderErrorPayload(payload: unknown, event?: string): boolean {
  if (hasErrorLikeEvent(event)) return true;
  if (!isPlainObject(payload)) return false;

  if (hasNonNullErrorField(payload)) return true;
  if (hasErrorLikeEvent(payload.event)) return true;
  if (hasErrorLikeType(payload.type)) return true;
  if (hasErrorLikeStatus(payload.status) && ('message' in payload || 'last_error' in payload || 'incomplete_details' in payload || hasNonNullErrorField(payload))) return true;

  const response = payload.response;
  if (isPlainObject(response)) {
    if (hasNonNullErrorField(response)) return true;
    if (hasErrorLikeStatus(response.status) && ('message' in response || 'last_error' in response || 'incomplete_details' in response || hasNonNullErrorField(response))) return true;
  }

  return false;
}

// ============ 非流式 ============

/** 处理非流式响应 */
export async function processResponse(
  res: Response,
  format: FormatAdapter,
): Promise<LLMResponse> {
  const headers = headersToRecord(res.headers);
  const { bodyText, rawBody } = await readResponseBody(res);
  const rawResponse = rawBody ?? bodyText;
  const observed = (chunk: LLMResponse) => observeLlmObject(chunk, getLlmResponseObserver(res), () => ({ kind: 'decoded', value: chunk }));

  if (!res.ok) {
    return observed(createErrorResponse({
      kind: 'http_error',
      status: res.status,
      statusText: res.statusText,
      headers,
      bodyText,
      ...(rawBody !== undefined ? { rawBody } : {}),
    }));
  }

  if (isProviderErrorPayload(rawResponse)) {
    return observed(createErrorResponse({
      kind: 'response_error',
      status: res.status,
      statusText: res.statusText,
      headers,
      bodyText,
      rawBody: rawResponse,
    }));
  }

  try {
    const decoded = format.decodeResponse(rawResponse);
    // 格式适配器识别出的错误（如 finish_reason:"error"）只知道响应体语义，这里补上 HTTP 上下文。
    const result = decoded.error && decoded.error.status === undefined
      ? { ...decoded, error: { ...decoded.error, status: res.status, statusText: res.statusText, headers, bodyText } }
      : decoded;
    return observeLlmObject(result, getLlmResponseObserver(res), () => ({ kind: 'decoded', value: result }));
  } catch (err) {
    return observed(createErrorResponse({
      kind: 'decode_error',
      status: res.status,
      statusText: res.statusText,
      headers,
      bodyText,
      rawBody: rawResponse,
      message: stringifyError(err),
    }));
  }
}

// ============ 流式 ============

/** 处理流式响应（SSE 解析 + 逐块解码） */
export async function* processStreamResponse(
  res: Response,
  format: FormatAdapter,
): AsyncGenerator<LLMStreamChunk> {
  const headers = headersToRecord(res.headers);
  const observed = (chunk: LLMStreamChunk, parent?: unknown) => observeLlmObject(chunk, getLlmResponseObserver(res), () => ({ kind: 'decoded', value: chunk, parent }));

  if (!res.ok) {
    const { bodyText, rawBody } = await readResponseBody(res);
    yield observed(createErrorStreamChunk({
      kind: 'http_error',
      status: res.status,
      statusText: res.statusText,
      headers,
      bodyText,
      ...(rawBody !== undefined ? { rawBody } : {}),
    }));
    return;
  }

  const state = format.createStreamState();
  // 格式适配器自己产出的错误块（如工具参数被截断）只知道解码语义，这里补上 HTTP 上下文。
  const withHttpErrorContext = (chunk: LLMStreamChunk): LLMStreamChunk => {
    if (!chunk.error || chunk.error.status !== undefined) return chunk;
    return {
      ...chunk,
      error: { ...chunk.error, status: res.status, statusText: res.statusText, headers },
    };
  };
  let lastPayload: unknown;
  let streamCompleted = false;
  // 流里已经发出过错误块（上游错误事件、非 JSON 数据、解码失败或适配器自己的错误块）。
  let emittedErrorChunk = false;
  const sseEnd: SSEEnd = { receivedDone: false };
  try {
    for await (const sse of parseSSE(res, sseEnd)) {
      const parsed = tryParseJson(sse.data);
      if (!parsed.ok) {
        emittedErrorChunk = true;
        yield observed(createErrorStreamChunk({
          kind: 'stream_parse_error',
          status: res.status,
          statusText: res.statusText,
          headers,
          event: sse.event,
          data: sse.data,
          bodyText: sse.data,
          message: `SSE data 不是 JSON: ${sse.data}`,
          rawChunk: sse.data,
        }), getLlmObservation(sse));
        continue;
      }

      const payload = isPlainObject(parsed.value)
        ? { ...parsed.value, ...(sse.event ? { event: sse.event } : {}) }
        : parsed.value;
      observeLlmObject(payload, getLlmResponseObserver(res), () => ({
        kind: 'decode_input', parent: getLlmObservation(sse), value: payload,
      }));

      if (isProviderErrorPayload(payload, sse.event)) {
        emittedErrorChunk = true;
        yield observed(createErrorStreamChunk({
          kind: 'stream_error',
          status: res.status,
          statusText: res.statusText,
          headers,
          event: sse.event ?? stringField(isPlainObject(payload) ? payload.event : undefined),
          data: sse.data,
          rawChunk: payload,
        }), getLlmObservation(payload));
        continue;
      }

      lastPayload = payload;
      try {
        const chunk = withHttpErrorContext(format.decodeStreamChunk(payload, state));
        if (chunk.error) emittedErrorChunk = true;
        yield observeLlmObject(chunk, getLlmResponseObserver(res), () => ({
          kind: 'decoded', parent: getLlmObservation(payload), value: chunk,
        }));
      } catch (err) {
        emittedErrorChunk = true;
        yield observed(createErrorStreamChunk({
          kind: 'decode_error',
          status: res.status,
          statusText: res.statusText,
          headers,
          event: sse.event,
          data: sse.data,
          rawChunk: payload,
          message: stringifyError(err),
        }), getLlmObservation(payload));
      }
    }
    streamCompleted = true;
  } catch (err) {
    yield observed(createErrorStreamChunk({
      kind: 'stream_read_error',
      status: res.status,
      statusText: res.statusText,
      headers,
      message: stringifyError(err),
    }));
  }

  // 流正常结束（[DONE] 或 EOF）后给格式适配器一次补发机会；读取中断时流不完整，不补发。
  // 已经发出过错误块的流（例如 OpenRouter 带顶层 error 和 finish_reason:"error" 的中途错误块，
  // 它在这里就被拦下，格式适配器看不到）已经失败，不再补发待定的工具调用、截断错误或签名信封。
  if (!streamCompleted || emittedErrorChunk || typeof format.finalizeStream !== 'function') return;
  // 告诉格式适配器流是否以 [DONE] 结束：没有 [DONE] 的 EOF 可能是连接在工具参数发完之前断开。
  state.streamEnd = sseEnd.receivedDone ? 'done' : 'eof';
  const parent = getLlmObservation(lastPayload);
  let finalChunk: LLMStreamChunk | undefined;
  try {
    finalChunk = format.finalizeStream(state);
  } catch (err) {
    yield observed(createErrorStreamChunk({
      kind: 'decode_error',
      status: res.status,
      statusText: res.statusText,
      headers,
      message: stringifyError(err),
    }), parent);
    return;
  }
  if (!finalChunk) return;
  const chunk = withHttpErrorContext(finalChunk);
  yield observeLlmObject(chunk, getLlmResponseObserver(res), () => ({ kind: 'decoded', parent, value: chunk }));
}

// ============ SSE 解析 ============

export interface SSEChunk {
  event?: string;
  data: string;
}

/** parseSSE 结束时写入：是否收到了 `data: [DONE]`（否则是没有 [DONE] 的 EOF）。 */
interface SSEEnd {
  receivedDone: boolean;
}

/**
 * 从 fetch Response 中解析 SSE 流，逐条 yield 包含 data 字段的原始字符串的对象。
 * 遇到 `data: [DONE]` 时自动结束，并把 end.receivedDone 置为 true；
 * `for await` 拿不到生成器的返回值，所以通过这个对象告诉调用方。
 */
async function* parseSSE(response: Response, end: SSEEnd): AsyncGenerator<SSEChunk> {
  const body = response.body;
  if (!body) throw new Error('Response body is null');

  const reader = (body as any).getReader() as ReadableStreamDefaultReader<Uint8Array>;
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent: string | undefined;
  let dataLines: string[] = [];
  let chunksRead = 0;
  const observer = getLlmResponseObserver(response);
  const observation = observer ? new LlmSseObservation(observer) : undefined;

  const dispatch = (): SSEChunk | 'done' | undefined => {
    const data = dataLines.join('\n');
    const event = currentEvent;
    dataLines = [];
    currentEvent = undefined;

    const chunk = data ? { event, data } : undefined;
    observation?.dispatch(chunk, data, event);
    if (data === '[DONE]') {
      end.receivedDone = true;
      return 'done';
    }
    return chunk;
  };

  const handleLine = (rawLine: string): SSEChunk | 'done' | undefined => {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
    if (line === '') return dispatch();
    if (line.startsWith(':')) return undefined; // SSE comment / heartbeat

    const colonIndex = line.indexOf(':');
    const field = colonIndex >= 0 ? line.slice(0, colonIndex) : line;
    let value = colonIndex >= 0 ? line.slice(colonIndex + 1) : '';
    // SSE 规范：冒号后可选一个空格。Anthropic 兼容端点可能发送 `data:{...}`，
    // 也可能发送 `data: {...}`，两种都必须接受。
    if (value.startsWith(' ')) value = value.slice(1);

    if (field === 'event') {
      const event = value.trim();
      currentEvent = event || undefined;
    } else if (field === 'data') {
      dataLines.push(value);
    }
    return undefined;
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunksRead++;
      observation?.read(value);

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        observation?.line();
        const chunk = handleLine(line);
        if (chunk === 'done') return;
        if (chunk) yield chunk;
      }
    }

    observation?.end();
    buffer += decoder.decode();
    if (buffer) {
      const lines = buffer.split('\n');
      for (const line of lines) {
        const chunk = handleLine(line);
        if (chunk === 'done') return;
        if (chunk) yield chunk;
      }
    }

    const chunk = dispatch();
    if (chunk !== 'done' && chunk) yield chunk;
  } catch (err) {
    // 为连接中断错误补充上下文（已接收块数帮助判断是建连失败还是中途断开）
    const msg = err instanceof Error ? err.message : String(err);
    const wrapped = new Error(`SSE 流读取中断（已接收 ${chunksRead} 个数据块）: ${msg}`);
    (wrapped as any).cause = err;
    throw wrapped;
  } finally {
    reader.releaseLock();
  }
}
