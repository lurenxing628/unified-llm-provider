/**
 * OpenAI Compatible 格式适配器
 *
 * Gemini ↔ OpenAI 格式的完整双向转换。
 * 适用于所有 OpenAI 兼容接口（OpenAI、DeepSeek、本地模型等）。
 *
 * 支持 reasoning_content（DeepSeek / KIMI 等模型的 thinking 字段）。
 */

import {
  LLMRequest, LLMResponse, LLMStreamChunk, Part, FunctionResponsePart,
  isTextPart, isVisibleTextPart, isInlineDataPart, isFunctionCallPart, isFunctionResponsePart,
} from '../../types.js';
import { FormatAdapter, StreamDecodeState } from './types.js';
import { consumeCallId, normalizeCallId, resolveCallId } from './tool-call-ids.js';
import { sanitizeSchemaForOpenAI } from './schema-sanitizer.js';
import { mapDeepSeekThinkingLevel, mapOpenAIThinkingLevel } from './thinking-level.js';
import { isToolResponseDocumentMimeType, isToolResponseImageMimeType, parseBase64DataUrl, toBase64DataUrl } from '../vision.js';

export class OpenAICompatibleFormat implements FormatAdapter {
  constructor(private model: string, private providerKind: 'openai-compatible' | 'deepseek' = 'openai-compatible') {}

  // ============ 编码请求：Gemini → OpenAI ============

  encodeRequest(request: LLMRequest, stream?: boolean): unknown {
    const messages: Record<string, unknown>[] = [];

    // systemInstruction → system message
    if (request.systemInstruction?.parts) {
      const text = request.systemInstruction.parts
        .filter(isVisibleTextPart).map(p => p.text).join('\n');
      if (text) messages.push({ role: 'system', content: text });
    }

    // contents → messages
    const pendingToolCallIds: string[] = [];
    let generatedToolCallIdCounter = 0;
    for (const content of request.contents) {
      const textParts = content.parts.filter(isVisibleTextPart);
      const funcCallParts = content.parts.filter(isFunctionCallPart);
      const funcRespParts = content.parts.filter(isFunctionResponsePart);

      if (content.role === 'model') {
        // 提取 thinking/reasoning 内容（thought: true 的 text parts）
        const thoughtParts = content.parts.filter(p => isTextPart(p) && p.thought === true);
        const reasoningContent = thoughtParts.map(p => (p as any).text || '').join('') || null;
        const reasoningSignature = thoughtParts.map(p => (p as any).thoughtSignatures?.['openai-compatible']).find((value: unknown) => typeof value === 'string' && value.trim()) || null;

        if (funcCallParts.length > 0) {
          const toolCalls = funcCallParts.map((part, i) => {
            if (!isFunctionCallPart(part)) {
              throw new Error('unreachable');
            }
            const callId = resolveCallId(part.functionCall.callId, `call_${generatedToolCallIdCounter + i}`);
            pendingToolCallIds.push(callId);
            return {
              id: callId,
              type: 'function' as const,
              function: {
                name: part.functionCall.name,
                arguments: JSON.stringify(part.functionCall.args),
              },
            };
          });
          generatedToolCallIdCounter += funcCallParts.length;
          const text = textParts.map(p => {
            if (!isTextPart(p)) throw new Error('unreachable');
            return p.text;
          }).join('') || null;
          const msg: Record<string, unknown> = { role: 'assistant', content: text, tool_calls: toolCalls };
          if (reasoningContent) msg.reasoning_content = reasoningContent;
          if (reasoningSignature) msg.reasoning_signature = reasoningSignature;
          messages.push(msg);
       } else {
          const text = textParts.map(p => {
            if (!isTextPart(p)) throw new Error('unreachable');
            return p.text;
          }).join('');
          const msg: Record<string, unknown> = { role: 'assistant', content: text };
          if (reasoningContent) msg.reasoning_content = reasoningContent;
          if (reasoningSignature) msg.reasoning_signature = reasoningSignature;
          messages.push(msg);
        }
      } else {
        if (funcRespParts.length > 0) {
          for (let i = 0; i < funcRespParts.length; i++) {
            const part = funcRespParts[i];
            if (!isFunctionResponsePart(part)) {
              throw new Error('unreachable');
            }
            const callId = consumeCallId({
              explicit: part.functionResponse.callId,
              pendingCallIds: pendingToolCallIds,
              providerLabel: 'OpenAI Compatible',
              toolName: part.functionResponse.name,
            });
            messages.push({
              role: 'tool',
              tool_call_id: callId,
              content: encodeOpenAICompatibleToolResultContent(part.functionResponse),
            });
          }
        } else {
          const contentBlocks: Record<string, unknown>[] = [];
          let hasStructuredContent = false;

          for (const part of content.parts) {
            if (isTextPart(part) && part.thought !== true && part.text) {
              contentBlocks.push({ type: 'text', text: part.text });
            } else if (isInlineDataPart(part)) {
              const mime = part.inlineData.mimeType;
              if (isToolResponseImageMimeType(mime)) {
                hasStructuredContent = true;
                contentBlocks.push({
                  type: 'image_url',
                  image_url: {
                    url: toBase64DataUrl(part.inlineData),
                  },
                });
              } else if (isToolResponseDocumentMimeType(mime)) {
                hasStructuredContent = true;
                contentBlocks.push({
                  type: 'file',
                  file: {
                    file_data: toBase64DataUrl(part.inlineData),
                  },
                });
              }
            }
          }
          if (hasStructuredContent) {
            messages.push({ role: 'user', content: contentBlocks });
          } else {
            const text = textParts.map(p => {
              if (!isTextPart(p)) throw new Error('unreachable');
              return p.text;
            }).join('');
            messages.push({ role: 'user', content: text });
          }
        }
      }
    }

    // 组装请求体
    const body: Record<string, unknown> = { model: this.model, messages };

    // tools 声明转换
    if (request.tools && request.tools.length > 0) {
      const allDecls = request.tools.flatMap(t => Array.isArray((t as any).functionDeclarations) ? (t as any).functionDeclarations : []);
      body.tools = allDecls.map(decl => ({
        type: 'function',
        function: { name: decl.name, description: decl.description, parameters: sanitizeSchemaForOpenAI(decl.parameters) },
      }));
    }

    // generationConfig 转换
    if (request.generationConfig) {
      const gc = request.generationConfig;
      if (gc.temperature !== undefined) body.temperature = gc.temperature;
      if (gc.topP !== undefined) body.top_p = gc.topP;
      if (gc.maxOutputTokens !== undefined) body.max_tokens = gc.maxOutputTokens;
      if (gc.stopSequences !== undefined) body.stop = gc.stopSequences;

      if (this.providerKind === 'deepseek') {
        const thinkingLevel = mapDeepSeekThinkingLevel(gc.thinkingConfig?.thinkingLevel);
        if (thinkingLevel === 'none') {
          body.thinking = { type: 'disabled' };
        } else if (thinkingLevel === 'high' || thinkingLevel === 'max') {
          body.thinking = { type: 'enabled' };
          body.reasoning_effort = thinkingLevel;
        }
      } else {
        const thinkingLevel = mapOpenAIThinkingLevel(gc.thinkingConfig?.thinkingLevel);
        if (thinkingLevel) body.reasoning_effort = thinkingLevel;
      }
    }

    // 流式参数
    if (stream) {
      body.stream = true;
      body.stream_options = { include_usage: true };
    }

    return body;
  }

  // ============ 解码响应：OpenAI → Gemini ============

  decodeResponse(raw: unknown): LLMResponse {
    const data = raw as any;
    const choice = data.choices?.[0];
    if (choice?.finish_reason === 'error') {
      return createFinishReasonErrorResponse(data, choice);
    }
    if (!choice?.message) {
      throw new Error(`OpenAI Compatible API 未返回有效内容: ${JSON.stringify(data)}`);
    }

    const msg = choice.message;
    const parts: Part[] = [];

    // reasoning_content → thought part（DeepSeek / KIMI 等模型的 thinking 输出）
    if ((typeof msg.reasoning_content === 'string' && msg.reasoning_content) || typeof msg.reasoning_signature === 'string') {
      const thoughtPart: Part = {
        text: typeof msg.reasoning_content === 'string' ? msg.reasoning_content : '',
        thought: true,
        ...(typeof msg.reasoning_signature === 'string' ? { thoughtSignature: msg.reasoning_signature } : {}),
      } as any;
      parts.push(thoughtPart);
    }

    if (typeof msg.content === 'string') {
      parts.push({ text: msg.content });
    } else if (Array.isArray(msg.content)) {
      for (const block of msg.content) {
        appendOpenAIContentBlock(parts, block);
      }
    }
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        parts.push({
          functionCall: {
            name: tc.function.name,
            args: decodeNonStreamToolArguments(tc, choice.finish_reason),
            callId: normalizeCallId(tc.id),
          },
        });
      }
    }
    if (parts.length === 0) parts.push({ text: '' });

    return {
      content: { role: 'model', parts },
      finishReason: choice.finish_reason,
      usageMetadata: data.usage
        ? (() => {
            const cached = data.usage.prompt_tokens_details?.cached_tokens ?? data.usage.prompt_cache_hit_tokens ?? 0;
            const reasoningTokens = data.usage.completion_tokens_details?.reasoning_tokens;
            return {
              promptTokenCount: data.usage.prompt_tokens,
              ...(cached > 0 ? { cachedContentTokenCount: cached } : {}),
              ...(typeof reasoningTokens === 'number' ? { thoughtsTokenCount: reasoningTokens } : {}),
              candidatesTokenCount: data.usage.completion_tokens,
              totalTokenCount: data.usage.total_tokens,
            };
          })()
        : undefined,
    };
  }

  // ============ 流式解码 ============

  decodeStreamChunk(raw: unknown, state: StreamDecodeState): LLMStreamChunk {
    const data = raw as any;
    const choice = data.choices?.[0];
    const chunk: LLMStreamChunk = {};

    if (choice?.finish_reason === 'error') {
      // 流已被上游以错误终止：丢弃尚未完成的工具调用，流结束时也不再补发任何内容。
      const streamState = state as OpenAICompatibleStreamState;
      streamState.pendingToolCalls.clear();
      streamState.lastToolCallKey = undefined;
      streamState.terminatedWithError = true;
      return createFinishReasonErrorChunk(data, choice);
    }

    // reasoning_content 流式增量（DeepSeek / KIMI 等模型的 thinking 输出）
    if (choice?.delta?.reasoning_content) {
      chunk.partsDelta = [
        ...(chunk.partsDelta || []),
        { text: choice.delta.reasoning_content, thought: true } as any,
      ];
    }

    if (choice?.delta?.reasoning_signature) {
      chunk.partsDelta = [
        ...(chunk.partsDelta || []),
        { thought: true, thoughtSignature: choice.delta.reasoning_signature } as any,
      ];
      chunk.thoughtSignature = choice.delta.reasoning_signature;
    }

    if (choice?.delta?.content) {
      chunk.textDelta = choice.delta.content;
      chunk.partsDelta = [
        ...(chunk.partsDelta || []),
        { text: choice.delta.content },
      ];
    }

    // 流式边执行优化：累积工具调用分片，并在检测到工具参数完整时立即输出。
    //
    // OpenAI 的 tool_call 分片按 index 顺序发送，没有"单个工具参数完成"的显式信号。
    // 但有一个规律：当 delta 中出现新的 tool_call index 时，说明前一个 index 的
    // 参数已经流完了。利用这个规律，在新 index 出现时立即输出前一个已完成的工具调用，
    // 让 StreamingToolExecutor 可以在 LLM 还在输出后续工具参数时提前启动执行。
    // finish_reason 到达时，最后一个工具也输出；上游没有 finish_reason 时由 finalizeStream 补发。
    const streamState = state as OpenAICompatibleStreamState;
    const pending = streamState.pendingToolCalls;
    const emitPendingToolCall = (entry: PendingStreamToolCall, options?: { allowEmptyArgs?: boolean }) =>
      emitStreamToolCall(chunk, entry, options);
    if (choice?.delta?.tool_calls) {
      for (const tc of choice.delta.tool_calls) {
        const key = resolveStreamToolCallKey(streamState, tc);
        // 新调用出现时，前面未输出的工具调用的参数一定已经完整，立即输出
        if (!pending.has(key) && pending.size > 0) {
          for (const [, entry] of pending) {
            emitPendingToolCall(entry, { allowEmptyArgs: true });
          }
        }
        if (!pending.has(key)) {
          pending.set(key, { callId: undefined, name: '', arguments: '', emitted: false });
        }
        streamState.lastToolCallKey = key;
        const entry = pending.get(key)!;
        if (tc.id) entry.callId = normalizeCallId(tc.id) ?? entry.callId;
        if (tc.function?.name) entry.name = tc.function.name;
        appendStreamToolArguments(entry, tc.function?.arguments);
        // 单个 tool_call 没有“下一个 index”可作为完成信号；当参数 JSON 已经完整时立即输出，
        // 让 AskQuestionFirst 这类交互工具可以在 message 结束前显示面板。
        emitPendingToolCall(entry);
      }
    }
    // finish_reason 到达时，输出最后一个（及所有尚未输出的）工具调用
    if (choice?.finish_reason) {
      chunk.finishReason = choice.finish_reason;
      if (pending.size > 0) {
        const failures: FailedStreamToolCall[] = [];
        for (const [, entry] of pending) {
          const problem = emitPendingToolCall(entry, { allowEmptyArgs: true });
          if (problem) failures.push({ entry, problem });
        }
        pending.clear();
        streamState.lastToolCallKey = undefined;
        if (failures.length > 0) {
          // 参数不完整（多为 finish_reason=length 截断）的调用不再静默丢弃，按解码错误上报。
          return createToolArgumentsErrorChunk(failures, choice.finish_reason, data);
        }
      }
    }

    // usage
    if (data.usage) {
      const cached = data.usage.prompt_tokens_details?.cached_tokens ?? data.usage.prompt_cache_hit_tokens ?? 0;
      const reasoningTokens = data.usage.completion_tokens_details?.reasoning_tokens;
      chunk.usageMetadata = {
        promptTokenCount: data.usage.prompt_tokens,
        ...(cached > 0
          ? { cachedContentTokenCount: cached }
          : {}),
        ...(typeof reasoningTokens === 'number' ? { thoughtsTokenCount: reasoningTokens } : {}),
        candidatesTokenCount: data.usage.completion_tokens,
        totalTokenCount: data.usage.total_tokens,
      };
    }

    return chunk;
  }

  createStreamState(): StreamDecodeState {
    const state: OpenAICompatibleStreamState = {
      pendingToolCalls: new Map<number | string, PendingStreamToolCall>(),
      anonymousToolCallCount: 0,
    };
    return state;
  }

  /**
   * 流结束（[DONE] / EOF）后补发仍在等待的工具调用。
   *
   * 部分网关（实测：把 Chat Completions 转成其他协议的中转）在工具调用流里不发 finish_reason，
   * 最后一个调用（尤其是 arguments 为空串的无参数调用）永远等不到输出信号。流已正常结束时，
   * 空参数按 `{}` 发出；参数 JSON 不完整的调用视为被截断，返回解码错误块而不是静默丢弃。
   */
  finalizeStream(state: StreamDecodeState): LLMStreamChunk | undefined {
    const streamState = state as OpenAICompatibleStreamState;
    if (streamState.terminatedWithError) return undefined;
    const pending = streamState.pendingToolCalls;
    if (!pending || pending.size === 0) return undefined;

    const chunk: LLMStreamChunk = {};
    const failures: FailedStreamToolCall[] = [];
    for (const [, entry] of pending) {
      const problem = emitStreamToolCall(chunk, entry, { allowEmptyArgs: true });
      if (problem) failures.push({ entry, problem });
    }
    pending.clear();
    streamState.lastToolCallKey = undefined;
    if (failures.length > 0) {
      return createToolArgumentsErrorChunk(failures, undefined, {
        tool_calls: failures.map(({ entry }) => ({
          id: entry.callId,
          function: { name: entry.name, arguments: entry.arguments },
        })),
      });
    }
    return chunk.functionCalls?.length ? chunk : undefined;
  }
}

// ============ 流式工具调用累积 ============

interface PendingStreamToolCall {
  callId?: string;
  name: string;
  arguments: string;
  emitted?: boolean;
}

interface OpenAICompatibleStreamState extends StreamDecodeState {
  pendingToolCalls: Map<number | string, PendingStreamToolCall>;
  /** 最近一次写入的调用；既没有 index 也没有 id 的续传分片归到这里。 */
  lastToolCallKey?: number | string;
  anonymousToolCallCount: number;
  /** 已收到 finish_reason:"error"。 */
  terminatedWithError?: boolean;
}

type StreamToolArgumentsProblem = 'incomplete' | 'not_object';

interface FailedStreamToolCall {
  entry: PendingStreamToolCall;
  problem: StreamToolArgumentsProblem;
}

/**
 * 为一个 tool_call delta 找到它所属的累积条目。
 *
 * 带 index 的分片（OpenAI 官方流式格式）按 index 归并，行为与以前完全一致。
 * 个别兼容实现的 delta 不带 index：此时按 id 区分调用（已见过的 id 续写原调用，新 id 视为新调用），
 * 既没有 index 也没有 id 的分片是上一个调用的参数续传。以前这些分片都落在 `undefined` 这个 key 上，
 * 并行调用会被拼成一条参数错乱的调用。
 */
function resolveStreamToolCallKey(state: OpenAICompatibleStreamState, tc: any): number | string {
  if (typeof tc?.index === 'number') return tc.index;
  const id = normalizeCallId(tc?.id);
  if (id) {
    for (const [key, entry] of state.pendingToolCalls) {
      if (entry.callId === id) return key;
    }
    return `id:${id}`;
  }
  if (state.lastToolCallKey !== undefined && state.pendingToolCalls.has(state.lastToolCallKey)) {
    return state.lastToolCallKey;
  }
  state.anonymousToolCallCount += 1;
  return `anonymous:${state.anonymousToolCallCount}`;
}

function appendStreamToolArguments(entry: PendingStreamToolCall, value: unknown): void {
  if (typeof value === 'string') {
    if (value) entry.arguments += value;
    return;
  }
  // 少数兼容实现直接给出已解析的参数对象；按完整 JSON 文本累积。
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    entry.arguments += JSON.stringify(value);
  }
}

function parseStreamToolArguments(rawArgs: string): { args: Record<string, unknown> } | { problem: StreamToolArgumentsProblem } {
  if (!rawArgs.trim()) return { args: {} };
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawArgs);
  } catch {
    return { problem: 'incomplete' };
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return { problem: 'not_object' };
  return { args: parsed as Record<string, unknown> };
}

/**
 * 尝试把一个累积中的调用写进 chunk。成功或还不该输出时返回 undefined；
 * 参数无法解析时返回问题类型，由调用方决定继续等待还是上报错误。
 */
function emitStreamToolCall(
  chunk: LLMStreamChunk,
  entry: PendingStreamToolCall,
  options?: { allowEmptyArgs?: boolean },
): StreamToolArgumentsProblem | undefined {
  if (entry.emitted || !entry.name) return undefined;
  const rawArgs = entry.arguments ?? '';
  if (!rawArgs.trim() && !options?.allowEmptyArgs) {
    // OpenAI-compatible providers often send an initial tool_call delta with
    // function.name and arguments="", followed by later arguments fragments.
    // Treating empty arguments as {} here would prematurely emit the tool
    // call and drop subsequent argument deltas.
    return undefined;
  }
  const parsed = parseStreamToolArguments(rawArgs);
  // 参数 JSON 尚未完整时继续等待后续 delta / finish_reason / 流结束。
  if ('problem' in parsed) return parsed.problem;
  if (!chunk.functionCalls) chunk.functionCalls = [];
  const part = {
    functionCall: {
      name: entry.name,
      args: parsed.args,
      callId: entry.callId,
    },
  };
  chunk.functionCalls.push(part);
  chunk.partsDelta = [...(chunk.partsDelta || []), part];
  entry.emitted = true;
  return undefined;
}

const TOOL_ARGUMENTS_PREVIEW_CHARS = 200;

function previewToolArguments(rawArgs: string): string {
  return rawArgs.length > TOOL_ARGUMENTS_PREVIEW_CHARS
    ? `${rawArgs.slice(0, TOOL_ARGUMENTS_PREVIEW_CHARS)}…`
    : rawArgs;
}

function describeToolCall(name: unknown, callId: unknown): string {
  const label = typeof name === 'string' && name ? `"${name}"` : '(unknown tool)';
  return typeof callId === 'string' && callId ? `${label}（${callId}）` : label;
}

function describeToolArgumentsProblem(
  name: unknown,
  callId: unknown,
  rawArgs: string,
  problem: StreamToolArgumentsProblem,
  finishReason?: string,
): string {
  const call = describeToolCall(name, callId);
  if (problem === 'not_object') {
    return `工具调用 ${call} 的参数不是 JSON 对象：${previewToolArguments(rawArgs)}`;
  }
  const reason = finishReason ? `finish_reason: ${finishReason}` : '流结束时仍未收到完整参数';
  return `工具调用 ${call} 的参数 JSON 不完整，参数可能被截断（${reason}，已收到 ${rawArgs.length} 个字符）：${previewToolArguments(rawArgs)}`;
}

function createToolArgumentsErrorChunk(
  failures: FailedStreamToolCall[],
  finishReason: string | undefined,
  rawChunk: unknown,
): LLMStreamChunk {
  const message = failures
    .map(({ entry, problem }) => describeToolArgumentsProblem(entry.name, entry.callId, entry.arguments, problem, finishReason))
    .join('\n');
  return {
    error: { kind: 'decode_error', message, rawChunk },
    rawChunk,
    ...(finishReason ? { finishReason } : {}),
  };
}

// ============ finish_reason: "error" ============

/**
 * OpenRouter 文档（https://openrouter.ai/docs/api/reference/errors-and-debugging）：
 * 已经返回 200 之后发生的错误，流式以一个 `finish_reason: "error"` 的块终止流，非流式把 error
 * 放在 choice 里并带 `finish_reason: "error"`；choice 上可能带 `native_finish_reason`
 * （如 Gemini 的 MALFORMED_FUNCTION_CALL）。顶层带 `error` 的块已由 response 层按 stream_error
 * 处理；这里补上只有 `finish_reason: "error"` 的情况，按错误上报而不是当作正常结束。
 */
function describeFinishReasonError(data: any, choice: any): { message: string; code?: string } {
  const nativeFinishReason = typeof choice?.native_finish_reason === 'string' && choice.native_finish_reason
    ? choice.native_finish_reason
    : undefined;
  const nested = choice?.error && typeof choice.error === 'object' ? choice.error : undefined;
  const upstreamMessage = typeof nested?.message === 'string' && nested.message
    ? nested.message
    : typeof data?.error?.message === 'string' && data.error.message ? data.error.message : undefined;
  const rawCode = nested?.code ?? data?.error?.code;
  const code = typeof rawCode === 'string' || typeof rawCode === 'number' ? String(rawCode) : undefined;
  const details = [
    nativeFinishReason ? `native_finish_reason: ${nativeFinishReason}` : undefined,
    code ? `code: ${code}` : undefined,
  ].filter(Boolean).join('，');
  const message = `上游以 finish_reason: "error" 结束了生成${details ? `（${details}）` : ''}${upstreamMessage ? `：${upstreamMessage}` : ''}`;
  return { message, ...(code ? { code } : {}) };
}

function createFinishReasonErrorChunk(data: any, choice: any): LLMStreamChunk {
  const { message, code } = describeFinishReasonError(data, choice);
  return {
    error: { kind: 'stream_error', message, ...(code ? { code } : {}), rawChunk: data },
    rawChunk: data,
    finishReason: 'error',
  };
}

function createFinishReasonErrorResponse(data: any, choice: any): LLMResponse {
  const { message, code } = describeFinishReasonError(data, choice);
  return {
    content: { role: 'model', parts: [{ text: '' }] },
    finishReason: 'error',
    error: { kind: 'response_error', message, ...(code ? { code } : {}), rawBody: data },
    rawResponse: data,
  };
}

/**
 * 非流式 tool_calls[].function.arguments 解码。
 *
 * OpenAI 文档：arguments 是模型生成的 JSON 字符串，“the model does not always generate valid JSON”
 * （https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/）。
 * 兼容实现里还会见到空串（无参数调用）或已经解析好的对象，这两种按 `{}` / 原对象接受；
 * 真正无法解析的 JSON 仍然失败，但错误信息写明工具名并提示参数可能被截断。
 */
function decodeNonStreamToolArguments(tc: any, finishReason?: string): Record<string, unknown> {
  const raw = tc?.function?.arguments;
  if (raw === undefined || raw === null) return {};
  if (typeof raw === 'object') {
    if (Array.isArray(raw)) {
      throw new Error(describeToolArgumentsProblem(tc?.function?.name, tc?.id, JSON.stringify(raw), 'not_object', finishReason));
    }
    return raw as Record<string, unknown>;
  }
  const text = String(raw);
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(describeToolArgumentsProblem(tc?.function?.name, tc?.id, text, 'incomplete', finishReason));
  }
}

function addInlineDataPart(parts: Part[], inlineData: ReturnType<typeof parseBase64DataUrl>): void {
  if (!inlineData) return;
  parts.push({ inlineData });
}

function parseOpenAICompatibleFilePart(item: any): ReturnType<typeof parseBase64DataUrl> | undefined {
  const file = item.file && typeof item.file === 'object' ? item.file : item;
  const inlineData = parseBase64DataUrl(file.file_data ?? file.data ?? item.file_data ?? item.data);
  if (!inlineData) return undefined;
  return isToolResponseDocumentMimeType(inlineData.mimeType) ? inlineData : undefined;
}

function encodeOpenAICompatibleToolMediaBlock(part: NonNullable<FunctionResponsePart['functionResponse']['parts']>[number]): Record<string, unknown> | undefined {
  const mime = part.inlineData.mimeType;
  if (isToolResponseImageMimeType(mime)) {
    return {
      type: 'image_url',
      image_url: {
        url: toBase64DataUrl(part.inlineData),
      },
    };
  }
  if (isToolResponseDocumentMimeType(mime)) {
    return {
      type: 'file',
      file: {
        file_data: toBase64DataUrl(part.inlineData),
      },
    };
  }
  return undefined;
}

function encodeOpenAICompatibleToolResultContent(response: FunctionResponsePart['functionResponse']): unknown {
  const text = JSON.stringify(response.response);
  const mediaBlocks = (response.parts ?? [])
    .map(part => encodeOpenAICompatibleToolMediaBlock(part))
    .filter((block): block is Record<string, unknown> => !!block);

  if (mediaBlocks.length === 0) return text;
  return [
    { type: 'text', text },
    ...mediaBlocks,
  ];
}

function appendOpenAIContentBlock(parts: Part[], block: unknown): void {
  if (typeof block === 'string') {
    parts.push({ text: block });
    return;
  }
  if (!block || typeof block !== 'object') return;
  const item = block as any;
  if (item.type === 'text' && typeof item.text === 'string') {
    parts.push({ text: item.text });
  } else if (item.type === 'image_url') {
    addInlineDataPart(parts, parseBase64DataUrl(item.image_url?.url));
  } else if (item.type === 'file') {
    addInlineDataPart(parts, parseOpenAICompatibleFilePart(item));
  }
}
