/**
 * OpenAI Compatible 格式适配器
 *
 * Gemini ↔ OpenAI 格式的完整双向转换。
 * 适用于所有 OpenAI 兼容接口（OpenAI、DeepSeek、本地模型等）。
 *
 * 支持 reasoning_content（DeepSeek / KIMI 等模型的 thinking 字段），以及 OpenRouter 的
 * reasoning / reasoning_details（https://openrouter.ai/docs/use-cases/reasoning-tokens）。
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
        const reasoningReplay = collectReasoningReplay(thoughtParts);

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
          applyReasoningReplay(msg, reasoningContent, reasoningReplay);
          messages.push(msg);
       } else {
          const text = textParts.map(p => {
            if (!isTextPart(p)) throw new Error('unreachable');
            return p.text;
          }).join('');
          const msg: Record<string, unknown> = { role: 'assistant', content: text };
          applyReasoningReplay(msg, reasoningContent, reasoningReplay);
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
    // OpenRouter：没有 reasoning_content 时取 message.reasoning；reasoning_details 原样存进签名信封。
    const reasoningFromReasoningField = !nonEmptyString(msg.reasoning_content) && nonEmptyString(msg.reasoning);
    const reasoningDetails = Array.isArray(msg.reasoning_details) && msg.reasoning_details.length > 0
      ? msg.reasoning_details as unknown[]
      : undefined;
    const replaySignature = reasoningDetails || reasoningFromReasoningField
      ? serializeReasoningReplay({
          reasoning_details: reasoningDetails,
          reasoning_signature: nonEmptyString(msg.reasoning_signature) ? msg.reasoning_signature : undefined,
          reasoning_field: reasoningFromReasoningField ? 'reasoning' : undefined,
        })
      : undefined;
    if ((typeof msg.reasoning_content === 'string' && msg.reasoning_content) || typeof msg.reasoning_signature === 'string' || replaySignature) {
      const text = typeof msg.reasoning_content === 'string' && msg.reasoning_content
        ? msg.reasoning_content
        : reasoningFromReasoningField ? msg.reasoning : typeof msg.reasoning_content === 'string' ? msg.reasoning_content : '';
      const thoughtPart: Part = {
        text,
        thought: true,
        ...(replaySignature
          ? { thoughtSignature: replaySignature }
          : typeof msg.reasoning_signature === 'string' ? { thoughtSignature: msg.reasoning_signature } : {}),
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

    const reasoningState = state as OpenAICompatibleStreamState;

    // reasoning_content 流式增量（DeepSeek / KIMI 等模型的 thinking 输出）
    if (choice?.delta?.reasoning_content) {
      chunk.partsDelta = [
        ...(chunk.partsDelta || []),
        { text: choice.delta.reasoning_content, thought: true } as any,
      ];
    } else if (nonEmptyString(choice?.delta?.reasoning)) {
      // OpenRouter 的思考文本字段；同一块已有 reasoning_content 时不重复取。
      reasoningState.reasoningField = 'reasoning';
      chunk.partsDelta = [
        ...(chunk.partsDelta || []),
        { text: choice.delta.reasoning, thought: true } as any,
      ];
    }

    // OpenRouter reasoning_details：只累积，流结束时整体作为签名信封发出（见 finalizeStream）。
    if (Array.isArray(choice?.delta?.reasoning_details)) {
      accumulateReasoningDetails(reasoningState.reasoningDetails, choice.delta.reasoning_details);
    }

    if (choice?.delta?.reasoning_signature) {
      chunk.partsDelta = [
        ...(chunk.partsDelta || []),
        { thought: true, thoughtSignature: choice.delta.reasoning_signature } as any,
      ];
      chunk.thoughtSignature = choice.delta.reasoning_signature;
      if (typeof choice.delta.reasoning_signature === 'string') {
        reasoningState.reasoningSignature = choice.delta.reasoning_signature;
      }
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
      reasoningDetails: [],
    };
    return state;
  }

  /**
   * 流结束（[DONE] / EOF）后补发仍在等待的内容。
   *
   * 1. 工具调用：部分网关（实测：把 Chat Completions 转成其他协议的中转）在工具调用流里不发
   *    finish_reason，最后一个调用（尤其是 arguments 为空串的无参数调用）永远等不到输出信号。
   *    流已正常结束时，空参数按 `{}` 发出；参数 JSON 不完整的调用视为被截断，返回解码错误块。
   * 2. OpenRouter reasoning_details：整条流累积完成后才完整（签名、加密块常在正文或工具调用之后
   *    才到），所以只在这里作为一个仅含签名的思考 part 发出一次，与官方 AI SDK 在 finish 时给出
   *    完整 reasoning_details 的做法一致。
   */
  finalizeStream(state: StreamDecodeState): LLMStreamChunk | undefined {
    const streamState = state as OpenAICompatibleStreamState;
    if (streamState.terminatedWithError) return undefined;
    const pending = streamState.pendingToolCalls;

    const chunk: LLMStreamChunk = {};
    const replaySignature = streamState.reasoningDetails.length > 0 || streamState.reasoningField
      ? serializeReasoningReplay({
          reasoning_details: streamState.reasoningDetails.length > 0 ? streamState.reasoningDetails : undefined,
          reasoning_signature: streamState.reasoningSignature,
          reasoning_field: streamState.reasoningField,
        })
      : undefined;
    if (replaySignature) {
      chunk.partsDelta = [{ text: '', thought: true, thoughtSignature: replaySignature } as Part];
      chunk.thoughtSignature = replaySignature;
    }

    if (pending.size > 0) {
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
    }
    return chunk.partsDelta?.length ? chunk : undefined;
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
  /** 按 OpenRouter 规则累积的 reasoning_details。 */
  reasoningDetails: Record<string, unknown>[];
  /** 思考文本来自 `reasoning` 字段（而不是 reasoning_content）。 */
  reasoningField?: 'reasoning';
  /** 最近一次 delta.reasoning_signature，与 reasoning_details 一起放进签名信封。 */
  reasoningSignature?: string;
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

// ============ OpenRouter reasoning / reasoning_details ============

/**
 * 放在思考 part 的 `thoughtSignatures['openai-compatible']` 里的回放信封（JSON 对象字符串）。
 *
 * 扩展侧每个 part 只保存一个便携签名 `openai-compatible:<value>`，因此把回放所需的全部状态
 * 编成一个 JSON 对象：
 *   - reasoning_details：OpenRouter 返回的 reasoning_details，原样保存、原样回放
 *     （文档：“Pass back unmodified”；Gemini 3 的思考签名就在其中的 reasoning.encrypted 里）；
 *   - reasoning_signature：同一响应里若还带了旧的 reasoning_signature，一并保留；
 *   - reasoning_field："reasoning" 表示思考文本来自 `reasoning` 字段而不是 reasoning_content。
 * 纯字符串签名（不是这种 JSON 对象）仍按 reasoning_signature 处理，行为不变。
 */
interface ReasoningReplay {
  reasoning_details?: unknown[];
  reasoning_signature?: string;
  reasoning_field?: 'reasoning';
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

function serializeReasoningReplay(replay: ReasoningReplay): string {
  const envelope: Record<string, unknown> = {};
  if (replay.reasoning_details) envelope.reasoning_details = replay.reasoning_details;
  if (replay.reasoning_signature) envelope.reasoning_signature = replay.reasoning_signature;
  if (replay.reasoning_field) envelope.reasoning_field = replay.reasoning_field;
  return JSON.stringify(envelope);
}

function parseReasoningReplay(value: string): ReasoningReplay | undefined {
  const trimmed = value.trim();
  if (!trimmed.startsWith('{')) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return undefined;
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined;
  const envelope = parsed as Record<string, unknown>;
  const details = Array.isArray(envelope.reasoning_details) ? envelope.reasoning_details : undefined;
  const field = envelope.reasoning_field === 'reasoning' ? 'reasoning' as const : undefined;
  if (!details && !field) return undefined;
  return {
    ...(details ? { reasoning_details: details } : {}),
    ...(nonEmptyString(envelope.reasoning_signature) ? { reasoning_signature: envelope.reasoning_signature } : {}),
    ...(field ? { reasoning_field: field } : {}),
  };
}

/** 从一条 model 内容的思考 parts 里取回放状态：第一个信封 + 第一个纯字符串签名。 */
function collectReasoningReplay(thoughtParts: Part[]): { replay?: ReasoningReplay; signature?: string } {
  let replay: ReasoningReplay | undefined;
  let signature: string | undefined;
  for (const part of thoughtParts) {
    const value = (part as any).thoughtSignatures?.['openai-compatible'];
    if (typeof value !== 'string' || !value.trim()) continue;
    const parsed = parseReasoningReplay(value);
    if (parsed) replay ??= parsed;
    else signature ??= value;
  }
  return { replay, signature };
}

/**
 * 把思考文本与回放状态写回 assistant 消息。
 *
 * 没有信封时与修复前完全一致：reasoning_content + reasoning_signature。
 * 有 reasoning_details 时原样放回 `reasoning_details`（OpenRouter 文档 “Preserving reasoning”）。
 * 思考文本来自 `reasoning` 字段时：有 reasoning_details 就用 `reasoning` 回传文本（与 OpenRouter
 * 官方 AI SDK 一致）；没有 reasoning_details 时不回传文本——这些接口（如 vLLM / Groq）的思考
 * 不需要回放，修复前也从未发送过，避免向不认识 reasoning_content 的接口发送未知字段。
 */
function applyReasoningReplay(
  msg: Record<string, unknown>,
  reasoningText: string | null,
  state: { replay?: ReasoningReplay; signature?: string },
): void {
  const { replay } = state;
  if (replay?.reasoning_field === 'reasoning') {
    if (reasoningText && replay.reasoning_details) msg.reasoning = reasoningText;
  } else if (reasoningText) {
    msg.reasoning_content = reasoningText;
  }
  const signature = replay?.reasoning_signature ?? state.signature;
  if (signature) msg.reasoning_signature = signature;
  if (replay?.reasoning_details) msg.reasoning_details = replay.reasoning_details;
}

const REASONING_TEXT = 'reasoning.text';
const REASONING_SUMMARY = 'reasoning.summary';

/**
 * 流式 reasoning_details 累积。
 *
 * OpenRouter 文档只说明“The complete reasoning sequence is built by concatenating all chunks in
 * order”。具体规则取自 OpenRouter 官方 AI SDK provider（github.com/OpenRouterTeam/ai-sdk-provider，
 * src/chat/index.ts，commit 1b22b05）：相邻的同类型 reasoning.text 合并（text 拼接，signature /
 * format 取先到的非空值），相邻的 reasoning.summary 合并（summary 拼接），reasoning.encrypted
 * 等其他类型是独立的不透明块，原样追加、从不合并。
 * 另外按文档里 index 的含义（“Sequential index of the reasoning detail”）：两个块都带数字 index
 * 且不同时视为不同的块，不合并；SDK 测试里同一块的增量总是同一个 index，结果与 SDK 一致。
 */
function accumulateReasoningDetails(target: Record<string, unknown>[], incoming: unknown[]): void {
  for (const raw of incoming) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue;
    const detail = raw as Record<string, unknown>;
    const last = target[target.length - 1];
    const mergeable = last !== undefined
      && last.type === detail.type
      && (detail.type === REASONING_TEXT || detail.type === REASONING_SUMMARY)
      && (typeof last.index !== 'number' || typeof detail.index !== 'number' || last.index === detail.index);
    if (!mergeable) {
      target.push({ ...detail });
      continue;
    }
    if (detail.type === REASONING_TEXT) {
      last.text = String(last.text || '') + String(detail.text || '');
      last.signature = last.signature || detail.signature;
    } else {
      last.summary = String(last.summary || '') + String(detail.summary || '');
    }
    last.format = last.format || detail.format;
  }
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
