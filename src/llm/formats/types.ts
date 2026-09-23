/**
 * 格式适配器接口
 *
 * 每个渠道格式（Gemini、OpenAI 等）实现此接口，
 * 负责内部 Gemini 格式 ↔ 渠道 API 格式的双向转换。
 */

import { LLMCompactResponse, LLMRequest, LLMResponse, LLMStreamChunk } from '../../types.js';

/** 流式解码跨 chunk 状态（如 OpenAI tool_call 分片累积） */
export interface StreamDecodeState {
  [key: string]: unknown;
}

export interface FormatAdapter {
  /** 编码请求：LLMRequest (Gemini) → 渠道请求体。stream=true 时可注入流式参数 */
  encodeRequest(request: LLMRequest, stream?: boolean): unknown;

  /** 解码非流式响应：渠道原始 JSON → LLMResponse (Gemini) */
  decodeResponse(raw: unknown): LLMResponse;

  /** 解码流式单块：渠道原始 JSON → LLMStreamChunk */
  decodeStreamChunk(raw: unknown, state: StreamDecodeState): LLMStreamChunk;

  /** 创建流式解码状态（每次流式调用前调用） */
  createStreamState(): StreamDecodeState;

  /**
   * 可选：流正常结束（`data: [DONE]` 或 EOF）后调用一次，返回需要补发的最后一块。
   *
   * 用于上游没有给出结束信号（如缺少 finish_reason）时，把仍在 state 里等待的内容
   * （未发出的工具调用、跨块累积的签名等）交给调用方。没有需要补发的内容时返回 undefined，
   * 此时流的输出与未实现该钩子时完全一致。读取中断（stream_read_error）时不会调用。
   */
  finalizeStream?(state: StreamDecodeState): LLMStreamChunk | undefined;

  /**
   * 可选：本格式能编码 `Content.claudeSystemMessage`（Claude 消息中段 system 消息）。
   * 只有 Claude 格式为 true；其余格式的编码入口遇到这种内容时报错（见 assertFormatAcceptsClaudeSystemMessages）。
   */
  readonly acceptsClaudeSystemMessages?: boolean;
}

/**
 * Claude 专用的消息中段 system 消息不能交给别的格式：别的编码器只认 user/model，会把它当成一条普通 user 消息发出去。
 * 这里在编码前直接报错，调用方必须只在 Claude 格式上使用它。
 */
export function assertFormatAcceptsClaudeSystemMessages(
  request: Pick<LLMRequest, 'contents'>,
  format: Pick<FormatAdapter, 'acceptsClaudeSystemMessages'>,
  formatLabel: string,
): void {
  if (format.acceptsClaudeSystemMessages === true || !Array.isArray(request.contents)) return;
  const index = request.contents.findIndex(content => content?.claudeSystemMessage !== undefined);
  if (index >= 0) {
    throw new Error(`contents[${index}] 是 Claude 专用的消息中段 system 消息，不能编码为 ${formatLabel} 请求。`);
  }
}

/** 支持独立 compact / compaction 端点的格式适配器扩展。 */
export interface CompactFormatAdapter extends FormatAdapter {
  /** 编码 compact 请求：LLMRequest (unified) → provider compact 请求体 */
  encodeCompactRequest(request: LLMRequest): unknown;
  /** 解码 compact 响应：provider 原始 JSON → unified compact 结果 */
  decodeCompactResponse(raw: unknown): LLMCompactResponse;
  /** unified compact 结果 → provider 原生 compact 响应（用于 outputFormat=provider format） */
  encodeCompactResponse?(response: LLMCompactResponse): unknown;
}

export function isCompactFormatAdapter(format: FormatAdapter): format is CompactFormatAdapter {
  return typeof (format as Partial<CompactFormatAdapter>).encodeCompactRequest === 'function'
    && typeof (format as Partial<CompactFormatAdapter>).decodeCompactResponse === 'function';
}
