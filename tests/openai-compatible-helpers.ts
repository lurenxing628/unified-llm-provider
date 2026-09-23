/**
 * OpenAI 兼容格式测试的共用夹具与工具。
 *
 * fixtures/openai-compatible/*.sse / *.json：从第三方 OpenAI 兼容网关实测抓取的真实响应体
 * （网关会把 Chat Completions 转成其他上游协议；只保存响应体，不含任何密钥）：
 *   - claude-noargs.sse：Claude 流，无参数工具调用 arguments:""，且没有 finish_reason；
 *   - claude-two-noargs.sse：两个并行无参数调用，同样没有 finish_reason；
 *   - tools-*.sse、gpt-noargs.sse：gpt-5.5 / claude-sonnet-5 / gemini-3.5-flash 的正常工具调用流。
 * fixtures/openai-compatible/baseline-16b50ab.json：修复前（16b50ab）的解码输出，用于证明
 * 原本正常的流解码结果逐字节不变。
 */
import { readFileSync } from 'node:fs';

import { OpenAICompatibleFormat, processResponse, processStreamResponse } from '../src/index.js';
import type { LLMResponse, LLMStreamChunk } from '../src/index.js';

const fixtureDir = new URL('./fixtures/openai-compatible/', import.meta.url);

export function readFixture(name: string): string {
  return readFileSync(new URL(name, fixtureDir), 'utf8');
}

export function readBaseline(name: string): unknown {
  const baseline = JSON.parse(readFixture('baseline-16b50ab.json')) as Record<string, unknown>;
  if (!(name in baseline)) throw new Error(`baseline 中没有 ${name}`);
  return baseline[name];
}

/** 把一组 SSE data 负载拼成 SSE 文本；字符串原样作为 data 行。 */
export function sse(...payloads: unknown[]): string {
  return payloads
    .map(payload => `data: ${typeof payload === 'string' ? payload : JSON.stringify(payload)}\n\n`)
    .join('');
}

export function chatChunk(delta: Record<string, unknown>, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'chatcmpl-test',
    object: 'chat.completion.chunk',
    created: 1,
    model: 'test-model',
    choices: [{ index: 0, delta, finish_reason: null, ...extra }],
  };
}

export async function decodeStream(
  body: string | ReadableStream<Uint8Array>,
  format = new OpenAICompatibleFormat('test-model'),
): Promise<LLMStreamChunk[]> {
  const res = new Response(body, { status: 200, statusText: 'OK', headers: { 'content-type': 'text/event-stream' } });
  const chunks: LLMStreamChunk[] = [];
  for await (const chunk of processStreamResponse(res, format)) chunks.push(chunk);
  return chunks;
}

export async function decodeJson(
  body: unknown,
  format = new OpenAICompatibleFormat('test-model'),
): Promise<LLMResponse> {
  const res = new Response(typeof body === 'string' ? body : JSON.stringify(body), {
    status: 200,
    statusText: 'OK',
    headers: { 'content-type': 'application/json' },
  });
  return processResponse(res, format);
}

/** 与 baseline JSON 比较前去掉 undefined 字段。 */
export function plain<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export function functionCallsOf(chunks: LLMStreamChunk[]): Array<{ name: string; args: Record<string, unknown>; callId?: string }> {
  return chunks.flatMap(chunk => (chunk.functionCalls ?? []).map(part => plain(part.functionCall)));
}
