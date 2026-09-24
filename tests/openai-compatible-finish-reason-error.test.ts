import { describe, expect, it, vi } from 'vitest';

import { createOpenAICompatibleProvider } from '../src/index.js';
import type { LLMStreamChunk } from '../src/index.js';
import { chatChunk, decodeJson, decodeStream, functionCallsOf, sse } from './openai-compatible-helpers.js';

/**
 * A3：finish_reason:"error" 按错误处理。
 * 依据 OpenRouter 文档 https://openrouter.ai/docs/api/reference/errors-and-debugging：
 * 200 之后发生的错误，流式以 `finish_reason: "error"` 的块终止（choice 上可带 native_finish_reason），
 * 非流式把 error 放进 choice 并带 `finish_reason: "error"`。
 */
describe('OpenAI 兼容 finish_reason:"error"（A3）', () => {
  it('流中只有 finish_reason:"error"（没有顶层 error）时返回 stream_error，并写明 native_finish_reason', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ content: 'partial ' }),
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'list_items', arguments: '' } }] }),
      chatChunk({ content: '' }, { finish_reason: 'error', native_finish_reason: 'MALFORMED_FUNCTION_CALL' }),
      '[DONE]',
    ));

    const last = chunks[chunks.length - 1];
    expect(last.finishReason).toBe('error');
    expect(last.error).toMatchObject({ kind: 'stream_error', status: 200 });
    expect(last.error?.message).toContain('finish_reason: "error"');
    expect(last.error?.message).toContain('native_finish_reason: MALFORMED_FUNCTION_CALL');
    // 被错误终止的流里未完成的调用不会在流结束时补发
    expect(functionCallsOf(chunks)).toEqual([]);
    expect(chunks.filter(chunk => chunk.error)).toHaveLength(1);
  });

  it('choice 内带 error 对象时，错误信息包含上游 message 与 code', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ content: '' }, { finish_reason: 'error', error: { code: 502, message: 'Provider disconnected mid-stream' } }),
    ));
    expect(chunks[0].error?.message).toContain('Provider disconnected mid-stream');
    expect(chunks[0].error?.code).toBe('502');
  });

  it('原行为不变：OpenRouter 文档里带顶层 error 的中途错误块仍由 response 层按 stream_error 透传', async () => {
    const payload = {
      id: 'gen-abc123',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'openai/gpt-4o',
      provider: 'OpenAI',
      error: { code: 429, message: 'Rate limit exceeded', metadata: { error_type: 'rate_limit_exceeded' } },
      choices: [{ index: 0, delta: { content: '' }, finish_reason: 'error' }],
    };
    const chunks = await decodeStream(sse(payload));
    expect(chunks).toHaveLength(1);
    expect(chunks[0].error).toMatchObject({ kind: 'stream_error', rawChunk: payload });
    expect(chunks[0].error?.message).toBeUndefined();
  });

  it('带顶层 error 的中途错误块之后，流结束时不再补发工具调用、解码错误或签名信封', async () => {
    // OpenRouter 文档的中途错误块同时带顶层 error 和 finish_reason:"error"。它由 response 层按 stream_error
    // 拦下，格式适配器看不到这一块；流此后已经失败，不能再补发任何内容。
    const payload = {
      id: 'gen-abc123',
      object: 'chat.completion.chunk',
      created: 1,
      model: 'google/gemini-3.5-flash',
      provider: 'Google',
      error: { code: 502, message: 'Provider disconnected mid-stream' },
      choices: [{ index: 0, delta: { content: '' }, finish_reason: 'error' }],
    };
    for (const end of [['[DONE]'], []]) {
      const chunks = await decodeStream(sse(
        chatChunk({ reasoning: 'thinking', reasoning_details: [{ type: 'reasoning.text', text: 'thinking', index: 0 }] }),
        chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'list_items', arguments: '' } }] }),
        chatChunk({ tool_calls: [{ index: 1, id: 'call_2', type: 'function', function: { name: 'read_file', arguments: '{"path":' } }] }),
        payload,
        ...end,
      ));
      const errors = chunks.filter(chunk => chunk.error);
      expect(errors, `end=${end.join()}`).toHaveLength(1);
      expect(errors[0].error).toMatchObject({ kind: 'stream_error', rawChunk: payload });
      expect(chunks[chunks.length - 1], `end=${end.join()}`).toBe(errors[0]);
      // list_items 在 call_2 出现时已按“下一个调用出现”规则发出；错误之后不再补发 read_file 或签名信封。
      expect(functionCallsOf(chunks).map(call => call.name)).toEqual(['list_items']);
      expect(chunks.some(chunk => chunk.thoughtSignature)).toBe(false);
    }
  });

  it('SSE data 不是 JSON 时（stream_parse_error）同样不再补发', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'list_items', arguments: '' } }] }),
      '{not json',
      '[DONE]',
    ));
    expect(chunks.filter(chunk => chunk.error).map(chunk => chunk.error?.kind)).toEqual(['stream_parse_error']);
    expect(functionCallsOf(chunks)).toEqual([]);
  });

  it('非流式 choice 带 finish_reason:"error" 时返回 response_error 而不是空的成功回复', async () => {
    const body = {
      id: 'gen-abc',
      object: 'chat.completion',
      choices: [{
        index: 0,
        message: { role: 'assistant', content: 'partial output...' },
        finish_reason: 'error',
        native_finish_reason: 'MAX_TOKENS',
        error: { code: 502, message: 'Provider disconnected mid-stream', metadata: { error_type: 'provider_unavailable' } },
      }],
    };
    const decoded = await decodeJson(body);
    expect(decoded.finishReason).toBe('error');
    expect(decoded.error).toMatchObject({ kind: 'response_error', status: 200, code: '502', rawBody: body });
    expect(decoded.error?.message).toContain('native_finish_reason: MAX_TOKENS');
    expect(decoded.error?.message).toContain('Provider disconnected mid-stream');
    expect(decoded.rawResponse).toEqual(body);
  });

  it.each(['stop', 'length', 'tool_calls', 'content_filter'])('原行为不变：finish_reason=%s 不视为错误', async (finishReason) => {
    const chunks = await decodeStream(sse(chatChunk({ content: 'hi' }, { finish_reason: finishReason }), '[DONE]'));
    expect(chunks.some(chunk => chunk.error)).toBe(false);
    expect(chunks[0].finishReason).toBe(finishReason);

    const decoded = await decodeJson({ choices: [{ index: 0, message: { role: 'assistant', content: 'hi' }, finish_reason: finishReason }] });
    expect(decoded.error).toBeUndefined();
    expect(decoded.finishReason).toBe(finishReason);
  });

  it('经 provider.chatStream（unified 输出）时错误块原样到达调用方', async () => {
    const fetch = vi.fn(async () => new Response(sse(
      chatChunk({ content: '' }, { finish_reason: 'error', native_finish_reason: 'MALFORMED_FUNCTION_CALL' }),
      '[DONE]',
    ), { status: 200, headers: { 'content-type': 'text/event-stream' } }));
    const provider = createOpenAICompatibleProvider({
      provider: 'openai-compatible',
      model: 'google/gemini-3-pro-preview',
      apiKey: 'sk-test',
      baseUrl: 'https://openrouter.test/api/v1',
      fetch: fetch as any,
    });
    const chunks: LLMStreamChunk[] = [];
    for await (const chunk of provider.chatStream({ contents: [{ role: 'user', parts: [{ text: 'hi' }] }] }, {
      inputFormat: 'unified',
      outputFormat: 'unified',
    })) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0].error?.kind).toBe('stream_error');
    expect(chunks[0].error?.message).toContain('MALFORMED_FUNCTION_CALL');
  });
});
