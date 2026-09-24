import { describe, expect, it, vi } from 'vitest';

import { OpenAICompatibleFormat, createOpenAICompatibleProvider } from '../src/index.js';
import type { LLMStreamChunk } from '../src/index.js';
import {
  chatChunk,
  decodeJson,
  decodeStream,
  functionCallsOf,
  plain,
  readBaseline,
  readFixture,
  sse,
} from './openai-compatible-helpers.js';

/**
 * A1：流结束时补发工具调用；参数被截断时上报解码错误。
 * A2：非流式 arguments 为空串/对象时容错；流式 delta 缺 index 时按 id 区分并行调用。
 *
 * 依据：OpenAI Chat Completions 参考
 * https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
 * （function.arguments：“the model does not always generate valid JSON”）以及网关实测抓包。
 */
describe('OpenAI 兼容流：结束时补发工具调用（A1）', () => {
  it('实测 Claude 流没有 finish_reason 时，无参数调用在流结束后以 {} 发出，且之前的块与修复前一致', async () => {
    const chunks = await decodeStream(readFixture('claude-noargs.sse'));
    const baseline = readBaseline('claude-noargs.sse') as LLMStreamChunk[];

    expect(plain(chunks.slice(0, baseline.length))).toEqual(baseline);
    expect(chunks).toHaveLength(baseline.length + 1);
    const finalChunk = chunks[chunks.length - 1];
    expect(plain(finalChunk)).toEqual({
      functionCalls: [{ functionCall: { name: 'list_items', args: {}, callId: 'toolu_012k3H6m6miaNTAcvNLTvN6d' } }],
      partsDelta: [{ functionCall: { name: 'list_items', args: {}, callId: 'toolu_012k3H6m6miaNTAcvNLTvN6d' } }],
    });
    expect(finalChunk.finishReason).toBeUndefined();
    expect(finalChunk.error).toBeUndefined();
  });

  it('实测两个并行无参数调用：第二个（最后一个）调用不再丢失', async () => {
    const chunks = await decodeStream(readFixture('claude-two-noargs.sse'));
    const baseline = readBaseline('claude-two-noargs.sse') as LLMStreamChunk[];

    expect(plain(chunks.slice(0, baseline.length))).toEqual(baseline);
    expect(functionCallsOf(chunks)).toEqual([
      { name: 'list_items', args: {}, callId: 'toolu_0158ATFP4c5wLpPvPCDUvDQU' },
      { name: 'get_time', args: {}, callId: 'toolu_01A6wnohPcKJ1eTdfqedhBMi' },
    ]);
  });

  it.each([
    'tools-gpt-5.5.sse',
    'tools-claude-sonnet-5.sse',
    'tools-v-gemini-3.5-flash.sse',
    'gpt-noargs.sse',
  ])('原行为不变：%s 的解码结果与修复前逐块一致，不追加补发块', async (name) => {
    const chunks = await decodeStream(readFixture(name));
    expect(plain(chunks)).toEqual(readBaseline(name));
  });

  it('经 provider.chatStream（unified 输出）也能收到补发的调用', async () => {
    const fetch = vi.fn(async () => new Response(readFixture('claude-noargs.sse'), {
      status: 200,
      headers: { 'content-type': 'text/event-stream' },
    }));
    const provider = createOpenAICompatibleProvider({
      provider: 'openai-compatible',
      model: 'claude-sonnet-5',
      apiKey: 'sk-test',
      baseUrl: 'https://gateway.test/v1',
      fetch: fetch as any,
    });

    const chunks: LLMStreamChunk[] = [];
    for await (const chunk of provider.chatStream({ contents: [{ role: 'user', parts: [{ text: 'call list_items' }] }] }, {
      inputFormat: 'unified',
      outputFormat: 'unified',
    })) {
      chunks.push(chunk);
    }
    expect(functionCallsOf(chunks)).toEqual([
      { name: 'list_items', args: {}, callId: 'toolu_012k3H6m6miaNTAcvNLTvN6d' },
    ]);
  });

  it('finish_reason=length 截断的参数不再静默丢弃，返回带工具名的解码错误块', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'write_file', arguments: '' } }] }),
      chatChunk({ tool_calls: [{ index: 0, function: { arguments: '{"path":"a.txt","content":"hel' } }] }),
      chatChunk({}, { finish_reason: 'length' }),
      '[DONE]',
    ));

    const last = chunks[chunks.length - 1];
    expect(functionCallsOf(chunks)).toEqual([]);
    expect(last.finishReason).toBe('length');
    expect(last.error).toMatchObject({ kind: 'decode_error', status: 200, statusText: 'OK' });
    expect(last.error?.message).toContain('"write_file"');
    expect(last.error?.message).toContain('call_1');
    expect(last.error?.message).toContain('参数可能被截断');
    expect(last.error?.message).toContain('finish_reason: length');
    expect(last.error?.headers?.['content-type']).toBe('text/event-stream');
  });

  it('没有 finish_reason 且流结束时参数仍不完整：流结束后返回解码错误块', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":' } }] }),
      '[DONE]',
    ));

    expect(chunks).toHaveLength(2);
    expect(functionCallsOf(chunks)).toEqual([]);
    expect(chunks[1].error).toMatchObject({ kind: 'decode_error', status: 200 });
    expect(chunks[1].error?.message).toContain('"read_file"');
    expect(chunks[1].error?.message).toContain('参数可能被截断');
    expect(chunks[1].error?.message).toContain('流结束时仍未收到完整参数');
  });

  it('参数是合法 JSON 但不是对象时也上报错误，而不是静默丢弃', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'x', arguments: '[1,2]' } }] }),
      chatChunk({}, { finish_reason: 'tool_calls' }),
      '[DONE]',
    ));
    expect(chunks[chunks.length - 1].error?.message).toContain('不是 JSON 对象');
  });

  it('连接干净断开（没有 finish_reason 也没有 [DONE]）时，只收到工具名的调用按参数截断报错，不按 {} 补发', async () => {
    // Chat Completions 流以 `data: [DONE]` 结束（OpenAI OpenAPI：stream_options.include_usage
    // “an additional chunk will be streamed before the data: [DONE] message”）。没有 [DONE] 的 EOF
    // 说明流没有正常结束，只收到工具名分片（arguments:""）的调用可能还没发完参数；按 {} 补发会让
    // 参数全可选的工具（如 write_file）真的以空参数执行。
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'write_file', arguments: '' } }] }),
    ));
    expect(functionCallsOf(chunks)).toEqual([]);
    const last = chunks[chunks.length - 1];
    expect(last.error).toMatchObject({ kind: 'decode_error', status: 200 });
    expect(last.error?.message).toContain('"write_file"');
    expect(last.error?.message).toContain('call_1');
    expect(last.error?.message).toContain('参数可能被截断');
    expect(last.error?.message).toContain('[DONE]');
  });

  it('连接干净断开时，参数已完整、已在流中发出的调用不受影响，也不追加错误块', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":"a"}' } }] }),
    ));
    expect(functionCallsOf(chunks)).toEqual([{ name: 'read_file', args: { path: 'a' }, callId: 'call_1' }]);
    expect(chunks.some(chunk => chunk.error)).toBe(false);
  });

  it('收到 [DONE] 后，只收到工具名的调用仍按 {} 补发（无参数调用）', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'list_items', arguments: '' } }] }),
      '[DONE]',
    ));
    expect(functionCallsOf(chunks)).toEqual([{ name: 'list_items', args: {}, callId: 'call_1' }]);
    expect(chunks.some(chunk => chunk.error)).toBe(false);
  });

  it('读取中断时流不完整，不补发调用（只返回 stream_read_error）', async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(sse(
          chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'list_items', arguments: '' } }] }),
        )));
        controller.error(new Error('socket hang up'));
      },
    });
    const chunks = await decodeStream(body);
    expect(functionCallsOf(chunks)).toEqual([]);
    expect(chunks[chunks.length - 1].error?.kind).toBe('stream_read_error');
  });

  it('finish_reason 已输出全部调用时，流结束不再追加任何块', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'list_items', arguments: '' } }] }),
      chatChunk({}, { finish_reason: 'tool_calls' }),
      '[DONE]',
    ));
    expect(chunks).toHaveLength(2);
    expect(functionCallsOf(chunks)).toEqual([{ name: 'list_items', args: {}, callId: 'call_1' }]);
  });

  it('finalizeStream 在没有待发内容时返回 undefined', () => {
    const format = new OpenAICompatibleFormat('m');
    expect(format.finalizeStream(format.createStreamState())).toBeUndefined();
  });
});

describe('OpenAI 兼容流：delta 缺 index 时按 id 区分调用（A2）', () => {
  it('并行调用都不带 index：新 id 视为新调用，无 id 的分片续写上一个调用', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ id: 'call_a', type: 'function', function: { name: 'get_weather', arguments: '' } }] }),
      chatChunk({ tool_calls: [{ function: { arguments: '{"city":' } }] }),
      chatChunk({ tool_calls: [{ function: { arguments: '"Paris"}' } }] }),
      chatChunk({ tool_calls: [{ id: 'call_b', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Tokyo"}' } }] }),
      chatChunk({}, { finish_reason: 'tool_calls' }),
      '[DONE]',
    ));
    expect(functionCallsOf(chunks)).toEqual([
      { name: 'get_weather', args: { city: 'Paris' }, callId: 'call_a' },
      { name: 'get_weather', args: { city: 'Tokyo' }, callId: 'call_b' },
    ]);
  });

  it('同一个 delta 里两个不带 index 的调用分别解码', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({
        tool_calls: [
          { id: 'call_a', type: 'function', function: { name: 'list_items', arguments: '' } },
          { id: 'call_b', type: 'function', function: { name: 'get_time', arguments: '' } },
        ],
      }),
      '[DONE]',
    ));
    expect(functionCallsOf(chunks)).toEqual([
      { name: 'list_items', args: {}, callId: 'call_a' },
      { name: 'get_time', args: {}, callId: 'call_b' },
    ]);
  });

  it('每个分片都重复同一个 id 时仍是同一个调用', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ id: 'call_a', type: 'function', function: { name: 'read_file', arguments: '{"path"' } }] }),
      chatChunk({ tool_calls: [{ id: 'call_a', function: { arguments: ':"a.txt"}' } }] }),
      chatChunk({}, { finish_reason: 'tool_calls' }),
      '[DONE]',
    ));
    expect(functionCallsOf(chunks)).toEqual([{ name: 'read_file', args: { path: 'a.txt' }, callId: 'call_a' }]);
  });

  it('流式 arguments 直接给出对象时按完整参数处理', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_a', type: 'function', function: { name: 'read_file', arguments: { path: 'a.txt' } } }] }),
      '[DONE]',
    ));
    expect(functionCallsOf(chunks)).toEqual([{ name: 'read_file', args: { path: 'a.txt' }, callId: 'call_a' }]);
  });
});

describe('OpenAI 兼容非流式 tool_calls 参数容错（A2）', () => {
  const response = (toolCalls: unknown[]) => ({
    id: 'chatcmpl-test',
    object: 'chat.completion',
    choices: [{ index: 0, message: { role: 'assistant', content: null, tool_calls: toolCalls }, finish_reason: 'tool_calls' }],
  });

  it('arguments 为空串、null 或缺失时按 {} 解码', async () => {
    const decoded = await decodeJson(response([
      { id: 'call_a', type: 'function', function: { name: 'list_items', arguments: '' } },
      { id: 'call_b', type: 'function', function: { name: 'get_time', arguments: null } },
      { id: 'call_c', type: 'function', function: { name: 'ping' } },
    ]));
    expect(decoded.error).toBeUndefined();
    expect(plain(decoded.content.parts)).toEqual([
      { functionCall: { name: 'list_items', args: {}, callId: 'call_a' } },
      { functionCall: { name: 'get_time', args: {}, callId: 'call_b' } },
      { functionCall: { name: 'ping', args: {}, callId: 'call_c' } },
    ]);
  });

  it('arguments 已经是对象时原样使用', async () => {
    const decoded = await decodeJson(response([
      { id: 'call_a', type: 'function', function: { name: 'read_file', arguments: { path: 'a.txt' } } },
    ]));
    expect(plain(decoded.content.parts)).toEqual([
      { functionCall: { name: 'read_file', args: { path: 'a.txt' }, callId: 'call_a' } },
    ]);
  });

  it('无法解析的 arguments 仍返回 decode_error，但错误信息写明工具名与截断提示', async () => {
    const decoded = await decodeJson({
      ...response([{ id: 'call_a', type: 'function', function: { name: 'write_file', arguments: '{"path":"a' } }]),
      choices: [{
        index: 0,
        message: { role: 'assistant', content: null, tool_calls: [{ id: 'call_a', type: 'function', function: { name: 'write_file', arguments: '{"path":"a' } }] },
        finish_reason: 'length',
      }],
    });
    expect(decoded.error?.kind).toBe('decode_error');
    expect(decoded.error?.message).toContain('"write_file"');
    expect(decoded.error?.message).toContain('参数可能被截断');
    expect(decoded.error?.message).toContain('finish_reason: length');
  });

  it('原行为不变：实测非流式响应（arguments 为 "{}" 字符串）解码结果与修复前一致', async () => {
    const decoded = new OpenAICompatibleFormat('m').decodeResponse(JSON.parse(readFixture('claude-noargs-nostream.json')));
    expect(plain(decoded)).toEqual(readBaseline('claude-noargs-nostream.json'));
  });
});

describe('finish_reason=length 截断的工具参数错误标为不可重试', () => {
  // OpenAI Chat Completions：finish_reason "length" 表示 “the maximum number of tokens specified in the
  // request was reached”。原样重发会再次在同一输出上限处截断，所以错误带 retryable:false，
  // 调用方的重试策略不应整包重发。其他原因的解析失败（没有 finish_reason、tool_calls 等）不标记。
  it('流式：finish_reason=length 的截断错误带 retryable:false', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'write_file', arguments: '{"path":"a.txt","content":"hel' } }] }),
      chatChunk({}, { finish_reason: 'length' }),
      '[DONE]',
    ));
    expect(chunks.at(-1)?.error).toMatchObject({ kind: 'decode_error', retryable: false });
  });

  it('流式：没有 finish_reason 的截断（流结束时参数仍不完整）不标记 retryable', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"path":' } }] }),
      '[DONE]',
    ));
    expect(chunks.at(-1)?.error?.kind).toBe('decode_error');
    expect(chunks.at(-1)?.error).not.toHaveProperty('retryable');
  });

  it('非流式：finish_reason=length 且参数无法解析时 decode_error 带 retryable:false', async () => {
    const decoded = await decodeJson({
      id: 'chatcmpl-test',
      object: 'chat.completion',
      choices: [{
        index: 0,
        message: { role: 'assistant', content: null, tool_calls: [{ id: 'call_a', type: 'function', function: { name: 'write_file', arguments: '{"path":"a' } }] },
        finish_reason: 'length',
      }],
    });
    expect(decoded.error).toMatchObject({ kind: 'decode_error', retryable: false });
  });

  it('非流式：其他原因的参数解析失败不标记 retryable', async () => {
    const decoded = await decodeJson({
      id: 'chatcmpl-test',
      object: 'chat.completion',
      choices: [{
        index: 0,
        message: { role: 'assistant', content: null, tool_calls: [{ id: 'call_a', type: 'function', function: { name: 'write_file', arguments: '{"path":"a' } }] },
        finish_reason: 'tool_calls',
      }],
    });
    expect(decoded.error?.kind).toBe('decode_error');
    expect(decoded.error).not.toHaveProperty('retryable');
  });
});
