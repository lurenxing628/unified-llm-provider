import { describe, expect, it } from 'vitest';

import { OpenAICompatibleFormat, decodeRequestFromFormat } from '../src/index.js';
import type { Content, LLMRequest } from '../src/index.js';
import { readFixture } from './openai-compatible-helpers.js';

/**
 * A5：同一条用户内容里既有工具结果又有文字/图片时，文字/图片不再丢弃。
 * A6：openai-compatible 的 tool 消息只放文字，工具结果里的图片/文件移到这一批 tool 消息之后的 user 消息。
 *
 * 依据：OpenAI Chat Completions 参考
 * https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
 * ——tool 消息 content 数组 “For tool messages, only type `text` is supported”（网关实测对图片块返回 400）；
 * assistant 的 tool_calls 之后必须紧跟对每个 tool_call_id 的 tool 消息。
 * DeepSeek API 文档（https://api-docs.deepseek.com/api/create-chat-completion）允许 tool 消息 content
 * 为含 image_url / file 的数组，因此 deepseek 的工具结果媒体保持在 tool 消息里。
 */

const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
const PDF = Buffer.from('%PDF-1.4 test').toString('base64');
const image = { type: 'image_url', image_url: { url: `data:image/png;base64,${PNG}` } };
const file = { type: 'file', file: { file_data: `data:application/pdf;base64,${PDF}` } };

function encode(contents: Content[], kind: 'openai-compatible' | 'deepseek' = 'openai-compatible'): any[] {
  const request = decodeRequestFromFormat({ contents } satisfies LLMRequest, { format: 'unified' });
  return (new OpenAICompatibleFormat('m', kind).encodeRequest(request, false) as any).messages;
}

const callTwo: Content = {
  role: 'model',
  parts: [
    { functionCall: { name: 'screenshot', args: {}, callId: 'call_a' } },
    { functionCall: { name: 'read_pdf', args: {}, callId: 'call_b' } },
  ],
};

describe('openai-compatible 工具结果媒体移出 tool 消息（A6）', () => {
  it('实测夹具：tool 消息只剩文字，图片放到其后的 user 消息并注明对应的调用；其余消息与修复前一致', () => {
    const requests = JSON.parse(readFixture('requests.json')) as Record<string, LLMRequest>;
    const baseline = JSON.parse(readFixture('baseline-16b50ab-requests.json')) as Record<string, any>;
    const request = decodeRequestFromFormat(requests['tool-result-media'], { format: 'unified' });
    const body = new OpenAICompatibleFormat('test-model').encodeRequest(request, true) as any;
    const before = baseline['tool-result-media|openai-compatible|stream'];

    expect(body.messages.slice(0, 2)).toEqual(before.messages.slice(0, 2));
    expect(body.messages[2]).toEqual({ role: 'tool', tool_call_id: 'call_s', content: '{"ok":true}' });
    expect(body.messages[3]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: '[The following attachment belongs to the result of tool call "screenshot" (tool_call_id: call_s).]' },
        image,
      ],
    });
    expect(body.messages).toHaveLength(4);
    const { messages: _messages, ...rest } = body;
    const { messages: _beforeMessages, ...beforeRest } = before;
    expect(rest).toEqual(beforeRest);
  });

  it('一批并行工具结果（分成多条内容）：全部 tool 消息相邻，媒体合并成其后的一条 user 消息', () => {
    const messages = encode([
      { role: 'user', parts: [{ text: 'go' }] },
      callTwo,
      { role: 'user', parts: [{ functionResponse: { name: 'screenshot', response: { ok: 1 }, callId: 'call_a', parts: [{ inlineData: { mimeType: 'image/png', data: PNG } }] } }] },
      { role: 'user', parts: [{ functionResponse: { name: 'read_pdf', response: { ok: 2 }, callId: 'call_b', parts: [{ inlineData: { mimeType: 'application/pdf', data: PDF } }] } }] },
      { role: 'model', parts: [{ text: 'done' }] },
    ]);
    expect(messages.map(message => message.role)).toEqual(['user', 'assistant', 'tool', 'tool', 'user', 'assistant']);
    expect(messages[2]).toEqual({ role: 'tool', tool_call_id: 'call_a', content: '{"ok":1}' });
    expect(messages[3]).toEqual({ role: 'tool', tool_call_id: 'call_b', content: '{"ok":2}' });
    expect(messages[4].content).toEqual([
      { type: 'text', text: '[The following attachment belongs to the result of tool call "screenshot" (tool_call_id: call_a).]' },
      image,
      { type: 'text', text: '[The following attachment belongs to the result of tool call "read_pdf" (tool_call_id: call_b).]' },
      file,
    ]);
  });

  it('不支持的媒体类型（如音频）照旧忽略，不产生额外的 user 消息', () => {
    const messages = encode([
      { role: 'user', parts: [{ text: 'go' }] },
      { role: 'model', parts: [{ functionCall: { name: 'record', args: {}, callId: 'call_a' } }] },
      { role: 'user', parts: [{ functionResponse: { name: 'record', response: { ok: 1 }, callId: 'call_a', parts: [{ inlineData: { mimeType: 'audio/wav', data: 'AAAA' } }] } }] },
    ]);
    expect(messages.map(message => message.role)).toEqual(['user', 'assistant', 'tool']);
    expect(messages[2].content).toBe('{"ok":1}');
  });

  it('deepseek 保持现状：媒体仍在 tool 消息的 content 数组里', () => {
    const messages = encode([
      { role: 'user', parts: [{ text: 'go' }] },
      { role: 'model', parts: [{ functionCall: { name: 'screenshot', args: {}, callId: 'call_a' } }] },
      { role: 'user', parts: [{ functionResponse: { name: 'screenshot', response: { ok: 1 }, callId: 'call_a', parts: [{ inlineData: { mimeType: 'image/png', data: PNG } }] } }] },
    ], 'deepseek');
    expect(messages[2]).toEqual({ role: 'tool', tool_call_id: 'call_a', content: [{ type: 'text', text: '{"ok":1}' }, image] });
    expect(messages).toHaveLength(3);
  });
});

describe('工具结果旁的文字/图片不再丢弃（A5）', () => {
  it.each(['openai-compatible', 'deepseek'] as const)('%s：同一内容里的文字放到这批 tool 消息之后的 user 消息', (kind) => {
    const messages = encode([
      { role: 'user', parts: [{ text: 'go' }] },
      callTwo,
      { role: 'user', parts: [
        { functionResponse: { name: 'screenshot', response: { ok: 1 }, callId: 'call_a' } },
        { text: 'Note: the file changed on disk.' },
      ] },
      { role: 'user', parts: [{ functionResponse: { name: 'read_pdf', response: { ok: 2 }, callId: 'call_b' } }] },
    ], kind);
    expect(messages.map(message => message.role)).toEqual(['user', 'assistant', 'tool', 'tool', 'user']);
    expect(messages[4]).toEqual({ role: 'user', content: 'Note: the file changed on disk.' });
  });

  it('文字与图片按原顺序合在一条 user 消息里；思考文本与空文本不发送', () => {
    const messages = encode([
      { role: 'user', parts: [{ text: 'go' }] },
      { role: 'model', parts: [{ functionCall: { name: 'screenshot', args: {}, callId: 'call_a' } }] },
      { role: 'user', parts: [
        { text: 'before ' },
        { functionResponse: { name: 'screenshot', response: { ok: 1 }, callId: 'call_a' } },
        { text: 'hidden', thought: true },
        { text: '' },
        { inlineData: { mimeType: 'image/png', data: PNG } },
        { text: 'after' },
      ] },
    ]);
    expect(messages.map(message => message.role)).toEqual(['user', 'assistant', 'tool', 'user']);
    expect(messages[3].content).toEqual([{ type: 'text', text: 'before ' }, image, { type: 'text', text: 'after' }]);
  });

  it('其后紧跟普通 user 内容时，暂存内容先单独发出，普通内容保持原样', () => {
    const messages = encode([
      { role: 'user', parts: [{ text: 'go' }] },
      { role: 'model', parts: [{ functionCall: { name: 'screenshot', args: {}, callId: 'call_a' } }] },
      { role: 'user', parts: [
        { functionResponse: { name: 'screenshot', response: { ok: 1 }, callId: 'call_a' } },
        { text: 'extra' },
      ] },
      { role: 'user', parts: [{ text: 'next question' }] },
    ]);
    expect(messages.slice(2)).toEqual([
      { role: 'tool', tool_call_id: 'call_a', content: '{"ok":1}' },
      { role: 'user', content: 'extra' },
      { role: 'user', content: 'next question' },
    ]);
  });

  it('原行为不变：只有工具结果的内容不会多出 user 消息', () => {
    const messages = encode([
      { role: 'user', parts: [{ text: 'go' }] },
      callTwo,
      { role: 'user', parts: [
        { functionResponse: { name: 'screenshot', response: { ok: 1 }, callId: 'call_a' } },
        { functionResponse: { name: 'read_pdf', response: { ok: 2 }, callId: 'call_b' } },
      ] },
      { role: 'model', parts: [{ text: 'done' }] },
    ]);
    expect(messages.map(message => message.role)).toEqual(['user', 'assistant', 'tool', 'tool', 'assistant']);
  });
});
