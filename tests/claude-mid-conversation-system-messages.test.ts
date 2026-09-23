/**
 * Claude 消息中段 system 消息（轮内系统消息 clear_at）。
 *
 * 官方依据：https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages
 * - 轮内系统消息写成 {"role":"system","clear_at":"next_user_message","content":"..."}，追加在 tool_result 之后；
 * - 只能是文本；块上不能带 cache_control，断点放在前一条 user 消息的最后一块上；
 * - 必须紧跟 user 轮，后面只能是 assistant 或数组结尾；紧跟另一条 user 消息是 400；不能是第一条；
 * - 连续多条 system 消息视为一段。
 */
import { describe, expect, it } from 'vitest';

import {
  ClaudeFormat,
  createClaudeProvider,
  createDeepSeekProvider,
  createGeminiProvider,
  createOpenAICompatibleProvider,
  createOpenAIResponsesProvider,
  decodeRequestFromFormat,
  encodeRequestToFormat,
} from '../src/index.js';
import type { Content, LLMRequest } from '../src/index.js';

const reminder = (text: string): Content => ({
  role: 'user',
  parts: [{ text }],
  claudeSystemMessage: { clearAt: 'next_user_message' },
});

/** 官方文档的工具循环例子：每个 tool_result 之后一条轮内提醒，之前的副本原样留在原位。 */
const TOOL_LOOP: LLMRequest = {
  systemInstruction: { parts: [{ text: 'You are a coding agent.' }] },
  tools: [{ functionDeclarations: [{ name: 'read_file', description: 'Read', parameters: { type: 'object', properties: { path: { type: 'string' } } } }] }],
  contents: [
    { role: 'user', parts: [{ text: 'Fix the failing test.' }] },
    reminder('Request independent reads in one turn.'),
    { role: 'model', parts: [
      { text: '', thought: true, thoughtSignatures: { claude: 'sig-1' } },
      { functionCall: { name: 'read_file', args: { path: 'test_auth.py' }, callId: 'toolu_01' } },
    ] },
    { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { content: '...' }, callId: 'toolu_01' } }] },
    reminder('Request independent reads in one turn.'),
    { role: 'model', parts: [
      { text: '', thought: true, thoughtSignatures: { claude: 'sig-2' } },
      { functionCall: { name: 'read_file', args: { path: 'auth.py' }, callId: 'toolu_02' } },
    ] },
    { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { content: '...' }, callId: 'toolu_02' } }] },
    reminder('The shell exited with status 137.'),
  ],
};

describe('Claude 轮内系统消息编码', () => {
  it('按原位置编码为 role:"system" + clear_at，内容是字符串，不与相邻 tool_result 消息合并', () => {
    const body = new ClaudeFormat('claude-fable-5-1').encodeRequest(TOOL_LOOP, true) as any;
    expect(body.messages.map((message: any) => message.role)).toEqual(
      ['user', 'system', 'assistant', 'user', 'system', 'assistant', 'user', 'system'],
    );
    expect(body.messages[1]).toEqual({ role: 'system', clear_at: 'next_user_message', content: 'Request independent reads in one turn.' });
    expect(body.messages[4]).toEqual({ role: 'system', clear_at: 'next_user_message', content: 'Request independent reads in one turn.' });
    expect(body.messages[7]).toEqual({ role: 'system', clear_at: 'next_user_message', content: 'The shell exited with status 137.' });
    expect(body.messages[6].content).toEqual([{ type: 'tool_result', tool_use_id: 'toolu_02', content: '{"content":"..."}' }]);
  });

  it('同一历史两次编码逐字节相同；追加一轮后前面的消息原样不变', () => {
    const format = new ClaudeFormat('claude-fable-5-1');
    const first = JSON.stringify(format.encodeRequest(TOOL_LOOP, true));
    expect(JSON.stringify(format.encodeRequest(structuredClone(TOOL_LOOP), true))).toBe(first);

    const next: LLMRequest = {
      ...TOOL_LOOP,
      contents: [
        ...TOOL_LOOP.contents,
        { role: 'model', parts: [{ text: 'Done.' }] },
        { role: 'user', parts: [{ text: 'thanks' }] },
        reminder('Next turn reminder.'),
      ],
    };
    const before = (format.encodeRequest(TOOL_LOOP, true) as any).messages;
    const after = (format.encodeRequest(next, true) as any).messages;
    expect(after.slice(0, before.length)).toEqual(before);
  });

  it('缓存断点：消息断点落在最后一条 user 消息（提醒之前）的最后一块上；system 与 tools 断点不变；提醒不带 cache_control', () => {
    const body = new ClaudeFormat('claude-fable-5-1', { enabled: true, ttl: '1h' }).encodeRequest(TOOL_LOOP, true) as any;
    const cacheControl = { type: 'ephemeral', ttl: '1h' };
    expect(body.tools[0].cache_control).toEqual(cacheControl);
    expect(body.system).toEqual([{ type: 'text', text: 'You are a coding agent.', cache_control: cacheControl }]);
    expect(body.messages[6].content[0].cache_control).toEqual(cacheControl);
    const marked = body.messages.flatMap((message: any, index: number) => Array.isArray(message.content)
      ? message.content.filter((block: any) => block.cache_control).map(() => index)
      : []);
    expect(marked).toEqual([6]);
    for (const message of body.messages.filter((candidate: any) => candidate.role === 'system')) {
      expect(typeof message.content).toBe('string');
      expect(message).not.toHaveProperty('cache_control');
    }
  });

  it('自动缓存只写顶层 cache_control，不改动轮内系统消息（服务端选断点时跳过它）', () => {
    const body = new ClaudeFormat('claude-fable-5-1', false, true).encodeRequest(TOOL_LOOP, true) as any;
    expect(body.cache_control).toEqual({ type: 'ephemeral' });
    expect(body.messages[7]).toEqual({ role: 'system', clear_at: 'next_user_message', content: 'The shell exited with status 137.' });
  });

  it('clearAt 缺省或 never 时不写 clear_at；多个文本 part 写成文本块数组', () => {
    const body = new ClaudeFormat('claude-opus-5-5').encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'hi' }] },
        { role: 'user', parts: [{ text: 'a' }, { text: 'b' }], claudeSystemMessage: {} },
        { role: 'user', parts: [{ text: 'c' }], claudeSystemMessage: { clearAt: 'never' } },
      ],
    }) as any;
    expect(body.messages.slice(1)).toEqual([
      { role: 'system', content: [{ type: 'text', text: 'a' }, { type: 'text', text: 'b' }] },
      { role: 'system', content: 'c' },
    ]);
  });

  it('面对的 assistant 消息没有可编码内容而被省略、后面紧跟 user 消息时，这段轮内系统消息整段去掉', () => {
    const body = new ClaudeFormat('claude-opus-5-5').encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'q1' }] },
        reminder('stale'),
        { role: 'model', parts: [{ text: '' }] },
        { role: 'user', parts: [{ text: 'q2' }] },
        reminder('current'),
      ],
    }) as any;
    expect(body.messages).toEqual([
      { role: 'user', content: 'q1' },
      { role: 'user', content: 'q2' },
      { role: 'system', clear_at: 'next_user_message', content: 'current' },
    ]);
  });

  it('违反位置规则的调用直接报错：不能是第一条、不能跟在 assistant 后；普通 system 消息后面不能是 user', () => {
    const format = new ClaudeFormat('claude-opus-5-5');
    expect(() => format.encodeRequest({ contents: [reminder('first')] })).toThrow(/紧跟一条 user 消息/);
    expect(() => format.encodeRequest({ contents: [
      { role: 'user', parts: [{ text: 'q' }] },
      { role: 'model', parts: [{ functionCall: { name: 'f', args: {}, callId: 'toolu_1' } }] },
      reminder('between tool_use and tool_result'),
      { role: 'user', parts: [{ functionResponse: { name: 'f', response: {}, callId: 'toolu_1' } }] },
    ] })).toThrow(/紧跟一条 user 消息/);
    expect(() => format.encodeRequest({ contents: [
      { role: 'user', parts: [{ text: 'q' }] },
      { role: 'user', parts: [{ text: 'policy' }], claudeSystemMessage: {} },
      { role: 'user', parts: [{ text: 'q2' }] },
    ] })).toThrow(/只能是 assistant/);
    expect(() => format.encodeRequest({ contents: [
      { role: 'user', parts: [{ text: 'q' }] },
      { role: 'user', parts: [{ inlineData: { mimeType: 'image/png', data: 'AA==' } }], claudeSystemMessage: { clearAt: 'next_user_message' } },
    ] })).toThrow(/只能包含文本/);
    expect(() => format.encodeRequest({ contents: [
      { role: 'user', parts: [{ text: 'q' }] },
      { role: 'model', parts: [{ text: 'm' }], claudeSystemMessage: { clearAt: 'next_user_message' } },
    ] })).toThrow(/role: "user"/);
  });

  it('Claude 请求里的 system 消息解码为 Claude 专用内容，并能原样编回', () => {
    const wire = new ClaudeFormat('claude-fable-5-1').encodeRequest(TOOL_LOOP, false) as any;
    const decoded = decodeRequestFromFormat(wire, { format: 'claude' });
    expect(decoded.contents[1]).toEqual({ role: 'user', parts: [{ text: 'Request independent reads in one turn.' }], claudeSystemMessage: { clearAt: 'next_user_message' } });
    const again = encodeRequestToFormat(decoded, { format: 'claude', sourceFormat: 'claude' }) as any;
    expect(again.messages.filter((message: any) => message.role === 'system')).toEqual(
      wire.messages.filter((message: any) => message.role === 'system'),
    );
  });
});

describe('其他格式永远收不到 Claude 专用的 system 消息', () => {
  const config = { apiKey: 'test-key', baseUrl: 'https://example.invalid/v1' };
  const providers = [
    ['openai-compatible', createOpenAICompatibleProvider({ ...config, provider: 'openai-compatible', model: 'gpt-test' })],
    ['openai-responses', createOpenAIResponsesProvider({ ...config, provider: 'openai-responses', model: 'gpt-test' })],
    ['gemini', createGeminiProvider({ ...config, provider: 'gemini', model: 'gemini-test' })],
    ['deepseek', createDeepSeekProvider({ ...config, provider: 'deepseek', model: 'deepseek-test' })],
  ] as const;

  for (const [name, provider] of providers) {
    it(`${name} 编码入口直接报错，不把它当成 user 消息发出`, async () => {
      await expect(provider.dryRun(TOOL_LOOP, { inputFormat: 'unified' })).rejects.toThrow(/Claude 专用的消息中段 system 消息/);
      expect(() => encodeRequestToFormat(TOOL_LOOP, { format: name })).toThrow(/Claude 专用的消息中段 system 消息/);
    });
  }

  it('Claude provider 正常编码', async () => {
    const provider = createClaudeProvider({ ...config, provider: 'claude', model: 'claude-fable-5-1' });
    const result = await provider.dryRun(TOOL_LOOP, { inputFormat: 'unified' });
    expect((result.body as any).messages.filter((message: any) => message.role === 'system')).toHaveLength(3);
  });

  it('不带 claudeSystemMessage 的请求在所有格式上与原来一致（字段缺省）', () => {
    const plain: LLMRequest = { contents: [{ role: 'user', parts: [{ text: 'hello' }] }] };
    for (const [name] of providers) {
      expect(() => encodeRequestToFormat(plain, { format: name })).not.toThrow();
    }
  });
});
