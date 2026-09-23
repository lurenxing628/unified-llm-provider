import { describe, expect, it, vi } from 'vitest';

import { OpenAICompatibleFormat, createOpenAICompatibleProvider, decodeRequestFromFormat } from '../src/index.js';
import type { Content, LLMRequest, LLMStreamChunk, Part } from '../src/index.js';
import { chatChunk, decodeJson, decodeStream, plain, readBaseline, readFixture, sse } from './openai-compatible-helpers.js';

/**
 * A4：OpenRouter 推理字段。
 * - https://openrouter.ai/docs/use-cases/reasoning-tokens：`reasoning` 文本字段、`reasoning_details`
 *   数组（reasoning.text / reasoning.summary / reasoning.encrypted），回放时“Pass back unmodified”。
 * - 流式合并规则来自 OpenRouter 官方 AI SDK provider（github.com/OpenRouterTeam/ai-sdk-provider，
 *   src/chat/index.ts，commit 1b22b05）：相邻同类型 text/summary 合并，encrypted 原样追加。
 */

const GEMINI_ENCRYPTED = 'CiQBjz1rX0Rj/ENCRYPTED/THOUGHT/SIGNATURE==';

function openRouterStream(): string {
  return sse(
    chatChunk({ role: 'assistant', content: '', reasoning: '**Plan**\n', reasoning_details: [{ type: 'reasoning.text', text: '**Plan**\n', format: 'google-gemini-v1', index: 0 }] }),
    chatChunk({ content: '', reasoning: 'Call the tool.', reasoning_details: [{ type: 'reasoning.text', text: 'Call the tool.', format: 'google-gemini-v1', index: 0 }] }),
    chatChunk({
      content: null,
      tool_calls: [{ index: 0, id: 'tool_get_weather_abc', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }],
      reasoning: null,
      reasoning_details: [{ type: 'reasoning.encrypted', data: GEMINI_ENCRYPTED, id: 'tool_get_weather_abc', format: 'google-gemini-v1', index: 0 }],
    }),
    chatChunk({}, { finish_reason: 'tool_calls', native_finish_reason: 'STOP' }),
    { ...chatChunk({}), choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } },
    '[DONE]',
  );
}

const EXPECTED_DETAILS = [
  { type: 'reasoning.text', text: '**Plan**\nCall the tool.', format: 'google-gemini-v1', index: 0 },
  { type: 'reasoning.encrypted', data: GEMINI_ENCRYPTED, id: 'tool_get_weather_abc', format: 'google-gemini-v1', index: 0 },
];

function envelopeOf(signature: string | undefined): any {
  expect(typeof signature).toBe('string');
  const raw = signature!.startsWith('openai-compatible:') ? signature!.slice('openai-compatible:'.length) : signature!;
  return JSON.parse(raw);
}

/**
 * 按扩展侧（limcode backend/capabilities/llmStreamEventProjection.ts 的 emitThoughtDeltas /
 * shouldCloseThoughtBlock，以及 llmCapabilityProviderAdapter.ts 的 completeLastThoughtPart）的语义
 * 把 unified 流块存成历史：每个思考块一个 part，part 上只保存最后一个便携签名字符串。
 */
function storeLikeExtension(chunks: LLMStreamChunk[]): Array<Record<string, unknown>> {
  const parts: Array<Record<string, unknown>> = [];
  let block: { text: string; signature?: string } | undefined;
  const closeBlock = () => {
    if (!block) return;
    const open = parts.findLast(part => part.thought === true && part.closed !== true);
    if (open) {
      open.closed = true;
      if (block.signature) open.thoughtSignature = block.signature;
    } else if (block.signature) {
      parts.push({ text: '', thought: true, thoughtSignature: block.signature, closed: true });
    }
    block = undefined;
  };
  for (const chunk of chunks) {
    if (chunk.error) throw new Error(chunk.error.message ?? chunk.error.kind);
    const chunkSignature = typeof chunk.thoughtSignature === 'string' ? chunk.thoughtSignature : undefined;
    if (chunkSignature) {
      block ??= { text: '' };
      block.signature = chunkSignature;
    }
    let thoughtText = false;
    for (const part of chunk.partsDelta ?? []) {
      if ((part as any).thought !== true) continue;
      block ??= { text: '' };
      const signature = (part as any).thoughtSignature;
      if (typeof signature === 'string' && signature) block.signature = signature;
      const text = (part as any).text ?? '';
      if (!text) continue;
      thoughtText = true;
      const open = parts.findLast(item => item.thought === true && item.closed !== true);
      if (open && parts[parts.length - 1] === open) open.text = `${open.text}${text}`;
      else parts.push({ text, thought: true });
    }
    const visible = (chunk.partsDelta ?? []).filter(part => 'text' in part && (part as any).thought !== true && (part as any).text);
    const calls = chunk.functionCalls ?? [];
    const signatureOnly = !!chunkSignature || (chunk.partsDelta ?? []).some(part => (part as any).thought === true && (part as any).thoughtSignature);
    if (block && (visible.length > 0 || calls.length > 0 || chunk.finishReason || (signatureOnly && !thoughtText))) closeBlock();
    for (const part of visible) parts.push({ text: (part as any).text });
    for (const call of calls) parts.push({ id: call.functionCall.callId, functionCall: { name: call.functionCall.name, args: call.functionCall.args } });
  }
  closeBlock();
  return parts.map(({ closed: _closed, ...rest }) => rest);
}

/** 扩展 unifiedMessageConversion.toUnifiedPart：便携签名还原为 thoughtSignature + thoughtSignatures。 */
function toUnifiedParts(stored: Array<Record<string, unknown>>): Part[] {
  return stored.map((part) => {
    const signature = typeof part.thoughtSignature === 'string' ? part.thoughtSignature : undefined;
    const colon = signature?.indexOf(':') ?? -1;
    const thoughtSignatures = signature && colon > 0 ? { [signature.slice(0, colon)]: signature.slice(colon + 1).trim() } : undefined;
    if ('functionCall' in part) {
      return { functionCall: { ...(part.functionCall as any), callId: part.id } } as Part;
    }
    return {
      text: part.text as string,
      ...(part.thought !== undefined ? { thought: part.thought as boolean } : {}),
      ...(signature ? { thoughtSignature: signature } : {}),
      ...(thoughtSignatures ? { thoughtSignatures } : {}),
    } as Part;
  });
}

async function collect(provider: ReturnType<typeof createOpenAICompatibleProvider>, request: LLMRequest): Promise<LLMStreamChunk[]> {
  const chunks: LLMStreamChunk[] = [];
  for await (const chunk of provider.chatStream(request, { inputFormat: 'unified', outputFormat: 'unified' })) chunks.push(chunk);
  return chunks;
}

describe('OpenRouter reasoning / reasoning_details 解码（A4）', () => {
  it('流式：思考文本取自 delta.reasoning，reasoning_details 按官方规则累积并在流结束时作为签名信封发出', async () => {
    const chunks = await decodeStream(openRouterStream());

    const thoughtText = chunks.flatMap(chunk => chunk.partsDelta ?? [])
      .filter(part => (part as any).thought === true)
      .map(part => (part as any).text ?? '')
      .join('');
    expect(thoughtText).toBe('**Plan**\nCall the tool.');

    const finalChunk = chunks[chunks.length - 1];
    expect(finalChunk.functionCalls).toBeUndefined();
    expect(finalChunk.partsDelta).toHaveLength(1);
    expect((finalChunk.partsDelta![0] as any).thought).toBe(true);
    expect((finalChunk.partsDelta![0] as any).text).toBe('');
    expect(finalChunk.thoughtSignature).toBe((finalChunk.partsDelta![0] as any).thoughtSignature);
    expect(envelopeOf(finalChunk.thoughtSignature)).toEqual({ reasoning_details: EXPECTED_DETAILS, reasoning_field: 'reasoning' });

    // 签名只在流结束时发一次；工具调用照常在 finish_reason 时发出
    expect(chunks.filter(chunk => chunk.thoughtSignature)).toHaveLength(1);
    const callChunk = chunks.find(chunk => chunk.functionCalls?.length);
    expect(plain(callChunk!.functionCalls)).toEqual([{ functionCall: { name: 'get_weather', args: { city: 'Paris' }, callId: 'tool_get_weather_abc' } }]);
  });

  it('流式合并规则与官方 SDK 一致：相邻 text 合并并保留签名，encrypted 独立，其后的 text/summary 另起一项', async () => {
    // 对应 ai-sdk-provider 的测试 “should accumulate multiple late reasoning_details after text”
    const chunks = await decodeStream(sse(
      chatChunk({ reasoning_details: [{ type: 'reasoning.text', text: 'Let me think', index: 0, format: 'anthropic-claude-v1' }] }),
      chatChunk({ reasoning_details: [{ type: 'reasoning.text', text: ' about this.', index: 0, format: 'anthropic-claude-v1' }] }),
      chatChunk({ reasoning_details: [{ type: 'reasoning.text', text: '', index: 0, format: 'anthropic-claude-v1', signature: 'sig-1' }] }),
      chatChunk({ content: 'Result' }),
      chatChunk({ reasoning_details: [{ type: 'reasoning.encrypted', data: 'encrypted-blob-data' }] }),
      chatChunk({ reasoning_details: [{ type: 'reasoning.text', text: '', index: 0, format: 'anthropic-claude-v1', signature: 'sig123' }] }),
      chatChunk({ reasoning_details: [{ type: 'reasoning.summary', summary: 'Late ' }] }),
      chatChunk({ reasoning_details: [{ type: 'reasoning.summary', summary: 'summary' }] }),
      chatChunk({}, { finish_reason: 'stop' }),
      '[DONE]',
    ));
    expect(envelopeOf(chunks[chunks.length - 1].thoughtSignature)).toEqual({
      reasoning_details: [
        { type: 'reasoning.text', text: 'Let me think about this.', index: 0, format: 'anthropic-claude-v1', signature: 'sig-1' },
        { type: 'reasoning.encrypted', data: 'encrypted-blob-data' },
        { type: 'reasoning.text', text: '', index: 0, format: 'anthropic-claude-v1', signature: 'sig123' },
        { type: 'reasoning.summary', summary: 'Late summary' },
      ],
    });
  });

  it('带不同数字 index 的相邻同类型块不合并（index 表示不同的推理块）', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ reasoning_details: [{ type: 'reasoning.text', text: 'first', index: 0, signature: 's0' }] }),
      chatChunk({ reasoning_details: [{ type: 'reasoning.text', text: 'second', index: 1, signature: 's1' }] }),
      '[DONE]',
    ));
    expect(envelopeOf(chunks[chunks.length - 1].thoughtSignature).reasoning_details).toEqual([
      { type: 'reasoning.text', text: 'first', index: 0, signature: 's0' },
      { type: 'reasoning.text', text: 'second', index: 1, signature: 's1' },
    ]);
  });

  it('同一块同时有 reasoning_content 与 reasoning 时只取 reasoning_content，不重复', async () => {
    const chunks = await decodeStream(sse(
      chatChunk({ reasoning_content: 'think', reasoning: 'think' }),
      chatChunk({ content: 'ok' }, { finish_reason: 'stop' }),
      '[DONE]',
    ));
    const thoughts = chunks.flatMap(chunk => chunk.partsDelta ?? []).filter(part => (part as any).thought === true);
    expect(plain(thoughts)).toEqual([{ text: 'think', thought: true }]);
    // 没有 reasoning_details、文本也不是来自 reasoning 字段：不追加任何块
    expect(chunks).toHaveLength(2);
  });

  it('非流式：message.reasoning 为思考文本，reasoning_details 原样进入签名信封', async () => {
    const decoded = await decodeJson({
      id: 'gen-1',
      object: 'chat.completion',
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: null,
          reasoning: '**Plan**\nCall the tool.',
          reasoning_details: EXPECTED_DETAILS,
          tool_calls: [{ id: 'tool_get_weather_abc', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }],
        },
        finish_reason: 'tool_calls',
      }],
    });
    const [thought, call] = decoded.content.parts as any[];
    expect(thought.text).toBe('**Plan**\nCall the tool.');
    expect(thought.thought).toBe(true);
    expect(envelopeOf(thought.thoughtSignature)).toEqual({ reasoning_details: EXPECTED_DETAILS, reasoning_field: 'reasoning' });
    expect(call.functionCall).toEqual({ name: 'get_weather', args: { city: 'Paris' }, callId: 'tool_get_weather_abc' });
  });

  it('非流式：reasoning_content + reasoning_details + reasoning_signature 同时存在时全部保留', async () => {
    const decoded = await decodeJson({
      choices: [{
        index: 0,
        message: { role: 'assistant', content: 'ok', reasoning_content: 'rc', reasoning: 'rc', reasoning_signature: 'plain-sig', reasoning_details: [{ type: 'reasoning.encrypted', data: 'x' }] },
        finish_reason: 'stop',
      }],
    });
    const thought = decoded.content.parts[0] as any;
    expect(thought.text).toBe('rc');
    expect(envelopeOf(thought.thoughtSignature)).toEqual({ reasoning_details: [{ type: 'reasoning.encrypted', data: 'x' }], reasoning_signature: 'plain-sig' });
  });

  it('原行为不变：只有 reasoning_content / reasoning_signature 的响应解码结果与修复前一致', async () => {
    const body = {
      choices: [{ index: 0, message: { role: 'assistant', content: 'answer', reasoning_content: 'thinking', reasoning_signature: 'sig-1' }, finish_reason: 'stop' }],
    };
    expect(plain(new OpenAICompatibleFormat('m').decodeResponse(body))).toEqual({
      content: { role: 'model', parts: [{ text: 'thinking', thought: true, thoughtSignature: 'sig-1' }, { text: 'answer' }] },
      finishReason: 'stop',
    });
    const emptySignature = { choices: [{ index: 0, message: { role: 'assistant', content: 'a', reasoning_signature: '' }, finish_reason: 'stop' }] };
    expect(plain(new OpenAICompatibleFormat('m').decodeResponse(emptySignature).content.parts)).toEqual([
      { text: '', thought: true, thoughtSignature: '' },
      { text: 'a' },
    ]);
    // reasoning_details 为空数组时不产生任何思考 part
    const emptyDetails = { choices: [{ index: 0, message: { role: 'assistant', content: 'a', reasoning_details: [] }, finish_reason: 'stop' }] };
    expect(plain(new OpenAICompatibleFormat('m').decodeResponse(emptyDetails).content.parts)).toEqual([{ text: 'a' }]);
  });

  it('原行为不变：实测 gemini 流（reasoning_content）解码与修复前逐块一致', async () => {
    const chunks = await decodeStream(readFixture('tools-v-gemini-3.5-flash.sse'));
    expect(plain(chunks)).toEqual(readBaseline('tools-v-gemini-3.5-flash.sse'));
  });
});

describe('reasoning_details 回编（A4）', () => {
  const model = (parts: Part[]): Content => ({ role: 'model', parts });
  const encode = (contents: Content[], kind: 'openai-compatible' | 'deepseek' = 'openai-compatible') => {
    const request = decodeRequestFromFormat({ contents }, { format: 'unified' });
    return (new OpenAICompatibleFormat('m', kind).encodeRequest(request, false) as any).messages as any[];
  };
  const envelope = JSON.stringify({ reasoning_details: EXPECTED_DETAILS, reasoning_field: 'reasoning' });

  it('信封里的 reasoning_details 原样放回 assistant 消息，思考文本走 reasoning 字段', () => {
    const [assistant] = encode([
      model([
        { text: '**Plan**\nCall the tool.', thought: true },
        { text: '', thought: true, thoughtSignatures: { 'openai-compatible': envelope } },
        { functionCall: { name: 'get_weather', args: { city: 'Paris' }, callId: 'tool_get_weather_abc' } },
      ]),
    ]);
    expect(assistant).toEqual({
      role: 'assistant',
      content: null,
      tool_calls: [{ id: 'tool_get_weather_abc', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }],
      reasoning: '**Plan**\nCall the tool.',
      reasoning_details: EXPECTED_DETAILS,
    });
  });

  it('便携字符串签名（openai-compatible:<信封>）同样能还原', () => {
    const [assistant] = encode([
      model([{ text: 'x', thought: true, thoughtSignature: `openai-compatible:${envelope}` } as any, { text: 'done' }]),
    ]);
    expect(assistant.reasoning_details).toEqual(EXPECTED_DETAILS);
    expect(assistant.reasoning).toBe('x');
    expect(assistant.reasoning_content).toBeUndefined();
    expect(assistant.reasoning_signature).toBeUndefined();
  });

  it('文本来自 reasoning_content 时仍用 reasoning_content，并带回 reasoning_details / reasoning_signature', () => {
    const [assistant] = encode([
      model([
        { text: 'rc', thought: true, thoughtSignatures: { 'openai-compatible': JSON.stringify({ reasoning_details: [{ type: 'reasoning.encrypted', data: 'x' }], reasoning_signature: 'plain-sig' }) } },
        { text: 'ok' },
      ]),
    ]);
    expect(assistant).toEqual({
      role: 'assistant',
      content: 'ok',
      reasoning_content: 'rc',
      reasoning_signature: 'plain-sig',
      reasoning_details: [{ type: 'reasoning.encrypted', data: 'x' }],
    });
  });

  it('只有 reasoning 字段、没有 reasoning_details（如 vLLM/Groq）时不回传思考文本，与修复前的线上请求一致', () => {
    const [assistant] = encode([
      model([
        { text: 'thinking', thought: true, thoughtSignatures: { 'openai-compatible': JSON.stringify({ reasoning_field: 'reasoning' }) } },
        { text: 'ok' },
      ]),
    ]);
    expect(assistant).toEqual({ role: 'assistant', content: 'ok' });
  });

  it('原行为不变：纯字符串签名仍按 reasoning_signature 发送（包括看起来像 JSON 但不是信封的值）', () => {
    const [plainAssistant] = encode([model([{ text: 't', thought: true, thoughtSignatures: { 'openai-compatible': 'sig-1' } }, { text: 'a' }])]);
    expect(plainAssistant).toEqual({ role: 'assistant', content: 'a', reasoning_content: 't', reasoning_signature: 'sig-1' });

    const jsonLike = '{"foo":1}';
    const [jsonAssistant] = encode([model([{ text: 't', thought: true, thoughtSignatures: { 'openai-compatible': jsonLike } }, { text: 'a' }])], 'deepseek');
    expect(jsonAssistant).toEqual({ role: 'assistant', content: 'a', reasoning_content: 't', reasoning_signature: jsonLike });
  });
});

describe('reasoning_details 经扩展存储往返（A4，dry-run 验证）', () => {
  it('流式 → 按扩展语义存储（每个 part 一个便携签名）→ 下一轮请求原样带回 reasoning_details', async () => {
    const fetch = vi.fn(async () => new Response(openRouterStream(), { status: 200, headers: { 'content-type': 'text/event-stream' } }));
    const provider = createOpenAICompatibleProvider({
      provider: 'openai-compatible',
      model: 'google/gemini-3-pro-preview',
      apiKey: 'sk-test',
      baseUrl: 'https://openrouter.test/api/v1',
      fetch: fetch as any,
    });
    const userTurn: Content = { role: 'user', parts: [{ text: 'weather in Paris?' }] };
    const chunks = await collect(provider, { contents: [userTurn] });

    const stored = storeLikeExtension(chunks);
    const signatures = stored.filter(part => typeof part.thoughtSignature === 'string');
    expect(signatures).toHaveLength(1);
    expect(String(signatures[0].thoughtSignature).startsWith('openai-compatible:{')).toBe(true);

    const dry = await provider.dryRun({
      contents: [
        userTurn,
        { role: 'model', parts: toUnifiedParts(stored) },
        { role: 'user', parts: [{ functionResponse: { name: 'get_weather', response: { temp: '18C' }, callId: 'tool_get_weather_abc' } }] },
      ],
    }, { stream: true });
    const assistant = (dry.body as any).messages[1];
    expect(assistant.reasoning_details).toEqual(EXPECTED_DETAILS);
    expect(assistant.reasoning).toBe('**Plan**\nCall the tool.');
    expect(assistant.tool_calls).toEqual([{ id: 'tool_get_weather_abc', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }]);
    expect(assistant.reasoning_content).toBeUndefined();
    expect(assistant.reasoning_signature).toBeUndefined();
    expect((dry.body as any).messages[2]).toEqual({ role: 'tool', tool_call_id: 'tool_get_weather_abc', content: '{"temp":"18C"}' });
  });

  it('非流式 → 便携签名 → 下一轮请求原样带回 reasoning_details', async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify({
      choices: [{ index: 0, message: { role: 'assistant', content: 'Sunny.', reasoning: 'r', reasoning_details: EXPECTED_DETAILS }, finish_reason: 'stop' }],
    }), { status: 200, headers: { 'content-type': 'application/json' } }));
    const provider = createOpenAICompatibleProvider({
      provider: 'openai-compatible',
      model: 'google/gemini-3-pro-preview',
      apiKey: 'sk-test',
      baseUrl: 'https://openrouter.test/api/v1',
      fetch: fetch as any,
    });
    const response = await provider.chat({ contents: [{ role: 'user', parts: [{ text: 'q' }] }] }, { inputFormat: 'unified', outputFormat: 'unified' }) as any;
    const thought = response.content.parts[0];
    expect(thought.thoughtSignature.startsWith('openai-compatible:{')).toBe(true);

    const dry = await provider.dryRun({
      contents: [
        { role: 'user', parts: [{ text: 'q' }] },
        { role: 'model', parts: toUnifiedParts(response.content.parts) },
        { role: 'user', parts: [{ text: 'next' }] },
      ],
    }, { stream: false });
    expect((dry.body as any).messages[1]).toEqual({ role: 'assistant', content: 'Sunny.', reasoning: 'r', reasoning_details: EXPECTED_DETAILS });
  });
});
