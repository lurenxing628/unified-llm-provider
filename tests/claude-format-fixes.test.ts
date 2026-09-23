/**
 * Claude 格式修复（组 B：B1 思考块顺序、B2 redacted_thinking、B3 工具调用 id 规范化）。
 *
 * “原行为不变”回归：BASELINE_* 常量是修复前（limcode/provider-fixes 16b50ab）的编码 / 解码结果，
 * 用 JSON.stringify 逐字节比较（含键顺序）。
 */
import { describe, expect, it } from 'vitest';

import {
  CLAUDE_REDACTED_THINKING_SIGNATURE_PREFIX,
  ClaudeFormat,
  createClaudeProvider,
  decodeResponseFromFormat,
} from '../src/index.js';
import type { LLMRequest } from '../src/index.js';

const TOOLS = [{ functionDeclarations: [
  { name: 'read_file', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' }, limit: { type: 'integer' } }, required: ['path'] } },
  { name: 'list_conversations', description: 'List', parameters: { type: 'object', properties: {} } },
] }] as LLMRequest['tools'];

/** 官方 Claude 正常历史：思考在前、文本、并行工具调用、工具结果、再一轮思考 + 文本、带图片的用户消息。 */
const STANDARD_REQUEST: LLMRequest = {
  systemInstruction: { parts: [{ text: 'You are helpful.' }] },
  tools: TOOLS,
  contents: [
    { role: 'user', parts: [{ text: 'read a.txt' }] },
    { role: 'model', parts: [
      { text: 'I should read it.', thought: true, thoughtSignatures: { claude: 'EqQBCgIYAhIMsig1' } },
      { text: 'Reading now.' },
      { functionCall: { name: 'read_file', args: { path: 'a.txt' }, callId: 'toolu_01A' } },
      { functionCall: { name: 'list_conversations', args: {}, callId: 'toolu_01B' } },
    ] },
    { role: 'user', parts: [
      { functionResponse: { name: 'read_file', response: { content: 'hello' }, callId: 'toolu_01A' } },
      { functionResponse: { name: 'list_conversations', response: { items: [] }, callId: 'toolu_01B' } },
    ] },
    { role: 'model', parts: [
      { text: '', thought: true, thoughtSignatures: { claude: 'EqQBCgIYAhIMsig2' } },
      { text: 'The file says hello.' },
    ] },
    { role: 'user', parts: [{ text: 'thanks' }, { inlineData: { mimeType: 'image/png', data: 'iVBORw0KGgo=' } }] },
  ],
  generationConfig: { maxOutputTokens: 2048, thinkingConfig: { thinkingLevel: 'high' } },
};

const BASELINE_STANDARD_STREAM_BODY = {
  "model": "claude-sonnet-4-6",
  "system": "You are helpful.",
  "messages": [
    {
      "role": "user",
      "content": "read a.txt"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "I should read it.",
          "signature": "EqQBCgIYAhIMsig1"
        },
        {
          "type": "text",
          "text": "Reading now."
        },
        {
          "type": "tool_use",
          "id": "toolu_01A",
          "name": "read_file",
          "input": {
            "path": "a.txt"
          }
        },
        {
          "type": "tool_use",
          "id": "toolu_01B",
          "name": "list_conversations",
          "input": {}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01A",
          "content": "{\"content\":\"hello\"}"
        },
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01B",
          "content": "{\"items\":[]}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "",
          "signature": "EqQBCgIYAhIMsig2"
        },
        {
          "type": "text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "thanks"
        },
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgo="
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "limit": {
            "type": "integer"
          }
        },
        "required": [
          "path"
        ]
      }
    },
    {
      "name": "list_conversations",
      "description": "List",
      "input_schema": {
        "type": "object",
        "properties": {}
      }
    }
  ],
  "max_tokens": 2048,
  "thinking": {
    "type": "adaptive"
  },
  "output_config": {
    "effort": "high"
  },
  "stream": true
};

const BASELINE_STANDARD_CACHED_BODY = {
  "model": "claude-sonnet-4-6",
  "system": [
    {
      "type": "text",
      "text": "You are helpful.",
      "cache_control": {
        "type": "ephemeral",
        "ttl": "1h"
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "read a.txt"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "I should read it.",
          "signature": "EqQBCgIYAhIMsig1"
        },
        {
          "type": "text",
          "text": "Reading now."
        },
        {
          "type": "tool_use",
          "id": "toolu_01A",
          "name": "read_file",
          "input": {
            "path": "a.txt"
          }
        },
        {
          "type": "tool_use",
          "id": "toolu_01B",
          "name": "list_conversations",
          "input": {}
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01A",
          "content": "{\"content\":\"hello\"}"
        },
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01B",
          "content": "{\"items\":[]}"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "",
          "signature": "EqQBCgIYAhIMsig2"
        },
        {
          "type": "text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "thanks"
        },
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "iVBORw0KGgo="
          },
          "cache_control": {
            "type": "ephemeral",
            "ttl": "1h"
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string"
          },
          "limit": {
            "type": "integer"
          }
        },
        "required": [
          "path"
        ]
      }
    },
    {
      "name": "list_conversations",
      "description": "List",
      "input_schema": {
        "type": "object",
        "properties": {}
      },
      "cache_control": {
        "type": "ephemeral",
        "ttl": "1h"
      }
    }
  ],
  "max_tokens": 2048,
  "thinking": {
    "type": "adaptive"
  },
  "output_config": {
    "effort": "high"
  },
  "cache_control": {
    "type": "ephemeral",
    "ttl": "1h"
  }
};

const CLAUDE_RAW_RESPONSE = {
  id: 'msg_1', type: 'message', role: 'assistant', stop_reason: 'tool_use',
  content: [
    { type: 'thinking', thinking: 'plan', signature: 'EqQBsigA' },
    { type: 'text', text: 'Let me look.' },
    { type: 'thinking', thinking: '', signature: 'EqQBsigB' },
    { type: 'tool_use', id: 'toolu_01X', name: 'read_file', input: { path: 'a' } },
  ],
  usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 2 },
};

const BASELINE_DECODED_RESPONSE = {
  "content": {
    "role": "model",
    "parts": [
      {
        "text": "plan",
        "thought": true,
        "thoughtSignatures": {
          "claude": "EqQBsigA"
        }
      },
      {
        "text": "Let me look."
      },
      {
        "text": "",
        "thought": true,
        "thoughtSignatures": {
          "claude": "EqQBsigB"
        }
      },
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "toolu_01X"
        }
      }
    ]
  },
  "finishReason": "TOOL_CALLS",
  "usageMetadata": {
    "promptTokenCount": 12,
    "cachedContentTokenCount": 2,
    "candidatesTokenCount": 5,
    "totalTokenCount": 17
  }
};

const CLAUDE_STREAM_EVENTS = [
  { type: 'message_start', message: { usage: { input_tokens: 10 } } },
  { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '', signature: '' } },
  { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'plan' } },
  { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'EqQBsigA' } },
  { type: 'content_block_stop', index: 0 },
  { type: 'content_block_start', index: 1, content_block: { type: 'text', text: '' } },
  { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'Let me look.' } },
  { type: 'content_block_stop', index: 1 },
  { type: 'content_block_start', index: 2, content_block: { type: 'tool_use', id: 'toolu_01X', name: 'read_file', input: {} } },
  { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '{"path":' } },
  { type: 'content_block_delta', index: 2, delta: { type: 'input_json_delta', partial_json: '"a"}' } },
  { type: 'content_block_stop', index: 2 },
  { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 5 } },
  { type: 'message_stop' },
];

const BASELINE_STREAM_CHUNKS = [
  {},
  {},
  {
    "partsDelta": [
      {
        "text": "plan",
        "thought": true
      }
    ]
  },
  {
    "partsDelta": [
      {
        "thought": true,
        "thoughtSignatures": {
          "claude": "EqQBsigA"
        }
      }
    ],
    "thoughtSignatures": {
      "claude": "EqQBsigA"
    }
  },
  {},
  {},
  {
    "textDelta": "Let me look."
  },
  {},
  {},
  {},
  {},
  {
    "functionCalls": [
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "toolu_01X"
        }
      }
    ]
  },
  {
    "finishReason": "TOOL_CALLS",
    "usageMetadata": {
      "promptTokenCount": 10,
      "candidatesTokenCount": 5,
      "totalTokenCount": 15
    }
  },
  {}
];

function encode(request: LLMRequest, format = new ClaudeFormat('claude-sonnet-4-6')): any {
  return format.encodeRequest(request, false);
}

describe('ClaudeFormat 原行为不变（修复前基线逐字节比较）', () => {
  it('官方正常历史的流式请求体与修复前逐字节一致', () => {
    const body = new ClaudeFormat('claude-sonnet-4-6').encodeRequest(STANDARD_REQUEST, true);
    expect(JSON.stringify(body)).toBe(JSON.stringify(BASELINE_STANDARD_STREAM_BODY));
  });

  it('开启 1h 手动断点 + 自动缓存时请求体与修复前逐字节一致', () => {
    const body = new ClaudeFormat('claude-sonnet-4-6', { enabled: true, ttl: '1h' }, true).encodeRequest(STANDARD_REQUEST, false);
    expect(JSON.stringify(body)).toBe(JSON.stringify(BASELINE_STANDARD_CACHED_BODY));
  });

  it('没有 redacted_thinking 的非流式响应解码结果与修复前一致', () => {
    const decoded = new ClaudeFormat('claude-sonnet-4-6').decodeResponse(structuredClone(CLAUDE_RAW_RESPONSE));
    expect(JSON.stringify(decoded)).toBe(JSON.stringify(BASELINE_DECODED_RESPONSE));
  });

  it('没有 redacted_thinking 的流式事件解码结果与修复前一致', () => {
    const format = new ClaudeFormat('claude-sonnet-4-6');
    const state = format.createStreamState();
    const chunks = CLAUDE_STREAM_EVENTS.map(event => format.decodeStreamChunk(structuredClone(event), state));
    expect(JSON.stringify(chunks)).toBe(JSON.stringify(BASELINE_STREAM_CHUNKS));
  });
});

describe('B1 Claude 思考块保持原顺序', () => {
  // 依据：https://platform.claude.com/docs/en/build-with-claude/thinking#preserving-thinking-blocks
  // 与 Messages API ThinkingBlockParam：“Thinking blocks must be passed back unmodified and in their original order”。
  it('交错思考 [thinking, text, thinking, tool_use] 按存储顺序回放，不再把思考块挪到最前', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { text: 'first', thought: true, thoughtSignatures: { claude: 'sigA' } },
          { text: 'Let me check.' },
          { text: '', thought: true, thoughtSignatures: { claude: 'sigB' } },
          { functionCall: { name: 'read_file', args: { path: 'a' }, callId: 'toolu_1' } },
        ] },
        { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { ok: 1 }, callId: 'toolu_1' } }] },
      ],
    });

    expect(body.messages[1].content).toEqual([
      { type: 'thinking', thinking: 'first', signature: 'sigA' },
      { type: 'text', text: 'Let me check.' },
      { type: 'thinking', thinking: '', signature: 'sigB' },
      { type: 'tool_use', id: 'toolu_1', name: 'read_file', input: { path: 'a' } },
    ]);
  });

  it('工具调用之间的思考块（tool_use, thinking, tool_use）保持在原位置', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { text: 'a', thought: true, thoughtSignatures: { claude: 'sigA' } },
          { functionCall: { name: 'read_file', args: { path: 'a' }, callId: 'toolu_1' } },
          { text: 'b', thought: true, thoughtSignatures: { claude: 'sigB' } },
          { functionCall: { name: 'read_file', args: { path: 'b' }, callId: 'toolu_2' } },
          { text: 'done' },
        ] },
      ],
    });

    expect(body.messages[1].content.map((block: any) => block.type)).toEqual([
      'thinking', 'tool_use', 'thinking', 'tool_use', 'text',
    ]);
    expect(body.messages[1].content[2]).toEqual({ type: 'thinking', thinking: 'b', signature: 'sigB' });
  });

  it('空文本且无签名的 thought、空文本 part 仍被跳过，其余块顺序不变', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { text: 'answer' },
          { text: '', thought: true },
          { text: '' },
          { text: 'unsigned reasoning', thought: true },
        ] },
      ],
    });

    expect(body.messages[1].content).toEqual([
      { type: 'text', text: 'answer' },
      { type: 'thinking', thinking: 'unsigned reasoning' },
    ]);
  });
});

describe('B2 Claude redacted_thinking 保留并原样回放', () => {
  // 依据：https://platform.claude.com/docs/en/build-with-claude/thinking#redacted-thinking-blocks
  // “pass redacted_thinking blocks back to the API unchanged”；过滤时只认 thinking 会破坏多轮协议。
  const REDACTED_DATA = 'EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpPkNRj2YfWXGmKDxH4mPnZ5sQ7vB5URj';

  it('前缀不含冒号且以非 base64 字符开头，不会与真实签名或便携签名 provider 前缀冲突', () => {
    expect(CLAUDE_REDACTED_THINKING_SIGNATURE_PREFIX).toBe('#redacted_thinking#');
    expect(CLAUDE_REDACTED_THINKING_SIGNATURE_PREFIX).not.toContain(':');
    expect(/^[A-Za-z0-9+/=]/.test(CLAUDE_REDACTED_THINKING_SIGNATURE_PREFIX)).toBe(false);
  });

  it('非流式解码：redacted_thinking 按原位置保存为 thought part，数据放在 claude 签名命名空间', () => {
    const decoded = new ClaudeFormat('claude-sonnet-4-6').decodeResponse({
      content: [
        { type: 'thinking', thinking: 'plan', signature: 'EqQBsigA' },
        { type: 'redacted_thinking', data: REDACTED_DATA },
        { type: 'text', text: 'ok' },
      ],
      stop_reason: 'end_turn',
    });

    expect(decoded.content.parts).toEqual([
      { text: 'plan', thought: true, thoughtSignatures: { claude: 'EqQBsigA' } },
      { text: '', thought: true, thoughtSignatures: { claude: `#redacted_thinking#${REDACTED_DATA}` } },
      { text: 'ok' },
    ]);
  });

  it('流式解码：content_block_start 的 redacted_thinking 立即产出 thought part 与 chunk 签名', () => {
    const format = new ClaudeFormat('claude-sonnet-4-6');
    const state = format.createStreamState();
    const chunk = format.decodeStreamChunk({
      type: 'content_block_start',
      index: 1,
      content_block: { type: 'redacted_thinking', data: REDACTED_DATA },
    }, state);
    const stop = format.decodeStreamChunk({ type: 'content_block_stop', index: 1 }, state);

    expect(chunk).toEqual({
      partsDelta: [{ text: '', thought: true, thoughtSignatures: { claude: `#redacted_thinking#${REDACTED_DATA}` } }],
      thoughtSignatures: { claude: `#redacted_thinking#${REDACTED_DATA}` },
    });
    expect(stop).toEqual({});
  });

  it('编码：带前缀的 thought part 原样还原为 {type:"redacted_thinking", data}，且保持在原位置', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { text: 'plan', thought: true, thoughtSignatures: { claude: 'EqQBsigA' } },
          { text: '', thought: true, thoughtSignatures: { claude: `#redacted_thinking#${REDACTED_DATA}` } },
          { functionCall: { name: 'read_file', args: { path: 'a' }, callId: 'toolu_1' } },
        ] },
        { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { ok: 1 }, callId: 'toolu_1' } }] },
      ],
    });

    expect(body.messages[1].content).toEqual([
      { type: 'thinking', thinking: 'plan', signature: 'EqQBsigA' },
      { type: 'redacted_thinking', data: REDACTED_DATA },
      { type: 'tool_use', id: 'toolu_1', name: 'read_file', input: { path: 'a' } },
    ]);
  });

  it('真实签名（base64）仍编码为 thinking 块，不会被误认成 redacted_thinking', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [{ text: '', thought: true, thoughtSignatures: { claude: 'redacted_thinking+EqQB/sig==' } }] },
      ],
    });
    expect(body.messages[1].content).toEqual([
      { type: 'thinking', thinking: '', signature: 'redacted_thinking+EqQB/sig==' },
    ]);
  });

  it('经便携字符串签名（claude:<value>）往返后仍能还原：解码 → 字符串签名 → 按首个冒号拆分 → 回编', async () => {
    const response = decodeResponseFromFormat({
      content: [
        { type: 'redacted_thinking', data: REDACTED_DATA },
        { type: 'text', text: 'ok' },
      ],
      stop_reason: 'end_turn',
    }, { format: 'claude', model: 'claude-sonnet-4-6' });
    const redactedPart = response.content.parts[0] as any;
    expect(redactedPart.thoughtSignature).toBe(`claude:#redacted_thinking#${REDACTED_DATA}`);

    // 模拟 LimCode 存储：每个 part 只保留一个便携签名字符串，回编时按首个冒号拆回 { provider: value }。
    const portable: string = redactedPart.thoughtSignature;
    const colon = portable.indexOf(':');
    const restored = { [portable.slice(0, colon)]: portable.slice(colon + 1) };
    expect(restored).toEqual({ claude: `#redacted_thinking#${REDACTED_DATA}` });

    const provider = createClaudeProvider({
      provider: 'claude',
      model: 'claude-sonnet-4-6',
      apiKey: 'sk-test',
      baseUrl: 'https://api.anthropic.test/v1',
    });
    const dry = await provider.dryRun({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { text: '', thought: true, thoughtSignature: portable, thoughtSignatures: restored },
          { text: 'ok' },
        ] },
        { role: 'user', parts: [{ text: 'next' }] },
      ],
    } satisfies LLMRequest, { stream: false });

    expect((dry.body as any).messages[1].content).toEqual([
      { type: 'redacted_thinking', data: REDACTED_DATA },
      { type: 'text', text: 'ok' },
    ]);
  });

  it('data 含非 base64 字符（冒号等）时也能原样往返', () => {
    const odd = 'a:b#c/d+e=';
    const format = new ClaudeFormat('claude-sonnet-4-6');
    const decoded = format.decodeResponse({ content: [{ type: 'redacted_thinking', data: odd }], stop_reason: 'end_turn' });
    const body = encode({ contents: [{ role: 'user', parts: [{ text: 'go' }] }, decoded.content] }, format);
    expect(body.messages[1].content).toEqual([{ type: 'redacted_thinking', data: odd }]);
  });

  it('空 data 的 redacted_thinking 不产生 part（没有可回放的内容）', () => {
    const decoded = new ClaudeFormat('claude-sonnet-4-6').decodeResponse({
      content: [{ type: 'redacted_thinking', data: '' }, { type: 'text', text: 'ok' }],
      stop_reason: 'end_turn',
    });
    expect(decoded.content.parts).toEqual([{ text: 'ok' }]);
  });
});

describe('B3 Claude 工具调用 id 不合规时确定性改写', () => {
  // 依据：Messages API ToolUseBlockParam.id / ToolResultBlockParam.tool_use_id 的 pattern `^[a-zA-Z0-9_-]+$`
  // （https://platform.claude.com/docs/en/api/messages/create）。
  const CLAUDE_ID = /^[a-zA-Z0-9_-]+$/;

  function kimiHistory(): LLMRequest {
    return {
      contents: [
        { role: 'user', parts: [{ text: 'list' }] },
        { role: 'model', parts: [
          { functionCall: { name: 'list_conversations', args: {}, callId: 'functions.list_conversations:0' } },
          { functionCall: { name: 'list_conversations', args: { limit: 3 }, callId: 'functions.list_conversations:1' } },
        ] },
        { role: 'user', parts: [
          { functionResponse: { name: 'list_conversations', response: { ok: 1 }, callId: 'functions.list_conversations:1' } },
          { functionResponse: { name: 'list_conversations', response: { ok: 0 }, callId: 'functions.list_conversations:0' } },
        ] },
      ],
    };
  }

  it('Kimi 风格 id 改写为合规 id，tool_use.id 与 tool_result.tool_use_id 一致对应', () => {
    const body = encode(kimiHistory());
    const [first, second] = body.messages[1].content;
    const [resultForSecond, resultForFirst] = body.messages[2].content;

    for (const id of [first.id, second.id, resultForFirst.tool_use_id, resultForSecond.tool_use_id]) {
      expect(id).toMatch(CLAUDE_ID);
    }
    expect(first.id).toMatch(/^functions_list_conversations_0_[0-9a-f]{8}$/);
    expect(resultForFirst.tool_use_id).toBe(first.id);
    expect(resultForSecond.tool_use_id).toBe(second.id);
    expect(first.id).not.toBe(second.id);
  });

  it('改写是确定性的：同一历史两次编码结果完全相同', () => {
    expect(JSON.stringify(encode(kimiHistory()))).toBe(JSON.stringify(encode(kimiHistory())));
  });

  it('只差非法字符的两个 id 改写后不相撞', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { functionCall: { name: 'a', args: {}, callId: 'call.1' } },
          { functionCall: { name: 'a', args: {}, callId: 'call:1' } },
        ] },
        { role: 'user', parts: [
          { functionResponse: { name: 'a', response: {}, callId: 'call.1' } },
          { functionResponse: { name: 'a', response: {}, callId: 'call:1' } },
        ] },
      ],
    });
    const ids = body.messages[1].content.map((block: any) => block.id);
    expect(ids[0]).not.toBe(ids[1]);
    expect(body.messages[2].content.map((block: any) => block.tool_use_id)).toEqual(ids);
  });

  it('工具结果没有 callId 时按顺序配对到改写后的 id', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [{ functionCall: { name: 'a', args: {}, callId: 'functions.a:0' } }] },
        { role: 'user', parts: [{ functionResponse: { name: 'a', response: {} } }] },
      ],
    });
    expect(body.messages[2].content[0].tool_use_id).toBe(body.messages[1].content[0].id);
    expect(body.messages[1].content[0].id).toMatch(CLAUDE_ID);
  });

  it('合规 id（toolu_ / call_ / 带连字符）原样保留', () => {
    const body = encode({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { functionCall: { name: 'a', args: {}, callId: 'toolu_01A09q90qw90lq917835lq9' } },
          { functionCall: { name: 'a', args: {}, callId: 'call_abc-DEF_123' } },
        ] },
        { role: 'user', parts: [
          { functionResponse: { name: 'a', response: {}, callId: 'toolu_01A09q90qw90lq917835lq9' } },
          { functionResponse: { name: 'a', response: {}, callId: 'call_abc-DEF_123' } },
        ] },
      ],
    });
    expect(body.messages[1].content.map((block: any) => block.id)).toEqual(['toolu_01A09q90qw90lq917835lq9', 'call_abc-DEF_123']);
    expect(body.messages[2].content.map((block: any) => block.tool_use_id)).toEqual(['toolu_01A09q90qw90lq917835lq9', 'call_abc-DEF_123']);
  });

  it('解码 Claude 响应时 tool_use.id 不改写（存储保留原始 id）', () => {
    const decoded = new ClaudeFormat('claude-sonnet-4-6').decodeResponse({
      content: [{ type: 'tool_use', id: 'toolu_01X', name: 'a', input: {} }],
      stop_reason: 'tool_use',
    });
    expect((decoded.content.parts[0] as any).functionCall.callId).toBe('toolu_01X');
  });
});
