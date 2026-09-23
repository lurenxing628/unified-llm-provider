/**
 * Claude 格式修复（组 B：B1 思考块顺序、B2 redacted_thinking、B3 工具调用 id 规范化）。
 *
 * “原行为不变”回归：BASELINE_* 常量是修复前（limcode/provider-fixes 16b50ab）的编码 / 解码结果，
 * 用 JSON.stringify 逐字节比较（含键顺序）。
 */
import { describe, expect, it } from 'vitest';

import { ClaudeFormat } from '../src/index.js';
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
