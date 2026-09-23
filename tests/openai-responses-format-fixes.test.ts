/**
 * OpenAI Responses 格式修复（组 B：B4 strict、B5 显式缓存断点、B6 phase、B8 服务端 compaction）。
 *
 * “原行为不变”回归：BASELINE_* 常量是修复前（limcode/provider-fixes 16b50ab）的编码 / 解码结果，
 * 用 JSON.stringify 逐字节比较（含键顺序）。
 */
import { describe, expect, it } from 'vitest';

import { OpenAIResponsesFormat } from '../src/index.js';
import type { LLMRequest } from '../src/index.js';

// ---- 与 /tmp/api-research/responses/sse.mjs 同形的脚本化 SSE 事件 ----
function sseEvents(responseId: string, output: any[]): any[] {
  const events: any[] = [];
  let seq = 0;
  const push = (type: string, data: any) => events.push({ type, sequence_number: seq++, ...data });
  push('response.created', { response: { id: responseId, object: 'response', status: 'in_progress', output: [] } });
  output.forEach((item, output_index) => {
    if (item.type === 'reasoning') {
      push('response.output_item.added', { output_index, item: { type: 'reasoning', id: item.id, summary: [] } });
      (item.summary ?? []).forEach((s: any, summary_index: number) => {
        push('response.reasoning_summary_part.added', { item_id: item.id, output_index, summary_index, part: { type: 'summary_text', text: '' } });
        push('response.reasoning_summary_text.delta', { item_id: item.id, output_index, summary_index, delta: s.text });
        push('response.reasoning_summary_text.done', { item_id: item.id, output_index, summary_index, text: s.text });
        push('response.reasoning_summary_part.done', { item_id: item.id, output_index, summary_index, part: { type: 'summary_text', text: s.text } });
      });
      push('response.output_item.done', { output_index, item });
    } else if (item.type === 'function_call') {
      push('response.output_item.added', { output_index, item: { ...item, arguments: '', status: 'in_progress' } });
      push('response.function_call_arguments.delta', { item_id: item.id, output_index, delta: item.arguments });
      push('response.function_call_arguments.done', { item_id: item.id, output_index, arguments: item.arguments });
      push('response.output_item.done', { output_index, item: { ...item, status: 'completed' } });
    } else if (item.type === 'message') {
      const text = item.content[0].text;
      push('response.output_item.added', { output_index, item: { ...item, content: [], status: 'in_progress' } });
      push('response.content_part.added', { item_id: item.id, output_index, content_index: 0, part: { type: 'output_text', text: '', annotations: [] } });
      push('response.output_text.delta', { item_id: item.id, output_index, content_index: 0, delta: text.slice(0, 3) });
      push('response.output_text.delta', { item_id: item.id, output_index, content_index: 0, delta: text.slice(3) });
      push('response.output_text.done', { item_id: item.id, output_index, content_index: 0, text });
      push('response.content_part.done', { item_id: item.id, output_index, content_index: 0, part: { type: 'output_text', text, annotations: [] } });
      push('response.output_item.done', { output_index, item: { ...item, status: 'completed' } });
    } else {
      push('response.output_item.added', { output_index, item });
      push('response.output_item.done', { output_index, item });
    }
  });
  push('response.completed', { response: { id: responseId, object: 'response', status: 'completed', output: output.map(i => ({ ...i, ...(i.type === 'reasoning' ? {} : { status: 'completed' }) })), usage: { input_tokens: 100, output_tokens: 20, total_tokens: 120, input_tokens_details: { cached_tokens: 0 }, output_tokens_details: { reasoning_tokens: 10 } } } });
  return events;
}
const reasoning = (id: string, text: string) => ({ type: 'reasoning', id, summary: [{ type: 'summary_text', text }], encrypted_content: `gAAAAB_ENC_${id}` });
const fcall = (n: string, name: string, args: any = {}) => ({ type: 'function_call', id: `fc_${n}`, call_id: `call_${n}`, name, arguments: JSON.stringify(args) });
const msg = (n: string, text: string, phase?: string) => ({ type: 'message', id: `msg_${n}`, role: 'assistant', ...(phase ? { phase } : {}), content: [{ type: 'output_text', text, annotations: [] }] });

function decodeAll(format: OpenAIResponsesFormat, events: any[]): any[] {
  const state = format.createStreamState();
  return events.map(event => format.decodeStreamChunk(structuredClone(event), state));
}

const TOOLS = [{ functionDeclarations: [
  { name: 'read_file', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' }, limit: { type: 'integer' } }, required: ['path'] } },
  { name: 'list_conversations', description: 'List', parameters: { type: 'object', properties: {} } },
] }] as LLMRequest['tools'];

const HISTORY_REQUEST: LLMRequest = {
  systemInstruction: { parts: [{ text: 'You are helpful.' }] },
  contents: [
    { role: 'user', parts: [{ text: 'read a.txt' }] },
    { role: 'model', parts: [
      { text: 'plan', thought: true, thoughtSignatures: { 'openai-responses': 'gAAAAB_enc_1' } },
      { text: 'Reading now.' },
      { functionCall: { name: 'read_file', args: { path: 'a.txt' }, callId: 'call_1' } },
    ] },
    { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { content: 'hello' }, callId: 'call_1' } }] },
    { role: 'model', parts: [{ text: 'The file says hello.' }] },
    { role: 'user', parts: [{ text: 'thanks' }] },
  ],
  generationConfig: { maxOutputTokens: 2048, thinkingConfig: { thinkingLevel: 'high' } },
};

const TOOL_LOOP_REQUEST: LLMRequest = {
  contents: [
    { role: 'user', parts: [{ text: 'read a.txt' }] },
    { role: 'model', parts: [{ functionCall: { name: 'read_file', args: { path: 'a.txt' }, callId: 'call_1' } }] },
    { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { content: 'hello' }, callId: 'call_1' } }] },
  ],
};

const BASELINE_NO_CACHE_BODY = {
  "model": "gpt-5.5",
  "store": false,
  "include": [
    "reasoning.encrypted_content"
  ],
  "instructions": "You are helpful.",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "read a.txt"
        }
      ]
    },
    {
      "type": "reasoning",
      "summary": [
        {
          "type": "summary_text",
          "text": "plan"
        }
      ],
      "encrypted_content": "gAAAAB_enc_1"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Reading now."
        }
      ]
    },
    {
      "type": "function_call",
      "call_id": "call_1",
      "name": "read_file",
      "arguments": "{\"path\":\"a.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"content\":\"hello\"}"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "thanks"
        }
      ]
    }
  ],
  "max_output_tokens": 2048,
  "reasoning": {
    "effort": "high",
    "summary": "detailed"
  },
  "stream": true
};

const BASELINE_KEY_MODE_BODY = {
  "model": "gpt-5.5",
  "store": false,
  "include": [
    "reasoning.encrypted_content"
  ],
  "instructions": "You are helpful.",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "read a.txt"
        }
      ]
    },
    {
      "type": "reasoning",
      "summary": [
        {
          "type": "summary_text",
          "text": "plan"
        }
      ],
      "encrypted_content": "gAAAAB_enc_1"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Reading now."
        }
      ]
    },
    {
      "type": "function_call",
      "call_id": "call_1",
      "name": "read_file",
      "arguments": "{\"path\":\"a.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"content\":\"hello\"}"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "thanks"
        }
      ]
    }
  ],
  "prompt_cache_key": "k1",
  "max_output_tokens": 2048,
  "reasoning": {
    "effort": "high",
    "summary": "detailed"
  },
  "stream": true
};

const BASELINE_IMPLICIT_BODY = {
  "model": "gpt-5.6",
  "store": false,
  "include": [
    "reasoning.encrypted_content"
  ],
  "instructions": "You are helpful.",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "read a.txt"
        }
      ]
    },
    {
      "type": "reasoning",
      "summary": [
        {
          "type": "summary_text",
          "text": "plan"
        }
      ],
      "encrypted_content": "gAAAAB_enc_1"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Reading now."
        }
      ]
    },
    {
      "type": "function_call",
      "call_id": "call_1",
      "name": "read_file",
      "arguments": "{\"path\":\"a.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"content\":\"hello\"}"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "thanks",
          "prompt_cache_breakpoint": {
            "mode": "explicit"
          }
        }
      ]
    }
  ],
  "prompt_cache_key": "k1",
  "prompt_cache_options": {
    "mode": "implicit",
    "ttl": "30m"
  },
  "max_output_tokens": 2048,
  "reasoning": {
    "effort": "high",
    "summary": "detailed"
  },
  "stream": true
};

const BASELINE_IMPLICIT_TOOL_LOOP_BODY = {
  "model": "gpt-5.6",
  "store": false,
  "include": [
    "reasoning.encrypted_content"
  ],
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "read a.txt"
        }
      ]
    },
    {
      "type": "function_call",
      "call_id": "call_1",
      "name": "read_file",
      "arguments": "{\"path\":\"a.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"content\":\"hello\"}"
    },
    {
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": " ",
          "prompt_cache_breakpoint": {
            "mode": "explicit"
          }
        }
      ]
    }
  ],
  "prompt_cache_options": {
    "mode": "implicit",
    "ttl": "30m"
  },
  "stream": true
};

const BASELINE_EXPLICIT_BODY = {
  "model": "gpt-5.6",
  "store": false,
  "include": [
    "reasoning.encrypted_content"
  ],
  "instructions": "You are helpful.",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "read a.txt"
        }
      ]
    },
    {
      "type": "reasoning",
      "summary": [
        {
          "type": "summary_text",
          "text": "plan"
        }
      ],
      "encrypted_content": "gAAAAB_enc_1"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Reading now."
        }
      ]
    },
    {
      "type": "function_call",
      "call_id": "call_1",
      "name": "read_file",
      "arguments": "{\"path\":\"a.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"content\":\"hello\"}"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "thanks",
          "prompt_cache_breakpoint": {
            "mode": "explicit"
          }
        }
      ]
    }
  ],
  "prompt_cache_options": {
    "mode": "explicit",
    "ttl": "30m"
  },
  "max_output_tokens": 2048,
  "reasoning": {
    "effort": "high",
    "summary": "detailed"
  },
  "stream": true
};

const BASELINE_ASTRA_EXPLICIT_BODY = {
  "model": "gpt-6-astra",
  "store": false,
  "include": [
    "reasoning.encrypted_content"
  ],
  "input": [
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "You are helpful.",
          "prompt_cache_breakpoint": {
            "mode": "explicit"
          }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "read a.txt"
        }
      ]
    },
    {
      "type": "reasoning",
      "summary": [
        {
          "type": "summary_text",
          "text": "plan"
        }
      ],
      "encrypted_content": "gAAAAB_enc_1"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Reading now."
        }
      ]
    },
    {
      "type": "function_call",
      "call_id": "call_1",
      "name": "read_file",
      "arguments": "{\"path\":\"a.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"content\":\"hello\"}"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "thanks",
          "prompt_cache_breakpoint": {
            "mode": "explicit"
          }
        }
      ]
    }
  ],
  "prompt_cache_options": {
    "mode": "explicit",
    "ttl": "30m"
  },
  "max_output_tokens": 2048,
  "reasoning": {
    "effort": "high",
    "summary": "detailed"
  },
  "stream": true
};

const BASELINE_WITH_TOOLS_BODY = {
  "model": "gpt-5.5",
  "store": false,
  "include": [
    "reasoning.encrypted_content"
  ],
  "instructions": "You are helpful.",
  "input": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "read a.txt"
        }
      ]
    },
    {
      "type": "reasoning",
      "summary": [
        {
          "type": "summary_text",
          "text": "plan"
        }
      ],
      "encrypted_content": "gAAAAB_enc_1"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Reading now."
        }
      ]
    },
    {
      "type": "function_call",
      "call_id": "call_1",
      "name": "read_file",
      "arguments": "{\"path\":\"a.txt\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"content\":\"hello\"}"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The file says hello."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "thanks"
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "read_file",
      "description": "Read a file",
      "parameters": {
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
      "type": "function",
      "name": "list_conversations",
      "description": "List",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  ],
  "max_output_tokens": 2048,
  "reasoning": {
    "effort": "high",
    "summary": "detailed"
  },
  "stream": true
};

/** 不带 phase 的普通输出：reasoning → message → function_call。 */
const NO_PHASE_OUTPUT = [reasoning('rs_1', 'plan'), msg('1', 'Let me check.'), fcall('1', 'read_file', { path: 'a' })];
const PHASE_OUTPUT = [reasoning('rs_1', 'plan'), msg('1', 'Let me check.', 'commentary'), fcall('1', 'read_file', { path: 'a' })];

const BASELINE_STREAM_NO_PHASE = [
  {},
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
  {},
  {},
  {
    "partsDelta": [
      {
        "thought": true,
        "thoughtSignatures": {
          "openai-responses": "gAAAAB_ENC_rs_1"
        }
      }
    ],
    "thoughtSignatures": {
      "openai-responses": "gAAAAB_ENC_rs_1"
    }
  },
  {},
  {},
  {
    "textDelta": "Let",
    "partsDelta": [
      {
        "text": "Let"
      }
    ]
  },
  {
    "textDelta": " me check.",
    "partsDelta": [
      {
        "text": " me check."
      }
    ]
  },
  {},
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
          "callId": "call_1"
        }
      }
    ],
    "partsDelta": [
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "call_1"
        }
      }
    ]
  },
  {},
  {
    "usageMetadata": {
      "promptTokenCount": 100,
      "thoughtsTokenCount": 10,
      "candidatesTokenCount": 20,
      "totalTokenCount": 120
    }
  }
];

const BASELINE_DECODE_NO_PHASE = {
  "content": {
    "role": "model",
    "parts": [
      {
        "thought": true,
        "text": "plan",
        "thoughtSignatures": {
          "openai-responses": "gAAAAB_ENC_rs_1"
        }
      },
      {
        "text": "Let me check."
      },
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "call_1"
        }
      }
    ]
  },
  "usageMetadata": {
    "promptTokenCount": 1,
    "candidatesTokenCount": 2,
    "totalTokenCount": 3
  }
};

const BASELINE_ASTRA_STREAM_PHASE = [
  {
    "nativeEvent": {
      "type": "response.created",
      "responseId": "resp_a"
    }
  },
  {
    "outputItem": {
      "id": "rs_1",
      "ordinal": 0
    }
  },
  {
    "outputItem": {
      "id": "rs_1",
      "ordinal": 0
    }
  },
  {
    "outputItem": {
      "id": "rs_1",
      "ordinal": 0
    },
    "partsDelta": [
      {
        "text": "plan",
        "thought": true
      }
    ]
  },
  {
    "outputItem": {
      "id": "rs_1",
      "ordinal": 0
    }
  },
  {
    "outputItem": {
      "id": "rs_1",
      "ordinal": 0
    }
  },
  {
    "outputItem": {
      "id": "rs_1",
      "ordinal": 0
    },
    "partsDelta": [
      {
        "thought": true,
        "thoughtSignatures": {
          "openai-responses": "gAAAAB_ENC_rs_1"
        }
      }
    ],
    "thoughtSignatures": {
      "openai-responses": "gAAAAB_ENC_rs_1"
    }
  },
  {
    "outputItem": {
      "id": "msg_1",
      "ordinal": 1,
      "phase": "commentary"
    }
  },
  {},
  {
    "textDelta": "Let",
    "partsDelta": [
      {
        "text": "Let"
      }
    ],
    "outputItem": {
      "id": "msg_1",
      "ordinal": 1
    }
  },
  {
    "textDelta": " me check.",
    "partsDelta": [
      {
        "text": " me check."
      }
    ],
    "outputItem": {
      "id": "msg_1",
      "ordinal": 1
    }
  },
  {},
  {},
  {
    "outputItem": {
      "id": "msg_1",
      "ordinal": 1,
      "phase": "commentary"
    }
  },
  {
    "outputItem": {
      "id": "fc_1",
      "ordinal": 2
    }
  },
  {
    "outputItem": {
      "id": "fc_1",
      "ordinal": 2
    }
  },
  {
    "outputItem": {
      "id": "fc_1",
      "ordinal": 2
    },
    "functionCalls": [
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "call_1"
        }
      }
    ],
    "partsDelta": [
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "call_1"
        }
      }
    ]
  },
  {
    "outputItem": {
      "id": "fc_1",
      "ordinal": 2
    }
  },
  {
    "usageMetadata": {
      "promptTokenCount": 100,
      "thoughtsTokenCount": 10,
      "candidatesTokenCount": 20,
      "totalTokenCount": 120
    },
    "nativeEvent": {
      "type": "response.completed",
      "responseId": "resp_a",
      "usage": {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "input_tokens_details": {
          "cached_tokens": 0
        },
        "output_tokens_details": {
          "reasoning_tokens": 10
        }
      }
    },
    "completedContents": [
      {
        "role": "model",
        "parts": [
          {
            "thought": true,
            "text": "plan",
            "thoughtSignatures": {
              "openai-responses": "gAAAAB_ENC_rs_1"
            },
            "outputItem": {
              "id": "rs_1",
              "ordinal": 0
            }
          }
        ],
        "providerContext": {
          "provider": "openai",
          "format": "openai-responses",
          "endpoint": "responses",
          "itemType": "reasoning",
          "id": "rs_1",
          "encryptedContent": "gAAAAB_ENC_rs_1",
          "rawItem": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [
              {
                "type": "summary_text",
                "text": "plan"
              }
            ],
            "encrypted_content": "gAAAAB_ENC_rs_1"
          }
        }
      },
      {
        "role": "model",
        "parts": [
          {
            "text": "Let me check.",
            "outputItem": {
              "id": "msg_1",
              "ordinal": 1,
              "phase": "commentary"
            }
          }
        ],
        "providerContext": {
          "provider": "openai",
          "format": "openai-responses",
          "endpoint": "responses",
          "itemType": "message",
          "id": "msg_1",
          "rawItem": {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "phase": "commentary",
            "content": [
              {
                "type": "output_text",
                "text": "Let me check.",
                "annotations": []
              }
            ],
            "status": "completed"
          }
        }
      },
      {
        "role": "model",
        "parts": [
          {
            "functionCall": {
              "name": "read_file",
              "args": {
                "path": "a"
              },
              "callId": "call_1"
            },
            "outputItem": {
              "id": "fc_1",
              "ordinal": 2
            }
          }
        ],
        "providerContext": {
          "provider": "openai",
          "format": "openai-responses",
          "endpoint": "responses",
          "itemType": "function_call",
          "id": "fc_1",
          "rawItem": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": "{\"path\":\"a\"}",
            "status": "completed"
          }
        }
      }
    ]
  }
];

const BASELINE_ASTRA_DECODE_PHASE = {
  "content": {
    "role": "model",
    "parts": [
      {
        "thought": true,
        "text": "plan",
        "thoughtSignatures": {
          "openai-responses": "gAAAAB_ENC_rs_1"
        }
      },
      {
        "text": "Let me check."
      },
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "call_1"
        }
      }
    ]
  }
};

const BASELINE_ASTRA_WS_STREAM_PHASE = [
  {},
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
  {},
  {},
  {
    "partsDelta": [
      {
        "thought": true,
        "thoughtSignatures": {
          "openai-responses": "gAAAAB_ENC_rs_1"
        }
      }
    ],
    "thoughtSignatures": {
      "openai-responses": "gAAAAB_ENC_rs_1"
    }
  },
  {},
  {},
  {
    "textDelta": "Let",
    "partsDelta": [
      {
        "text": "Let"
      }
    ]
  },
  {
    "textDelta": " me check.",
    "partsDelta": [
      {
        "text": " me check."
      }
    ]
  },
  {},
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
          "callId": "call_1"
        }
      }
    ],
    "partsDelta": [
      {
        "functionCall": {
          "name": "read_file",
          "args": {
            "path": "a"
          },
          "callId": "call_1"
        }
      }
    ]
  },
  {},
  {
    "usageMetadata": {
      "promptTokenCount": 100,
      "thoughtsTokenCount": 10,
      "candidatesTokenCount": 20,
      "totalTokenCount": 120
    }
  }
];

describe('OpenAIResponsesFormat 原行为不变（修复前基线逐字节比较）', () => {
  it('无缓存、无工具的请求体与修复前逐字节一致', () => {
    const body = new OpenAIResponsesFormat('gpt-5.5').encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(body)).toBe(JSON.stringify(BASELINE_NO_CACHE_BODY));
  });

  it('key 缓存模式请求体与修复前逐字节一致', () => {
    const body = new OpenAIResponsesFormat('gpt-5.5', { enabled: true, mode: 'key', key: 'k1' }).encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(body)).toBe(JSON.stringify(BASELINE_KEY_MODE_BODY));
  });

  it('implicit 缓存模式请求体与修复前逐字节一致（B5 只改 explicit）', () => {
    const body = new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'implicit', key: 'k1' }).encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(body)).toBe(JSON.stringify(BASELINE_IMPLICIT_BODY));
    const loop = new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'implicit' }).encodeRequest(TOOL_LOOP_REQUEST, true);
    expect(JSON.stringify(loop)).toBe(JSON.stringify(BASELINE_IMPLICIT_TOOL_LOOP_BODY));
  });

  it('explicit 模式下末尾本来就是 user input_text 时，请求体与修复前逐字节一致', () => {
    const body = new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'explicit' }).encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(body)).toBe(JSON.stringify(BASELINE_EXPLICIT_BODY));
    const astra = new OpenAIResponsesFormat('gpt-6-astra', { enabled: true, mode: 'explicit' }).encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(astra)).toBe(JSON.stringify(BASELINE_ASTRA_EXPLICIT_BODY));
  });

  it('没有 phase 的流式事件解码结果与修复前逐字节一致（不附加 outputItem）', () => {
    const chunks = decodeAll(new OpenAIResponsesFormat('gpt-5.5', undefined, true), sseEvents('resp_1', NO_PHASE_OUTPUT));
    expect(JSON.stringify(chunks)).toBe(JSON.stringify(BASELINE_STREAM_NO_PHASE));
  });

  it('没有 phase 的非流式响应解码结果与修复前逐字节一致', () => {
    const decoded = new OpenAIResponsesFormat('gpt-5.5', undefined, true)
      .decodeResponse(structuredClone({ id: 'resp_1', output: NO_PHASE_OUTPUT, usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 } }));
    expect(JSON.stringify(decoded)).toBe(JSON.stringify(BASELINE_DECODE_NO_PHASE));
  });

  it('Astra 原生解码（HTTP 与 WS 构造方式）带 phase 时与修复前逐字节一致', () => {
    const http = decodeAll(new OpenAIResponsesFormat('gpt-6-astra', undefined, true), sseEvents('resp_a', PHASE_OUTPUT));
    expect(JSON.stringify(http)).toBe(JSON.stringify(BASELINE_ASTRA_STREAM_PHASE));
    const ws = decodeAll(new OpenAIResponsesFormat('gpt-6-astra'), sseEvents('resp_w', PHASE_OUTPUT));
    expect(JSON.stringify(ws)).toBe(JSON.stringify(BASELINE_ASTRA_WS_STREAM_PHASE));
    const decoded = new OpenAIResponsesFormat('gpt-6-astra', undefined, true).decodeResponse(structuredClone({ id: 'resp_a', output: PHASE_OUTPUT }));
    expect(JSON.stringify(decoded)).toBe(JSON.stringify(BASELINE_ASTRA_DECODE_PHASE));
  });
});

describe('B4 Responses 函数工具补 strict:false', () => {
  // 依据：https://developers.openai.com/api/docs/guides/function-calling#strict-mode
  // “If you omit strict ... Responses requests will attempt to normalize your schema into strict mode ...
  //  To opt out of strict mode in Responses and keep non-strict, best-effort function calling, explicitly set strict: false.”
  it('未声明 strict 的函数工具显式发送 strict:false，其余字段与修复前逐字节一致', () => {
    const body = new OpenAIResponsesFormat('gpt-5.5').encodeRequest({ ...HISTORY_REQUEST, tools: TOOLS }, true) as any;
    expect(body.tools.map((tool: any) => tool.strict)).toEqual([false, false]);
    expect(Object.keys(body.tools[0])).toEqual(['type', 'name', 'description', 'parameters', 'strict']);

    const withoutStrict = {
      ...body,
      tools: body.tools.map(({ strict: _strict, ...rest }: any) => rest),
    };
    expect(JSON.stringify(withoutStrict)).toBe(JSON.stringify(BASELINE_WITH_TOOLS_BODY));
  });

  it('声明里显式给了 strict 的保持原值（true / false）', () => {
    const body = new OpenAIResponsesFormat('gpt-5.5').encodeRequest({
      contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
      tools: [{ functionDeclarations: [
        { name: 'strict_tool', description: 's', strict: true, parameters: { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] } },
        { name: 'loose_tool', description: 'l', strict: false, parameters: { type: 'object', properties: {} } },
        { name: 'default_tool', description: 'd', parameters: { type: 'object', properties: {} } },
      ] }],
    }) as any;
    expect(body.tools.map((tool: any) => [tool.name, tool.strict])).toEqual([
      ['strict_tool', true],
      ['loose_tool', false],
      ['default_tool', false],
    ]);
  });

  it('Astra 异步工具声明仍原样透传 async:true，并同样补 strict:false', () => {
    const body = new OpenAIResponsesFormat('gpt-6-astra').encodeRequest({
      contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
      tools: [{ functionDeclarations: [{ name: 'bg', description: 'b', async: true, parameters: { type: 'object', properties: {} } }] }],
    }) as any;
    expect(body.tools).toEqual([{
      type: 'function', name: 'bg', description: 'b', parameters: { type: 'object', properties: {} }, strict: false, async: true,
    }]);
  });
});

describe('B5 explicit 缓存断点放到最后一个可承载块，不再追加空格假消息', () => {
  // 依据：https://developers.openai.com/api/docs/guides/prompt-caching
  // Explicit mode：“mark each desired breakpoint by adding prompt_cache_breakpoint: { "mode": "explicit" }
  // to a supported content block inside an input message”；多轮 agent 示例把断点放在
  // function_call_output 数组形式的 input_text 上。
  const BREAKPOINT = { mode: 'explicit' };
  const explicit = (model = 'gpt-5.6') => new OpenAIResponsesFormat(model, { enabled: true, mode: 'explicit' });

  it('工具循环末尾是字符串 function_call_output：断点落在最后一条 user input_text，不追加假消息', () => {
    const body = explicit().encodeRequest(TOOL_LOOP_REQUEST, true) as any;
    expect(body.input).toEqual([
      { role: 'user', content: [{ type: 'input_text', text: 'read a.txt', prompt_cache_breakpoint: BREAKPOINT }] },
      { type: 'function_call', call_id: 'call_1', name: 'read_file', arguments: '{"path":"a.txt"}' },
      { type: 'function_call_output', call_id: 'call_1', output: '{"content":"hello"}' },
    ]);
    expect(JSON.stringify(body)).not.toContain('"text":" "');
  });

  it('末尾 function_call_output 是数组形式时，断点落在它的最后一个内容块上', () => {
    const body = explicit().encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'shot' }] },
        { role: 'model', parts: [{ functionCall: { name: 'screenshot', args: {}, callId: 'call_s' } }] },
        { role: 'user', parts: [{ functionResponse: {
          name: 'screenshot', response: { ok: true }, callId: 'call_s',
          parts: [{ inlineData: { mimeType: 'image/png', data: 'iVBORw0KGgo=' } }],
        } }] },
      ],
    }, true) as any;
    const output = body.input.at(-1);
    expect(output.type).toBe('function_call_output');
    expect(output.output.at(-1)).toMatchObject({ type: 'input_image', prompt_cache_breakpoint: BREAKPOINT });
    expect(output.output[0].prompt_cache_breakpoint).toBeUndefined();
    expect(body.input[0].content[0].prompt_cache_breakpoint).toBeUndefined();
    expect(body.input).toHaveLength(3);
  });

  it('只打一个断点：更早的可承载块不重复标记', () => {
    const body = explicit().encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'q1' }] },
        { role: 'model', parts: [{ text: 'a1' }] },
        { role: 'user', parts: [{ text: 'q2' }] },
        { role: 'model', parts: [{ functionCall: { name: 'a', args: {}, callId: 'call_1' } }] },
        { role: 'user', parts: [{ functionResponse: { name: 'a', response: {}, callId: 'call_1' } }] },
      ],
    }, true) as any;
    const marked = JSON.stringify(body.input).split('prompt_cache_breakpoint').length - 1;
    expect(marked).toBe(1);
    expect(body.input[2].content[0]).toEqual({ type: 'input_text', text: 'q2', prompt_cache_breakpoint: BREAKPOINT });
  });

  it('providerContext 原样回放的 user message 也可承载，且不改动存储里的原始 item', () => {
    const rawUserItem = { id: 'msg_u1', type: 'message', status: 'completed', role: 'user', content: [{ type: 'input_text', text: 'first question' }] };
    const rawCompaction = { id: 'cmp_1', type: 'compaction', encrypted_content: 'gAAAAB_cmp' };
    const request: LLMRequest = {
      contents: [
        { role: 'user', parts: [{ providerContext: { provider: 'openai', format: 'openai-responses', endpoint: 'responses.compact', itemType: 'message', rawItem: rawUserItem } }] },
        { role: 'model', parts: [{ providerContext: { provider: 'openai', format: 'openai-responses', endpoint: 'responses.compact', itemType: 'compaction', rawItem: rawCompaction } }] },
      ],
    };
    const body = explicit().encodeRequest(request, true) as any;
    expect(body.input).toEqual([
      { ...rawUserItem, content: [{ type: 'input_text', text: 'first question', prompt_cache_breakpoint: BREAKPOINT }] },
      rawCompaction,
    ]);
    expect((rawUserItem.content[0] as any).prompt_cache_breakpoint).toBeUndefined();
  });

  it('整个 input 没有可承载块时才退回旧做法（追加带断点的空格 user 消息）', () => {
    const body = explicit().encodeRequest({
      contents: [{ role: 'model', parts: [{ providerContext: {
        provider: 'openai', format: 'openai-responses', endpoint: 'responses', itemType: 'compaction',
        rawItem: { id: 'cmp_1', type: 'compaction', encrypted_content: 'gAAAAB_cmp' },
      } }] }],
    }, true) as any;
    expect(body.input).toEqual([
      { id: 'cmp_1', type: 'compaction', encrypted_content: 'gAAAAB_cmp' },
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: ' ', prompt_cache_breakpoint: BREAKPOINT }] },
    ]);
  });

  it('Astra explicit：开发者指令断点保持不变，工具循环末尾不再追加假消息', () => {
    const body = explicit('gpt-6-astra').encodeRequest({
      systemInstruction: { parts: [{ text: 'stable' }] },
      ...TOOL_LOOP_REQUEST,
    }, true) as any;
    expect(body.instructions).toBeUndefined();
    expect(body.input[0]).toEqual({ role: 'developer', content: [{ type: 'input_text', text: 'stable', prompt_cache_breakpoint: BREAKPOINT }] });
    expect(body.input[1].content[0]).toEqual({ type: 'input_text', text: 'read a.txt', prompt_cache_breakpoint: BREAKPOINT });
    expect(body.input).toHaveLength(4);
  });

  it('breakpoints.messages=false 时不打消息断点（与修复前一致）', () => {
    const body = new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'explicit', breakpoints: { messages: false } })
      .encodeRequest(TOOL_LOOP_REQUEST, true) as any;
    expect(JSON.stringify(body.input)).not.toContain('prompt_cache_breakpoint');
    expect(body.input).toHaveLength(3);
    expect(body.prompt_cache_options).toEqual({ mode: 'explicit', ttl: '30m' });
  });
});

describe('B6 非 Astra 模型的 assistant message phase 保留在 outputItem 上', () => {
  // 依据：https://developers.openai.com/api/docs/guides/reasoning#phase-parameter
  // “If you replay assistant history manually, preserve each original phase value.
  //  Missing or dropped phase can cause preambles to be treated as final answers.”
  const TWO_MESSAGES = [
    reasoning('rs_1', 'plan'),
    msg('pre', 'Let me check first.', 'commentary'),
    fcall('1', 'read_file', { path: 'a' }),
    msg('fin', 'All done.', 'final_answer'),
  ];

  it('流式：output_item.added 带 phase 时，该 message 的 output_text.delta 附带 {id, ordinal, phase}', () => {
    const chunks = decodeAll(new OpenAIResponsesFormat('gpt-5.5', undefined, true), sseEvents('resp_1', TWO_MESSAGES));
    const textChunks = chunks.filter(chunk => chunk.textDelta);
    expect(textChunks.map(chunk => [chunk.textDelta, chunk.outputItem])).toEqual([
      ['Let', { id: 'msg_pre', ordinal: 1, phase: 'commentary' }],
      [' me check first.', { id: 'msg_pre', ordinal: 1, phase: 'commentary' }],
      ['All', { id: 'msg_fin', ordinal: 3, phase: 'final_answer' }],
      [' done.', { id: 'msg_fin', ordinal: 3, phase: 'final_answer' }],
    ]);
    // 只有文本 delta 附带 outputItem；reasoning / 工具调用 / 生命周期事件不变。
    expect(chunks.filter(chunk => chunk.outputItem)).toHaveLength(4);
  });

  it('流式：带 phase 与不带 phase 的同一输出，除文本 delta 的 outputItem 外逐字节相同', () => {
    const withPhase = decodeAll(new OpenAIResponsesFormat('gpt-5.5', undefined, true), sseEvents('resp_1', PHASE_OUTPUT))
      .map(({ outputItem: _outputItem, ...rest }) => rest);
    expect(JSON.stringify(withPhase)).toBe(JSON.stringify(BASELINE_STREAM_NO_PHASE));
  });

  it('流式：delta 缺 item_id 时按 output_index 找到 phase', () => {
    const format = new OpenAIResponsesFormat('gpt-5.5');
    const state = format.createStreamState();
    format.decodeStreamChunk({ type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_x', role: 'assistant', phase: 'final_answer', content: [] } }, state);
    const chunk = format.decodeStreamChunk({ type: 'response.output_text.delta', output_index: 0, delta: 'hi' }, state);
    expect(chunk.outputItem).toEqual({ id: 'msg_x', ordinal: 0, phase: 'final_answer' });
  });

  it('非法 phase 值（null / 未知字符串）视为没有 phase', () => {
    for (const phase of [null, 'analysis', '']) {
      const format = new OpenAIResponsesFormat('gpt-5.5');
      const state = format.createStreamState();
      format.decodeStreamChunk({ type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_x', role: 'assistant', phase, content: [] } }, state);
      const chunk = format.decodeStreamChunk({ type: 'response.output_text.delta', item_id: 'msg_x', output_index: 0, delta: 'hi' }, state);
      expect(chunk).toEqual({ textDelta: 'hi', partsDelta: [{ text: 'hi' }] });
    }
  });

  it('outputItem 形状满足 LimCode 扩展 modelOutputItemFromValue 的要求（非空 id、非负安全整数 ordinal）', () => {
    const chunks = decodeAll(new OpenAIResponsesFormat('gpt-5.5', undefined, true), sseEvents('resp_1', PHASE_OUTPUT));
    for (const { outputItem } of chunks.filter(chunk => chunk.outputItem)) {
      expect(typeof outputItem.id === 'string' && outputItem.id.trim().length > 0).toBe(true);
      expect(Number.isSafeInteger(outputItem.ordinal) && outputItem.ordinal >= 0).toBe(true);
      expect(['commentary', 'final_answer']).toContain(outputItem.phase);
    }
  });

  it('非流式：带 phase 的 message 文本 part 附带 outputItem，其余 part 不变', () => {
    const decoded = new OpenAIResponsesFormat('gpt-5.5', undefined, true).decodeResponse(structuredClone({ id: 'resp_1', output: TWO_MESSAGES }));
    expect(decoded.content.parts).toEqual([
      { text: 'plan', thought: true, thoughtSignatures: { 'openai-responses': 'gAAAAB_ENC_rs_1' } },
      { text: 'Let me check first.', outputItem: { id: 'msg_pre', ordinal: 1, phase: 'commentary' } },
      { functionCall: { name: 'read_file', args: { path: 'a' }, callId: 'call_1' } },
      { text: 'All done.', outputItem: { id: 'msg_fin', ordinal: 3, phase: 'final_answer' } },
    ]);
  });

  it('编码：带 phase 的文本按 outputItem 拆成各自的 assistant message 并回放 phase', () => {
    const decoded = new OpenAIResponsesFormat('gpt-5.5').decodeResponse(structuredClone({ id: 'resp_1', output: TWO_MESSAGES }));
    const body = new OpenAIResponsesFormat('gpt-5.5').encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        decoded.content,
        { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { ok: 1 }, callId: 'call_1' } }] },
      ],
    }) as any;
    expect(body.input).toEqual([
      { role: 'user', content: [{ type: 'input_text', text: 'go' }] },
      { type: 'reasoning', summary: [{ type: 'summary_text', text: 'plan' }], encrypted_content: 'gAAAAB_ENC_rs_1' },
      { type: 'message', role: 'assistant', phase: 'commentary', content: [{ type: 'output_text', text: 'Let me check first.' }] },
      { type: 'function_call', call_id: 'call_1', name: 'read_file', arguments: '{"path":"a"}' },
      { type: 'message', role: 'assistant', phase: 'final_answer', content: [{ type: 'output_text', text: 'All done.' }] },
      { type: 'function_call_output', call_id: 'call_1', output: '{"ok":1}' },
    ]);
  });

  it('编码：相邻两条不同 phase 的 message 不会被合并；无 phase 的连续文本仍合并为一条', () => {
    const body = new OpenAIResponsesFormat('gpt-5.5').encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'model', parts: [
          { text: 'a', outputItem: { id: 'msg_a', ordinal: 0, phase: 'commentary' } },
          { text: 'b', outputItem: { id: 'msg_b', ordinal: 1, phase: 'final_answer' } },
          { text: 'c' },
          { text: 'd' },
        ] },
      ],
    }) as any;
    expect(body.input.slice(1)).toEqual([
      { type: 'message', role: 'assistant', phase: 'commentary', content: [{ type: 'output_text', text: 'a' }] },
      { type: 'message', role: 'assistant', phase: 'final_answer', content: [{ type: 'output_text', text: 'b' }] },
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'c' }, { type: 'output_text', text: 'd' }] },
    ]);
  });

  it('只有 output_item.done 带 phase 时，已发出的文本 delta 不附加（与修复前一致）', () => {
    const format = new OpenAIResponsesFormat('gpt-5.5');
    const state = format.createStreamState();
    format.decodeStreamChunk({ type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg_x', role: 'assistant', content: [] } }, state);
    const delta = format.decodeStreamChunk({ type: 'response.output_text.delta', item_id: 'msg_x', output_index: 0, delta: 'hi' }, state);
    const done = format.decodeStreamChunk({ type: 'response.output_item.done', output_index: 0, item: { type: 'message', id: 'msg_x', role: 'assistant', phase: 'final_answer', content: [] } }, state);
    expect(delta).toEqual({ textDelta: 'hi', partsDelta: [{ text: 'hi' }] });
    expect(done).toEqual({});
  });
});
