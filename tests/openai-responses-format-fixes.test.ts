/**
 * OpenAI Responses 格式修复（组 B：B4 strict、B5 显式缓存断点、B6 phase、B8 服务端 compaction）。
 *
 * “原行为不变”回归：BASELINE_* 常量是修复前（limcode/provider-fixes 16b50ab）的编码 / 解码结果，
 * 用 JSON.stringify 逐字节比较（含键顺序）。
 */
import { describe, expect, it } from 'vitest';

import { OpenAIResponsesFormat, processStreamResponse } from '../src/index.js';
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

/** limcode.7：GPT-6 家族非流式解码与其他模型一样，给带 phase 的 message 文本附带 outputItem；其余字段与基线相同。 */
const ASTRA_DECODE_PHASE_WITH_OUTPUT_ITEM = {
  ...BASELINE_ASTRA_DECODE_PHASE,
  content: {
    ...BASELINE_ASTRA_DECODE_PHASE.content,
    parts: BASELINE_ASTRA_DECODE_PHASE.content.parts.map((part: any) => (part.text === 'Let me check.'
      ? { ...part, outputItem: { id: 'msg_1', ordinal: 1, phase: 'commentary' } }
      : part)),
  },
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
    // BASELINE_EXPLICIT_BODY 录制时用的是 gpt-5.6；limcode.4 起 GPT-5.6 及之后的官方 id 在 explicit 下把
    // instructions 转为带断点的 developer 消息（见下方“GPT-5.6 及之后的显式缓存”），所以这里用清单以外的
    // 模型验证原编码逐字节不变（只替换 model 字段）。
    const body = new OpenAIResponsesFormat('gpt-5.5', { enabled: true, mode: 'explicit' }).encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(body)).toBe(JSON.stringify({ ...BASELINE_EXPLICIT_BODY, model: 'gpt-5.5' }));
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
    // 非流式解码只多出带 phase 的 message 文本上的 outputItem（GPT-6 家族不再排除），其余与修复前逐字节一致。
    const decoded = new OpenAIResponsesFormat('gpt-6-astra', undefined, true).decodeResponse(structuredClone({ id: 'resp_a', output: PHASE_OUTPUT }));
    expect(JSON.stringify(decoded)).toBe(JSON.stringify(ASTRA_DECODE_PHASE_WITH_OUTPUT_ITEM));
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

describe('explicit 续接链：breakpoints.toolOutputs 让工具结果以数组形式承载断点', () => {
  // 依据：https://developers.openai.com/api/docs/guides/prompt-caching（Multi-turn agent 示例：
  // “A breakpoint is added after each tool result”，断点放在 function_call_output 数组形式的 input_text 上）；
  // Responses create 参考：function_call_output 的 output 为 “string or array of ResponseInputTextContent ...”，
  // 只有数组里的 input_text / input_image / input_file 接受 prompt_cache_breakpoint。
  const BREAKPOINT = { mode: 'explicit' };
  const withToolOutputs = (model = 'gpt-5.6', extra: Record<string, unknown> = {}) => new OpenAIResponsesFormat(model, {
    enabled: true, mode: 'explicit', breakpoints: { messages: true, toolOutputs: true }, ...extra,
  });
  const LOOP_TWO_CALLS: LLMRequest = {
    contents: [
      ...TOOL_LOOP_REQUEST.contents,
      { role: 'model', parts: [
        { functionCall: { name: 'read_file', args: { path: 'b.txt' }, callId: 'call_2' } },
        { functionCall: { name: 'read_file', args: { path: 'c.txt' }, callId: 'call_3' } },
      ] },
      { role: 'user', parts: [
        { functionResponse: { name: 'read_file', response: { content: 'bee' }, callId: 'call_2' } },
        { functionResponse: { name: 'read_file', response: { content: 'sea' }, callId: 'call_3' } },
      ] },
    ],
  };
  const withoutBreakpoints = (value: unknown) => JSON.parse(JSON.stringify(value, (key, child) =>
    key === 'prompt_cache_breakpoint' ? undefined : child));

  it('工具循环：所有工具结果转成 input_text 数组，断点只落在最新的工具结果上', () => {
    const body = withToolOutputs().encodeRequest(TOOL_LOOP_REQUEST, true) as any;
    expect(body.input).toEqual([
      { role: 'user', content: [{ type: 'input_text', text: 'read a.txt' }] },
      { type: 'function_call', call_id: 'call_1', name: 'read_file', arguments: '{"path":"a.txt"}' },
      { type: 'function_call_output', call_id: 'call_1', output: [
        { type: 'input_text', text: '{"content":"hello"}', prompt_cache_breakpoint: BREAKPOINT },
      ] },
    ]);
    expect(body.prompt_cache_options).toEqual({ mode: 'explicit', ttl: '30m' });
  });

  it('同一组并行工具结果：全部是数组形式，只在最后一个打断点', () => {
    const body = withToolOutputs().encodeRequest(LOOP_TWO_CALLS, true) as any;
    const outputs = body.input.filter((item: any) => item.type === 'function_call_output');
    expect(outputs.map((item: any) => item.output)).toEqual([
      [{ type: 'input_text', text: '{"content":"hello"}' }],
      [{ type: 'input_text', text: '{"content":"bee"}' }],
      [{ type: 'input_text', text: '{"content":"sea"}', prompt_cache_breakpoint: BREAKPOINT }],
    ]);
    expect(JSON.stringify(body.input).split('prompt_cache_breakpoint').length - 1).toBe(1);
  });

  it('后一请求只在断点标记上与前一请求不同：去掉断点后前一请求的 input 是后一请求的前缀', () => {
    const format = withToolOutputs('gpt-6-astra');
    const request = { systemInstruction: { parts: [{ text: 'stable' }] } };
    const first = format.encodeRequest({ ...request, ...TOOL_LOOP_REQUEST }, true) as any;
    const second = format.encodeRequest({ ...request, ...LOOP_TWO_CALLS }, true) as any;
    expect(first.input[0]).toEqual({ role: 'developer', content: [{ type: 'input_text', text: 'stable', prompt_cache_breakpoint: BREAKPOINT }] });
    expect(withoutBreakpoints(second.input.slice(0, first.input.length))).toEqual(withoutBreakpoints(first.input));
    expect(first.input.at(-1).output[0].prompt_cache_breakpoint).toEqual(BREAKPOINT);
    expect(second.input[first.input.length - 1].output[0].prompt_cache_breakpoint).toBeUndefined();
  });

  it('已是数组形式（带附件）的工具结果保持原样，断点落在最后一个内容块', () => {
    const body = withToolOutputs().encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'shot' }] },
        { role: 'model', parts: [{ functionCall: { name: 'screenshot', args: {}, callId: 'call_s' } }] },
        { role: 'user', parts: [{ functionResponse: {
          name: 'screenshot', response: { ok: true }, callId: 'call_s',
          parts: [{ inlineData: { mimeType: 'image/png', data: 'iVBORw0KGgo=' } }],
        } }] },
      ],
    }, true) as any;
    const output = body.input.at(-1).output;
    expect(output[0]).toEqual({ type: 'input_text', text: '{"ok":true}' });
    expect(output.at(-1)).toMatchObject({ type: 'input_image', prompt_cache_breakpoint: BREAKPOINT });
  });

  it('providerContext 回放的字符串工具结果换成新对象转换，不改动原始 item', () => {
    const rawOutput = { type: 'function_call_output', call_id: 'call_raw', output: 'raw result' };
    const body = withToolOutputs().encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'go' }] },
        { role: 'user', parts: [{ providerContext: { provider: 'openai', format: 'openai-responses', endpoint: 'responses', itemType: 'function_call_output', rawItem: rawOutput } }] },
      ],
    }, true) as any;
    expect(body.input.at(-1)).toEqual({ type: 'function_call_output', call_id: 'call_raw', output: [
      { type: 'input_text', text: 'raw result', prompt_cache_breakpoint: BREAKPOINT },
    ] });
    expect(rawOutput.output).toBe('raw result');
  });

  it('只在 explicit 且模型支持显式缓存、消息断点开启时生效；其他组合与不开启时逐字节一致', () => {
    const same = (withFlag: OpenAIResponsesFormat, without: OpenAIResponsesFormat, request: LLMRequest) =>
      expect(JSON.stringify(withFlag.encodeRequest(request, true))).toBe(JSON.stringify(without.encodeRequest(request, true)));
    for (const request of [TOOL_LOOP_REQUEST, LOOP_TWO_CALLS, HISTORY_REQUEST]) {
      same(new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'key', key: 'k1', breakpoints: { toolOutputs: true } }),
        new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'key', key: 'k1' }), request);
      same(new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'implicit', breakpoints: { toolOutputs: true } }),
        new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'implicit' }), request);
      same(new OpenAIResponsesFormat('gpt-5.6', { enabled: false, mode: 'explicit', breakpoints: { toolOutputs: true } }),
        new OpenAIResponsesFormat('gpt-5.6'), request);
      same(new OpenAIResponsesFormat('gpt-5.5', { enabled: true, mode: 'explicit', breakpoints: { toolOutputs: true } }),
        new OpenAIResponsesFormat('gpt-5.5', { enabled: true, mode: 'explicit' }), request);
      same(new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'explicit', breakpoints: { messages: false, toolOutputs: true } }),
        new OpenAIResponsesFormat('gpt-5.6', { enabled: true, mode: 'explicit', breakpoints: { messages: false } }), request);
    }
  });

  it('compact 请求不注入缓存，工具结果仍是字符串形式', () => {
    const body = withToolOutputs().encodeCompactRequest(TOOL_LOOP_REQUEST) as any;
    expect(body.input.at(-1)).toEqual({ type: 'function_call_output', call_id: 'call_1', output: '{"content":"hello"}' });
    expect(JSON.stringify(body)).not.toContain('prompt_cache');
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

describe('B8 普通响应里的服务端 compaction 输出项解码为 providerContext part 并可原样回放', () => {
  // 依据：https://developers.openai.com/api/docs/guides/compaction
  // “For stateless input-array chaining, append output items as usual.”（服务端 compaction 在
  // 请求设置 context_management 时出现在普通 /responses 输出里）
  const COMPACTION = { type: 'compaction', id: 'cmp_srv_1', encrypted_content: 'gAAAAB_SERVER_COMPACTION' };
  const OUTPUT_WITH_COMPACTION = [COMPACTION, reasoning('rs_2', 'answer'), msg('2', 'Answer one.')];
  const EXPECTED_PART = {
    providerContext: {
      provider: 'openai',
      format: 'openai-responses',
      endpoint: 'responses',
      itemType: 'compaction',
      id: 'cmp_srv_1',
      encryptedContent: 'gAAAAB_SERVER_COMPACTION',
      rawItem: COMPACTION,
    },
  };

  function collectParts(chunks: any[]): any[] {
    return chunks.flatMap(chunk => chunk.partsDelta ?? []).filter(part => 'providerContext' in part);
  }

  it('流式：同一个 compaction item 只发一次（output_item.done），response.completed 不再重复', () => {
    const chunks = decodeAll(new OpenAIResponsesFormat('gpt-5.5', undefined, true), sseEvents('resp_1', OUTPUT_WITH_COMPACTION));
    expect(collectParts(chunks)).toEqual([EXPECTED_PART]);
    const doneIndex = chunks.findIndex(chunk => chunk.partsDelta?.some((part: any) => 'providerContext' in part));
    const firstThoughtIndex = chunks.findIndex(chunk => chunk.partsDelta?.some((part: any) => part.thought));
    expect(doneIndex).toBeGreaterThanOrEqual(0);
    expect(doneIndex).toBeLessThan(firstThoughtIndex);
  });

  it('流式：只在 response.completed 里出现的 compaction item 仍会兜底发出一次', () => {
    const format = new OpenAIResponsesFormat('gpt-5.5');
    const state = format.createStreamState();
    const completed = format.decodeStreamChunk({
      type: 'response.completed',
      response: { id: 'resp_1', output: [COMPACTION, COMPACTION], usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } },
    }, state);
    expect(collectParts([completed])).toEqual([EXPECTED_PART]);
  });

  it('流式：没有 id 的 compaction item 按 encrypted_content 去重', () => {
    const noId = { type: 'compaction', encrypted_content: 'gAAAAB_NOID' };
    const format = new OpenAIResponsesFormat('gpt-5.5');
    const state = format.createStreamState();
    const done = format.decodeStreamChunk({ type: 'response.output_item.done', output_index: 0, item: noId }, state);
    const completed = format.decodeStreamChunk({ type: 'response.completed', response: { id: 'r', output: [noId] } }, state);
    expect(collectParts([done, completed])).toHaveLength(1);
  });

  it('非流式：compaction 按原位置解码为 providerContext part', () => {
    const decoded = new OpenAIResponsesFormat('gpt-5.5').decodeResponse(structuredClone({ id: 'resp_1', output: OUTPUT_WITH_COMPACTION }));
    expect(decoded.content.parts).toEqual([
      EXPECTED_PART,
      { text: 'answer', thought: true, thoughtSignatures: { 'openai-responses': 'gAAAAB_ENC_rs_2' } },
      { text: 'Answer one.' },
    ]);
  });

  it('与 /responses/compact 输出的 providerContext part 是同一表示（仅 endpoint 不同）', () => {
    const format = new OpenAIResponsesFormat('gpt-5.5');
    const compact = format.decodeCompactResponse({ id: 'resp_cmp', object: 'response.compaction', output: [COMPACTION] });
    const compactPart = compact.contents[0].parts[0] as any;
    expect(compactPart).toEqual({
      providerContext: { ...EXPECTED_PART.providerContext, endpoint: 'responses.compact' },
    });
  });

  it('回放：非流式解码的内容回编后 compaction item 原样出现在 input 中且只出现一次', () => {
    const format = new OpenAIResponsesFormat('gpt-5.5');
    const decoded = format.decodeResponse(structuredClone({ id: 'resp_1', output: OUTPUT_WITH_COMPACTION }));
    const body = format.encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'first question' }] },
        decoded.content,
        { role: 'user', parts: [{ text: 'second question' }] },
      ],
    }) as any;
    expect(body.input).toEqual([
      { role: 'user', content: [{ type: 'input_text', text: 'first question' }] },
      COMPACTION,
      { type: 'reasoning', summary: [{ type: 'summary_text', text: 'answer' }], encrypted_content: 'gAAAAB_ENC_rs_2' },
      { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Answer one.' }] },
      { role: 'user', content: [{ type: 'input_text', text: 'second question' }] },
    ]);
  });

  it('回放：流式 partsDelta 累积成的内容回编后 compaction item 只出现一次', () => {
    const chunks = decodeAll(new OpenAIResponsesFormat('gpt-5.5', undefined, true), sseEvents('resp_1', OUTPUT_WITH_COMPACTION));
    const parts = chunks.flatMap(chunk => chunk.partsDelta ?? [])
      .filter(part => 'providerContext' in part || part.thoughtSignatures || part.text);
    const body = new OpenAIResponsesFormat('gpt-5.5').encodeRequest({
      contents: [
        { role: 'user', parts: [{ text: 'first question' }] },
        { role: 'model', parts },
        { role: 'user', parts: [{ text: 'second question' }] },
      ],
    }) as any;
    expect(body.input.filter((item: any) => item.type === 'compaction')).toEqual([COMPACTION]);
    expect(body.input[1]).toEqual(COMPACTION);
  });
});

describe('GPT-5.6 及之后的显式缓存：instructions 转为带断点的 developer 消息', () => {
  // 依据：https://developers.openai.com/api/docs/guides/prompt-caching
  // “Top-level `instructions` cannot contain an explicit breakpoint. To mark reusable developer instructions,
  //  place them in an `input_text` block inside a developer message.”；Summary of model differences：
  //  Explicit breakpoints 只在 “GPT-5.6 and later” 上支持。只认官方精确 id 及其日期快照。
  const OFFICIAL = [
    'gpt-5.6', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna',
    'gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna',
    'gpt-6-sol-2026-05-01', 'GPT-6-Luna', ' gpt-5.6-2026-04-01 ',
  ];
  const NOT_OFFICIAL = [
    'gpt-5.5', 'gpt-5.4', 'gpt-5.6-sol-xhigh', '[az]gpt-5.6-sol', 'gpt-6-sol-xhigh', 'gpt-6-astra-pro',
    'gpt-6', 'gpt-6-luna-mini', 'gpt-6-sol-20260501',
  ];

  it('官方 id 与日期快照：与 Astra 基线相同（只替换 model 字段）', () => {
    for (const model of OFFICIAL) {
      const body = new OpenAIResponsesFormat(model, { enabled: true, mode: 'explicit' }).encodeRequest(HISTORY_REQUEST, true);
      expect(JSON.stringify(body), model).toBe(JSON.stringify({ ...BASELINE_ASTRA_EXPLICIT_BODY, model }));
    }
  });

  it('网关别名与更早的模型：instructions 保持顶层，与修复前基线逐字节一致', () => {
    for (const model of NOT_OFFICIAL) {
      const body = new OpenAIResponsesFormat(model, { enabled: true, mode: 'explicit' }).encodeRequest(HISTORY_REQUEST, true);
      expect(JSON.stringify(body), model).toBe(JSON.stringify({ ...BASELINE_EXPLICIT_BODY, model }));
    }
  });

  it('implicit / key 模式不转换 instructions（与修复前一致）', () => {
    const implicit = new OpenAIResponsesFormat('gpt-6-sol', { enabled: true, mode: 'implicit', key: 'k1' }).encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(implicit)).toBe(JSON.stringify({ ...BASELINE_IMPLICIT_BODY, model: 'gpt-6-sol' }));
    const key = new OpenAIResponsesFormat('gpt-6-luna', { enabled: true, mode: 'key', key: 'k1' }).encodeRequest(HISTORY_REQUEST, true);
    expect(JSON.stringify(key)).toBe(JSON.stringify({ ...BASELINE_KEY_MODE_BODY, model: 'gpt-6-luna' }));
  });
});

describe('GPT-6 Sol / Luna 与 Astra 走同一条原生解码路径', () => {
  // 依据：https://developers.openai.com/api/docs/guides/latest-model（Using GPT-6：GPT-6 家族包括 Astra、Sol、Luna，
  // 异步工具 / 转向 / configuration_update 是家族新特性）。LimCode 扩展对三个模型开放原生能力，
  // 它们的 HTTP/SSE 原生事件与 Astra 相同；WS 构造方式（无原生事件）也与 Astra 相同，不附加 B6 outputItem。
  const FAMILY = ['gpt-6-sol', 'gpt-6-luna', 'gpt-6-sol-2026-05-01', 'gpt-6-luna-2026-06-01'];

  it('HTTP/SSE 原生解码、WS 构造方式与非流式解码均与 Astra 基线逐字节一致', () => {
    for (const model of FAMILY) {
      const http = decodeAll(new OpenAIResponsesFormat(model, undefined, true), sseEvents('resp_a', PHASE_OUTPUT));
      expect(JSON.stringify(http), model).toBe(JSON.stringify(BASELINE_ASTRA_STREAM_PHASE));
      const ws = decodeAll(new OpenAIResponsesFormat(model), sseEvents('resp_w', PHASE_OUTPUT));
      expect(JSON.stringify(ws), model).toBe(JSON.stringify(BASELINE_ASTRA_WS_STREAM_PHASE));
      const decoded = new OpenAIResponsesFormat(model, undefined, true).decodeResponse(structuredClone({ id: 'resp_a', output: PHASE_OUTPUT }));
      expect(JSON.stringify(decoded), model).toBe(JSON.stringify(ASTRA_DECODE_PHASE_WITH_OUTPUT_ITEM));
    }
  });

  it('原生解码附加 nativeEvent 与 completedContents', () => {
    const chunks = decodeAll(new OpenAIResponsesFormat('gpt-6-luna', undefined, true), sseEvents('resp_l', PHASE_OUTPUT));
    expect(chunks[0].nativeEvent).toEqual({ type: 'response.created', responseId: 'resp_l' });
    expect(chunks.at(-1).nativeEvent).toMatchObject({ type: 'response.completed', responseId: 'resp_l' });
    expect(Array.isArray(chunks.at(-1).completedContents)).toBe(true);
  });

  it('网关别名不按 GPT-6 家族处理：没有原生事件，带 phase 的文本仍按 B6 附带 outputItem', () => {
    for (const model of ['gpt-6-sol-xhigh', '[az]gpt-6-luna', 'gpt-6-astra-pro', 'gpt-6']) {
      const chunks = decodeAll(new OpenAIResponsesFormat(model, undefined, true), sseEvents('resp_1', PHASE_OUTPUT));
      expect(chunks.some(chunk => chunk.nativeEvent || chunk.completedContents), model).toBe(false);
      const withoutPhase = chunks.map(({ outputItem: _outputItem, ...rest }) => rest);
      expect(JSON.stringify(withoutPhase), model).toBe(JSON.stringify(BASELINE_STREAM_NO_PHASE));
      expect(chunks.filter(chunk => chunk.outputItem).map(chunk => chunk.outputItem.phase), model).toEqual(['commentary', 'commentary']);
    }
  });
});

describe('非流式：GPT-6 家族的 assistant message 与其他模型一样保留 phase', () => {
  // 依据：OpenAI Responses 参考 https://developers.openai.com/api/reference/resources/responses
  // （phase：“preserve and resend phase on all assistant messages — dropping it can degrade performance”），
  // 以及 https://developers.openai.com/api/docs/guides/reasoning#phase-parameter。
  // 非流式没有原生事件和 completedContents，phase 只能随文本 part 的 outputItem 带出。
  const MODELS = ['gpt-6-sol', 'gpt-6-astra', 'gpt-6-luna-2026-06-01', 'gpt-5.5', 'gpt-5.6-sol'];

  it('decodeResponse 给带 phase 的 message 文本附带 outputItem，与模型无关', () => {
    for (const model of MODELS) {
      for (const native of [true, false]) {
        const decoded = new OpenAIResponsesFormat(model, undefined, native).decodeResponse(structuredClone({ id: 'resp_1', output: PHASE_OUTPUT }));
        expect(decoded.content.parts[1], `${model} native=${native}`).toEqual({
          text: 'Let me check.',
          outputItem: { id: 'msg_1', ordinal: 1, phase: 'commentary' },
        });
      }
    }
  });

  it('解码结果回编时带回原 phase', () => {
    for (const model of MODELS) {
      const format = new OpenAIResponsesFormat(model, undefined, true);
      const decoded = format.decodeResponse(structuredClone({ id: 'resp_1', output: PHASE_OUTPUT }));
      const body = format.encodeRequest({
        contents: [
          { role: 'user', parts: [{ text: 'go' }] },
          decoded.content,
          { role: 'user', parts: [{ functionResponse: { name: 'read_file', response: { ok: 1 }, callId: 'call_1' } }] },
        ],
      }) as any;
      const assistantMessages = body.input.filter((item: any) => item.type === 'message' && item.role === 'assistant');
      expect(assistantMessages, model).toEqual([
        { type: 'message', role: 'assistant', phase: 'commentary', content: [{ type: 'output_text', text: 'Let me check.' }] },
      ]);
    }
  });
});

describe('response.incomplete 在 HTTP/SSE 上按错误上报', () => {
  // response 层（isProviderErrorPayload）把事件名或 type 含 incomplete 的 SSE 事件当作上游错误，
  // 在格式适配器之前就返回 stream_error，所以原生解码里为 response.incomplete 产出 nativeEvent 的分支
  // 走不到，已删除。这与 LimCode WebSocket 原生会话一致：只有 WS 上的 steered 边界不算失败，其余
  // incomplete（如 max_output_tokens）都按失败处理。
  const incompleteEvent = {
    type: 'response.incomplete',
    response: {
      id: 'resp_inc', object: 'response', status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [msg('1', 'Partial answer')],
      usage: { input_tokens: 10, output_tokens: 3, total_tokens: 13 },
    },
  };

  it('原生解码的 GPT-6 家族 SSE 流：response.incomplete 只产生一个 stream_error，没有 response.incomplete 原生事件', async () => {
    const events = sseEvents('resp_inc', [msg('1', 'Partial answer')]).slice(0, -1);
    const body = [...events, incompleteEvent].map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join('');
    const res = new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
    const chunks: any[] = [];
    for await (const chunk of processStreamResponse(res, new OpenAIResponsesFormat('gpt-6-sol', undefined, true))) chunks.push(chunk);
    expect(chunks.filter(chunk => chunk.error).map(chunk => chunk.error.kind)).toEqual(['stream_error']);
    expect(chunks.at(-1).error.event).toBe('response.incomplete');
    expect(chunks.some(chunk => chunk.nativeEvent?.type === 'response.incomplete')).toBe(false);
  });

  it('格式适配器本身不再为 response.incomplete 产出原生事件', () => {
    const format = new OpenAIResponsesFormat('gpt-6-astra', undefined, true);
    expect(format.decodeStreamChunk(structuredClone(incompleteEvent), format.createStreamState())).toEqual({});
  });
});
