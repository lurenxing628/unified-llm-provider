import { describe, expect, it } from 'vitest';

import { OpenAICompatibleFormat, decodeRequestFromFormat } from '../src/index.js';
import type { LLMRequest } from '../src/index.js';
import { readFixture } from './openai-compatible-helpers.js';

/**
 * “原来正确的请求逐字节不变”回归：fixtures/openai-compatible/requests.json 里的统一请求，
 * 分别按 openai-compatible / deepseek、流式 / 非流式编码，与修复前（16b50ab）的编码结果
 * （baseline-16b50ab-requests.json）逐字节比较。
 *
 * 只有下面登记的组合允许变化，且每一项都有单独的测试说明新行为与依据。
 */
const INTENTIONAL_CHANGES: Record<string, string> = {
  // A6：OpenAI 文档 “For tool messages, only type `text` is supported”，图片移到 tool 消息之后的 user 消息。
  'tool-result-media|openai-compatible|nostream': 'A6',
  'tool-result-media|openai-compatible|stream': 'A6',
  // A7：DeepSeek 文档把 medium 映射为 high，现在显式发送 thinking + reasoning_effort。
  'plain-chat|deepseek|nostream': 'A7',
  'plain-chat|deepseek|stream': 'A7',
};

const requests = JSON.parse(readFixture('requests.json')) as Record<string, LLMRequest>;
const baseline = JSON.parse(readFixture('baseline-16b50ab-requests.json')) as Record<string, unknown>;

function encode(name: string, kind: 'openai-compatible' | 'deepseek', stream: boolean): unknown {
  const normalized = decodeRequestFromFormat(requests[name], { format: 'unified' });
  return new OpenAICompatibleFormat('test-model', kind).encodeRequest(normalized, stream);
}

const cases = Object.keys(baseline).map(key => {
  const [name, kind, stream] = key.split('|') as [string, 'openai-compatible' | 'deepseek', string];
  return { key, name, kind, stream: stream === 'stream' };
});

describe('OpenAI 兼容请求编码回归', () => {
  it.each(cases.filter(item => !INTENTIONAL_CHANGES[item.key]))('原行为不变：$key 与修复前逐字节一致', ({ key, name, kind, stream }) => {
    expect(JSON.stringify(encode(name, kind, stream))).toBe(JSON.stringify(baseline[key]));
  });

  it.each(cases.filter(item => INTENTIONAL_CHANGES[item.key]))('$key 按修复项有意变化', ({ key, name, kind, stream }) => {
    expect(JSON.stringify(encode(name, kind, stream))).not.toBe(JSON.stringify(baseline[key]));
  });
});

describe('有意变化的请求只改动对应字段', () => {
  it('A7 plain-chat|deepseek：medium 现在发送 thinking=enabled + reasoning_effort=high，其余字段与修复前一致', () => {
    for (const stream of [false, true]) {
      const key = `plain-chat|deepseek|${stream ? 'stream' : 'nostream'}`;
      const body = encode('plain-chat', 'deepseek', stream) as Record<string, unknown>;
      const before = baseline[key] as Record<string, unknown>;
      expect(before.thinking).toBeUndefined();
      expect(before.reasoning_effort).toBeUndefined();
      const { thinking, reasoning_effort, ...rest } = body;
      expect(thinking).toEqual({ type: 'enabled' });
      expect(reasoning_effort).toBe('high');
      expect(JSON.stringify(rest)).toBe(JSON.stringify(before));
    }
  });
});
