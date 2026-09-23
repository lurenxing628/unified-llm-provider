import { describe, expect, it } from 'vitest';

import {
  createOpenAICompatibleProvider,
  createOpenAIResponsesProvider,
  dereferenceLocalSchemaRefs,
  sanitizeSchemaForClaude,
  sanitizeSchemaForGemini,
  sanitizeSchemaForOpenAI,
} from '../src/index.js';
import { readFixture } from './openai-compatible-helpers.js';

/**
 * A8：OpenAI schema 清洗先展开本地 $ref 再删 $defs；数字 enum 不再转字符串。
 * 依据：https://developers.openai.com/api/docs/guides/structured-outputs “Supported schemas”
 * ——支持类型含 Integer / Number / Enum（示例里 enum 含 null，枚举限制也区分“string values”），
 * “Definitions are supported”（`$defs` + `$ref`）、“Recursive schemas are supported”（`$ref: "#"`）。
 * 函数调用指南：parameters 就是 JSON Schema（https://developers.openai.com/api/docs/guides/function-calling）。
 */

function collectRefs(node: unknown, refs: string[] = []): string[] {
  if (Array.isArray(node)) node.forEach(item => collectRefs(item, refs));
  else if (node && typeof node === 'object') {
    for (const [key, value] of Object.entries(node)) {
      if (key === '$ref' && typeof value === 'string') refs.push(value);
      else collectRefs(value, refs);
    }
  }
  return refs;
}

function resolvePointer(root: unknown, ref: string): unknown {
  if (ref === '#') return root;
  return ref.slice(2).split('/').reduce<any>((current, segment) => current?.[segment.replace(/~1/g, '/').replace(/~0/g, '~')], root);
}

function expectNoDanglingRefs(schema: unknown): void {
  for (const ref of collectRefs(schema)) {
    expect(resolvePointer(schema, ref), `dangling ${ref}`).toBeDefined();
  }
}

describe('sanitizeSchemaForOpenAI：本地 $ref 展开（A8）', () => {
  it('原行为不变：扩展内置工具与常见 MCP schema（无 $ref、字符串 enum）的清洗结果与修复前逐字节一致', () => {
    const inputs = JSON.parse(readFixture('schemas.json')) as Record<string, unknown>;
    const baseline = JSON.parse(readFixture('baseline-16b50ab-schemas.json')) as Record<string, unknown>;
    expect(Object.keys(inputs).length).toBeGreaterThan(20);
    for (const [name, schema] of Object.entries(inputs)) {
      expect(JSON.stringify(sanitizeSchemaForOpenAI(schema)), name).toBe(JSON.stringify(baseline[name]));
    }
  });

  it('展开 #/$defs/... 引用并删除 $defs，不留悬空引用；引用节点上的 description 覆盖目标', () => {
    const schema = {
      $schema: 'https://json-schema.org/draft/2020-12/schema',
      type: 'object',
      properties: {
        steps: { type: 'array', items: { $ref: '#/$defs/step' } },
        first: { $ref: '#/$defs/step', description: 'The first step' },
      },
      required: ['steps'],
      $defs: {
        step: {
          type: 'object',
          description: 'A step',
          properties: { explanation: { type: 'string' }, output: { $ref: '#/$defs/output' } },
          required: ['explanation', 'output'],
          additionalProperties: false,
        },
        output: { type: 'string', enum: ['ok', 'fail'] },
      },
    };
    const step = {
      type: 'object',
      description: 'A step',
      properties: { explanation: { type: 'string' }, output: { type: 'string', enum: ['ok', 'fail'] } },
      required: ['explanation', 'output'],
      additionalProperties: false,
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual({
      type: 'object',
      properties: {
        steps: { type: 'array', items: step },
        first: { ...step, description: 'The first step' },
      },
      required: ['steps'],
    });
  });

  it('展开 #/definitions/... 引用（draft-07 风格）', () => {
    const schema = {
      type: 'object',
      properties: { user: { $ref: '#/definitions/User' } },
      definitions: { User: { type: 'object', properties: { name: { type: 'string' } } } },
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual({
      type: 'object',
      properties: { user: { type: 'object', properties: { name: { type: 'string' } } } },
    });
  });

  it('递归引用不会无限展开：保留 $ref 与 $defs，且所有引用都能解析', () => {
    const schema = {
      $schema: 'http://json-schema.org/draft-07/schema#',
      type: 'object',
      properties: { root: { $ref: '#/$defs/node' } },
      $defs: {
        node: {
          type: 'object',
          properties: { name: { type: 'string' }, children: { type: 'array', items: { $ref: '#/$defs/node' } } },
        },
      },
    };
    const result = sanitizeSchemaForOpenAI(schema) as any;
    expect(result.$schema).toBeUndefined();
    expect(result.properties.root.properties.children.items).toEqual({ $ref: '#/$defs/node' });
    expect(result.$defs.node).toEqual(schema.$defs.node);
    expectNoDanglingRefs(result);
  });

  it('根递归 `$ref: "#"`（OpenAI 文档示例）原样保留', () => {
    const schema = {
      type: 'object',
      properties: { label: { type: 'string' }, children: { type: 'array', items: { $ref: '#' } } },
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual(schema);
  });

  it('根递归与普通定义引用并存：普通引用展开后不再保留 $defs', () => {
    const schema = {
      type: 'object',
      properties: { tag: { $ref: '#/$defs/tag' }, children: { type: 'array', items: { $ref: '#' } } },
      $defs: { tag: { type: 'string' } },
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual({
      type: 'object',
      properties: { tag: { type: 'string' }, children: { type: 'array', items: { $ref: '#' } } },
    });
  });

  it('找不到目标的引用与非本地引用原样保留（不做猜测）', () => {
    const schema = {
      type: 'object',
      properties: { a: { $ref: '#/$defs/missing' }, b: { $ref: 'https://example.com/schema.json' } },
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual(schema);
  });

  it('JSON Pointer 转义（~1、~0、百分号编码）能正确解析', () => {
    const schema = {
      type: 'object',
      properties: { a: { $ref: '#/$defs/a~1b' }, c: { $ref: '#/$defs/c~0d' }, e: { $ref: '#/$defs/e%20f' } },
      $defs: { 'a/b': { type: 'string' }, 'c~d': { type: 'number' }, 'e f': { type: 'boolean' } },
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual({
      type: 'object',
      properties: { a: { type: 'string' }, c: { type: 'number' }, e: { type: 'boolean' } },
    });
  });

  it('属性名叫 $ref、default 等以及 default/enum 里的数据不当作引用', () => {
    const schema = {
      type: 'object',
      properties: {
        $ref: { type: 'string' },
        default: { $ref: '#/$defs/s' },
        config: { type: 'object', default: { $ref: '#/$defs/s' } },
      },
      $defs: { s: { type: 'string' } },
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual({
      type: 'object',
      properties: {
        $ref: { type: 'string' },
        default: { type: 'string' },
        config: { type: 'object', default: { $ref: '#/$defs/s' } },
      },
    });
  });

  it('大量交叉引用超过展开上限时停止展开，保留定义且没有悬空引用', () => {
    const $defs: Record<string, unknown> = { d0: { type: 'string' } };
    for (let i = 1; i <= 14; i += 1) {
      $defs[`d${i}`] = { type: 'object', properties: { left: { $ref: `#/$defs/d${i - 1}` }, right: { $ref: `#/$defs/d${i - 1}` } } };
    }
    const schema = { type: 'object', properties: { root: { $ref: '#/$defs/d14' } }, $defs };
    const { requiresDefinitions } = dereferenceLocalSchemaRefs(schema);
    expect(requiresDefinitions).toBe(true);
    const result = sanitizeSchemaForOpenAI(schema) as any;
    expect(result.$defs).toBeDefined();
    expectNoDanglingRefs(result);
  });

  it('没有 $ref 时 dereferenceLocalSchemaRefs 返回原对象', () => {
    const schema = { type: 'object', properties: { a: { type: 'string' } } };
    expect(dereferenceLocalSchemaRefs(schema).schema).toBe(schema);
  });
});

describe('sanitizeSchemaForOpenAI：enum 原样保留（A8）', () => {
  it('数字 / 布尔 / null enum 不再转字符串，类型与取值保持一致', () => {
    const schema = {
      type: 'object',
      properties: {
        level: { type: 'integer', enum: [1, 2, 3] },
        ratio: { type: 'number', enum: [0.5, 1.5] },
        flag: { type: 'boolean', enum: [true] },
        category: { type: ['string', 'null'], enum: ['a', 'b', null] },
      },
    };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual(schema);
  });

  it('原行为不变：字符串 enum 不变', () => {
    const schema = { type: 'string', enum: ['celsius', 'fahrenheit'] };
    expect(sanitizeSchemaForOpenAI(schema)).toEqual(schema);
  });

  it('Gemini / Claude 清洗保持现状（数字 enum 仍转字符串）', () => {
    expect(sanitizeSchemaForGemini({ type: 'integer', enum: [1, 2] })).toEqual({ type: 'string', enum: ['1', '2'] });
    expect(sanitizeSchemaForClaude({ type: 'object', properties: { level: { type: 'integer', enum: [1, 2] } } }))
      .toEqual({ type: 'object', properties: { level: { type: 'integer', enum: ['1', '2'] } } });
  });
});

describe('OpenAI 系请求里的工具参数（A8）', () => {
  const parameters = {
    type: 'object',
    properties: { item: { $ref: '#/$defs/item' }, count: { type: 'integer', enum: [1, 5, 10] } },
    $defs: { item: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] } },
  };
  const expected = {
    type: 'object',
    properties: { item: { type: 'object', properties: { id: { type: 'string' } }, required: ['id'] }, count: { type: 'integer', enum: [1, 5, 10] } },
  };
  const request = {
    contents: [{ role: 'user' as const, parts: [{ text: 'hi' }] }],
    tools: [{ functionDeclarations: [{ name: 'pick', description: 'Pick an item', parameters }] }],
  };

  it('openai-compatible：展开引用并保留数字 enum', async () => {
    const provider = createOpenAICompatibleProvider({ provider: 'openai-compatible', model: 'gpt-test', apiKey: 'sk-test', baseUrl: 'https://api.openai.test/v1' });
    const dry = await provider.dryRun(request, { stream: false });
    expect((dry.body as any).tools[0].function.parameters).toEqual(expected);
  });

  it('openai-responses 共用同一清洗函数，结果相同', async () => {
    const provider = createOpenAIResponsesProvider({ provider: 'openai-responses', model: 'gpt-test', apiKey: 'sk-test', baseUrl: 'https://api.openai.test/v1' });
    const dry = await provider.dryRun(request, { stream: false });
    expect((dry.body as any).tools[0].parameters).toEqual(expected);
  });
});
