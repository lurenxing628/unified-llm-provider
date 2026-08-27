import { describe, expect, it } from 'vitest';

import {
  sanitizeSchemaForClaude,
  sanitizeSchemaForGemini,
  sanitizeSchemaForOpenAI,
} from '../src/llm/formats/schema-sanitizer.js';

/**
 * 回归背景：sanitize 递归时曾把 properties 的 value（{ 属性名: 子schema } 映射）
 * 当作 schema 节点处理，导致属性名为 title / default / $defs 的属性被当作
 * schema 关键字误删，而 required 仍引用它，Gemini 报 400：
 *
 *   GenerateContentRequest.tools[0].function_declarations[N].parameters
 *     ...required[0]: property is not defined
 */
describe('schema-sanitizer 属性名与 schema 关键字区分', () => {

  describe('Gemini', () => {
    it('属性名为 title 且被 required 引用时必须完整保留（复现 update_task_list 400）', () => {
      const schema = {
        type: 'object',
        additionalProperties: false,
        properties: {
          title: { type: 'string', description: 'Task title' },
          status: { type: 'string' },
        },
        required: ['title'],
      };

      const result = sanitizeSchemaForGemini(schema) as any;

      expect(result.properties.title).toEqual({ type: 'string', description: 'Task title' });
      expect(result.required).toEqual(['title']);
      expect(result.additionalProperties).toBeUndefined();
    });

    it('嵌套 items 里的 title 属性必须保留（复现 items[].title 400）', () => {
      const schema = {
        type: 'object',
        properties: {
          items: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                title: { type: 'string' },
                description: { type: 'string' },
              },
              required: ['title'],
            },
          },
        },
        required: ['items'],
      };

      const result = sanitizeSchemaForGemini(schema) as any;
      const element = result.properties.items.items as any;

      expect(element.properties.title).toEqual({ type: 'string' });
      expect(element.required).toEqual(['title']);
    });

    it('submit_plan 形态：taskList 嵌套 items[].title 必须保留', () => {
      const itemSchema = {
        type: 'object',
        properties: { title: { type: 'string' } },
        required: ['title'],
      };
      const schema = {
        type: 'object',
        properties: {
          plan: { type: 'string' },
          taskList: {
            type: 'object',
            properties: {
              mode: { type: 'string', enum: ['rewrite', 'update'] },
              items: { type: 'array', items: itemSchema },
            },
            required: ['mode', 'items'],
          },
        },
        required: ['plan'],
      };

      const result = sanitizeSchemaForGemini(schema) as any;
      const element = result.properties.taskList.properties.items.items as any;

      expect(element.properties.title).toEqual({ type: 'string' });
      expect(element.required).toEqual(['title']);
    });

    it('属性名为 default / const / $defs 时也必须保留', () => {
      const schema = {
        type: 'object',
        properties: {
          default: { type: 'string' },
          const: { type: 'boolean' },
          $defs: { type: 'number' },
        },
        required: ['default', 'const', '$defs'],
      };

      const result = sanitizeSchemaForGemini(schema) as any;

      expect(result.properties.default).toEqual({ type: 'string' });
      expect(result.properties.const).toEqual({ type: 'boolean' });
      expect(result.properties.$defs).toEqual({ type: 'number' });
      expect(result.required).toEqual(['default', 'const', '$defs']);
    });

    it('schema 节点层级的 title / default / const 关键字仍然删除', () => {
      const schema = {
        type: 'object',
        title: '任务清单',
        default: { title: 'x' },
        const: 'fixed',
        properties: {
          name: { type: 'string', title: '名称注解', default: 'abc' },
        },
      };

      const result = sanitizeSchemaForGemini(schema) as any;

      expect(result.title).toBeUndefined();
      expect(result.default).toBeUndefined();
      expect(result.const).toBeUndefined();
      // 属性的子 schema 里的注解关键字同样删除
      expect(result.properties.name).toEqual({ type: 'string' });
    });

    it('原有降级行为不回归：additionalProperties 删除、enum 转字符串、anyOf 展平', () => {
      const schema = {
        type: 'object',
        additionalProperties: false,
        properties: {
          level: { type: 'integer', enum: [1, 2, 3] },
          choice: {
            anyOf: [
              { type: 'string' },
              { type: 'number' },
            ],
          },
          nested: {
            type: 'object',
            properties: {
              flag: { type: 'boolean' },
            },
            required: ['flag'],
            additionalProperties: true,
          },
        },
        required: ['level'],
      };

      const result = sanitizeSchemaForGemini(schema) as any;

      expect(result.additionalProperties).toBeUndefined();
      expect(result.properties.level).toEqual({ type: 'string', enum: ['1', '2', '3'] });
      // anyOf 是唯一字段 → 取第一个分支展开
      expect(result.properties.choice).toEqual({ type: 'string' });
      expect(result.properties.nested.additionalProperties).toBeUndefined();
      expect(result.properties.nested.required).toEqual(['flag']);
    });
  });

  describe('OpenAI', () => {
    it('属性名为 definitions / $schema 时必须保留', () => {
      const schema = {
        type: 'object',
        definitions: { inner: { type: 'string' } },
        properties: {
          definitions: { type: 'string' },
          $schema: { type: 'string' },
        },
        required: ['definitions', '$schema'],
      };

      const result = sanitizeSchemaForOpenAI(schema) as any;

      expect(result.definitions).toBeUndefined();
      expect(result.properties.definitions).toEqual({ type: 'string' });
      expect(result.properties.$schema).toEqual({ type: 'string' });
      expect(result.required).toEqual(['definitions', '$schema']);
    });
  });

  describe('Claude', () => {
    it('属性名为 definitions 时必须保留，顶层 oneOf 仍取第一个分支', () => {
      const schema = {
        oneOf: [
          { type: 'object', properties: { kind: { type: 'string' } }, required: ['kind'] },
          { type: 'null' },
        ],
      };

      const result = sanitizeSchemaForClaude(schema) as any;

      expect(result.type).toBe('object');
      expect(result.properties.kind).toEqual({ type: 'string' });
      expect(result.required).toEqual(['kind']);
    });

    it('嵌套 properties 中的 definitions 属性名必须保留', () => {
      const schema = {
        type: 'object',
        properties: {
          definitions: { type: 'string' },
          items: {
            type: 'array',
            items: {
              type: 'object',
              properties: { title: { type: 'string' } },
              required: ['title'],
            },
          },
        },
        required: ['definitions'],
      };

      const result = sanitizeSchemaForClaude(schema) as any;

      expect(result.properties.definitions).toEqual({ type: 'string' });
      expect(result.required).toEqual(['definitions']);
      expect((result.properties.items.items as any).properties.title).toEqual({ type: 'string' });
    });
  });
});
