/**
 * 工具 Schema 降级处理
 *
 * MCP 工具返回的 inputSchema 是完整的 JSON Schema，
 * 但各 LLM provider 对 function calling 的 schema 支持程度不同。
 *
 * 此模块提供按 provider 降级的函数，在保留尽可能多信息的前提下，
 * 确保 schema 能被对应 provider 接受。
 *
 * 核心原则：递归时必须区分两种对象
 *
 *   1. schema 节点（type / properties / required / items / ...）
 *      —— 只有在这一层才能按 schema 关键字过滤（删 title、default 等）
 *
 *   2. properties / patternProperties 的 value（{ 属性名: 子schema } 映射表）
 *      —— 这里的 key 是属性名，属性名可能恰好叫 title / default / const / $defs，
 *         绝不能按 schema 关键字删除。
 *         否则 properties.title 被删掉后 required 仍引用它，
 *         Gemini 会报 "required[0]: property is not defined" 400。
 *
 * 已知限制（基于 2026-03 实测 + 社区 issue 调研）：
 *
 *   Gemini:
 *     - enum 值必须是字符串（数字 enum 报 TYPE_STRING 错误）
 *     - 不支持 additionalProperties（function declaration 中）
 *     - anyOf 不能与其他字段混用（必须是属性中的唯一字段）
 *     - 不支持 $ref（已由 dereference 层处理）
 *     - 不支持 title、default、const、not、if/then/else
 *     - 嵌套深度有限制（未文档化）
 *     - schema 复杂度有隐式上限（"too many states"）
 *
 *   OpenAI:
 *     - 函数参数就是 JSON Schema；Structured Outputs 文档列出的支持类型包含 Integer / Number / Enum，
 *       并支持 definitions（`$defs` + `$ref`）与递归 schema
 *       （https://developers.openai.com/api/docs/guides/structured-outputs “Supported schemas”）。
 *       因此数字 enum 原样保留（以前把 `{type:'integer', enum:[1,2]}` 改成字符串 enum，类型与取值自相矛盾）。
 *     - 本地 `$ref` 先展开再清洗；以前直接删除 `$defs` 却保留 `$ref`，留下悬空引用。
 *
 *   Claude:
 *     - 不支持顶层 oneOf/allOf/anyOf（嵌套内可以）
 *     - 其余基本完整支持
 */

// ===================== 通用辅助 =====================

/**
 * 处理 properties / patternProperties 的 value：{ 属性名: 子schema } 映射表。
 * 属性名原样保留（包括与 schema 关键字同名的 title / default / const 等），
 * 只对每个属性的 schema 值递归降级。
 */
function sanitizePropertyMap(
  value: unknown,
  sanitizeChild: (schema: unknown) => unknown,
): unknown {
  if (value === null || value === undefined || typeof value !== 'object' || Array.isArray(value)) {
    return value;
  }
  const result: Record<string, unknown> = {};
  for (const [name, childSchema] of Object.entries(value as Record<string, unknown>)) {
    result[name] = sanitizeChild(childSchema);
  }
  return result;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// ===================== Gemini =====================

/** Gemini 不支持的 schema 关键字（仅在 schema 节点层级删除，不影响属性名） */
const GEMINI_DROPPED_KEYWORDS = [
  'title', 'default', 'const', '$defs', 'definitions', '$schema',
  'not', 'if', 'then', 'else', 'prefixItems',
];

/**
 * 为 Gemini 降级 schema。最严格的处理：
 *   1. enum 数字值 → 字符串
 *   2. 删除 additionalProperties
 *   3. anyOf/oneOf/allOf → 尝试展平或取第一个分支
 *   4. 删除 title、default、const、$defs、definitions、$schema（仅 schema 关键字层级）
 *   5. 递归处理所有嵌套
 */
export function sanitizeSchemaForGemini(schema: unknown): unknown {
  if (schema === null || schema === undefined || typeof schema !== 'object') {
    return schema;
  }

  if (Array.isArray(schema)) {
    return schema.map(sanitizeSchemaForGemini);
  }

  const obj = schema as Record<string, unknown>;
  const result: Record<string, unknown> = {};
  let hasStringifiedEnum = false;

  for (const [key, value] of Object.entries(obj)) {
    // 删除 Gemini 不支持的 schema 关键字
    // $schema: MCP 工具的 inputSchema 常携带此字段，Gemini API 不识别会直接 400
    // 注意：这里删的是 schema 节点上的关键字；properties 里的同名属性由
    // sanitizePropertyMap 处理，不会被误删。
    if (GEMINI_DROPPED_KEYWORDS.includes(key)) {
      continue;
    }

    // 删除 additionalProperties
    if (key === 'additionalProperties') {
      continue;
    }

    // properties / patternProperties：key 是属性名，value 是 { 属性名: 子schema } 映射
    if (key === 'properties' || key === 'patternProperties') {
      result[key] = sanitizePropertyMap(value, sanitizeSchemaForGemini);
      continue;
    }

    // anyOf/oneOf/allOf: 尝试展平或取第一个
    if ((key === 'anyOf' || key === 'oneOf' || key === 'allOf') && Array.isArray(value)) {
      // 如果是属性层级的 anyOf（与 type/properties 等混用），跳过 anyOf
      // 如果是唯一字段，取第一个分支展开
      const otherKeys = Object.keys(obj).filter(k =>
        k !== key && !GEMINI_DROPPED_KEYWORDS.includes(k) && k !== 'additionalProperties'
      );
      if (otherKeys.length === 0 && value.length > 0) {
        // anyOf 是唯一有意义的字段 → 取第一个分支展开
        const first = sanitizeSchemaForGemini(value[0]);
        if (isPlainObject(first)) {
          Object.assign(result, first);
        }
        continue;
      }
      // 与其他字段混用 → 直接丢弃 anyOf
      continue;
    }

    // enum: 所有值转为字符串
    if (key === 'enum' && Array.isArray(value)) {
      result[key] = value.map(v => String(v));
      hasStringifiedEnum = true;
      continue;
    }

    // 递归处理嵌套 schema 节点
    result[key] = sanitizeSchemaForGemini(value);
  }

  // enum 值已转为字符串，type 需要同步改为 string
  // 放在循环结束后处理，避免被后续的 type 字段赋值覆盖
  if (hasStringifiedEnum && (result.type === 'integer' || result.type === 'number')) {
    result.type = 'string';
  }

  return result;
}

// ===================== 本地 $ref 展开 =====================

/** 这些关键字的值是实例数据而不是 schema，里面的 `$ref` 字样不是引用。 */
const SCHEMA_DATA_KEYWORDS = new Set(['enum', 'const', 'default', 'examples', 'example']);

/** 这些关键字的值是 { 名称: 子schema } 映射表，名称可能与 schema 关键字同名。 */
const SCHEMA_MAP_KEYWORDS = new Set(['properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas']);

/** 防止病态 schema（大量交叉引用）展开后体积爆炸；超过后保留引用和定义，不再展开。 */
const MAX_REF_EXPANSIONS = 1000;

function decodeJsonPointerSegment(segment: string): string {
  let decoded = segment;
  try {
    decoded = decodeURIComponent(segment);
  } catch {
    // 不是合法的百分号编码时按原文处理
  }
  return decoded.replace(/~1/g, '/').replace(/~0/g, '~');
}

function escapeJsonPointerSegment(segment: string): string {
  return segment.replace(/~/g, '~0').replace(/\//g, '~1');
}

/** 解析文档内引用：`#` 或 `#/a/b`（JSON Pointer，RFC 6901）。 */
function resolveLocalRef(root: unknown, ref: string): { found: true; value: unknown } | { found: false } {
  if (ref === '#') return { found: true, value: root };
  if (!ref.startsWith('#/')) return { found: false };
  let current: unknown = root;
  for (const rawSegment of ref.slice(2).split('/')) {
    const segment = decodeJsonPointerSegment(rawSegment);
    if (Array.isArray(current) && /^\d+$/.test(segment) && Number(segment) < current.length) {
      current = current[Number(segment)];
    } else if (isPlainObject(current) && Object.prototype.hasOwnProperty.call(current, segment)) {
      current = current[segment];
    } else {
      return { found: false };
    }
  }
  return { found: true, value: current };
}

/** 引用的规范形式（段先解码再按 RFC 6901 转义），用来和节点位置比较。 */
function canonicalLocalRef(ref: string): string {
  if (ref === '#') return '#';
  return `#/${ref.slice(2).split('/').map(segment => escapeJsonPointerSegment(decodeJsonPointerSegment(segment))).join('/')}`;
}

function isSameOrAncestorPointer(ref: string, location: string): boolean {
  return ref === '#' || location === ref || location.startsWith(`${ref}/`);
}

export interface DereferencedSchema {
  schema: unknown;
  /**
   * 有引用因递归、目标不是对象或展开次数超限而保留，且它指向 `$defs` / `definitions` 内部。
   * 此时这些定义必须随 schema 一起保留，否则会变成悬空引用。
   */
  requiresDefinitions: boolean;
}

/**
 * 展开 schema 内的本地 `$ref`（`#`、`#/$defs/...`、`#/definitions/...` 等 JSON Pointer）。
 *
 * - 非递归引用：用目标 schema 替换引用节点；引用节点上的其他关键字（如 description）覆盖目标同名字段。
 * - 递归引用（目标是当前节点自身或祖先，或已在当前展开链上）：保留 `$ref`；指向定义表内部时
 *   通过 requiresDefinitions 告知调用方保留定义。根递归 `$ref: "#"` 因此原样保留。
 * - 找不到目标或非本地引用：原样保留，不做猜测。
 * - 没有任何本地 `$ref` 时返回原对象本身，清洗结果与以前完全一致。
 */
export function dereferenceLocalSchemaRefs(schema: unknown): DereferencedSchema {
  if (!isPlainObject(schema)) return { schema, requiresDefinitions: false };
  const root = schema;
  let sawRef = false;
  let requiresDefinitions = false;
  let expansions = 0;

  const child = (location: string, key: string | number) => `${location}/${escapeJsonPointerSegment(String(key))}`;

  const expandMap = (value: unknown, stack: readonly string[], location: string): unknown => {
    if (!isPlainObject(value)) return expand(value, stack, location);
    const result: Record<string, unknown> = {};
    for (const [name, childSchema] of Object.entries(value)) result[name] = expand(childSchema, stack, child(location, name));
    return result;
  };

  const expandObject = (node: Record<string, unknown>, stack: readonly string[], location: string): Record<string, unknown> => {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(node)) {
      if (SCHEMA_DATA_KEYWORDS.has(key)) {
        result[key] = value;
      } else if (SCHEMA_MAP_KEYWORDS.has(key)) {
        // 定义表本身不展开：它们只在 requiresDefinitions 时随 schema 保留，保留时引用目标仍然有效。
        result[key] = key === '$defs' || key === 'definitions' ? value : expandMap(value, stack, child(location, key));
      } else {
        result[key] = expand(value, stack, child(location, key));
      }
    }
    return result;
  };

  const expand = (node: unknown, stack: readonly string[], location: string): unknown => {
    if (Array.isArray(node)) return node.map((item, index) => expand(item, stack, child(location, index)));
    if (!isPlainObject(node)) return node;
    const ref = node.$ref;
    if (typeof ref !== 'string' || !ref.startsWith('#')) return expandObject(node, stack, location);

    sawRef = true;
    const { $ref: _ref, ...siblings } = node;
    const target = resolveLocalRef(root, ref);
    if (!target.found) return expandObject(node, stack, location);
    const canonical = canonicalLocalRef(ref);
    const recursive = stack.includes(canonical) || isSameOrAncestorPointer(canonical, location);
    if (recursive || !isPlainObject(target.value) || expansions >= MAX_REF_EXPANSIONS) {
      if (/\/(?:\$defs|definitions)(?:\/|$)/.test(canonical)) requiresDefinitions = true;
      return expandObject(node, stack, location);
    }
    expansions += 1;
    const expandedTarget = expand(target.value, [...stack, canonical], canonical) as Record<string, unknown>;
    return { ...expandedTarget, ...expandObject(siblings, stack, location) };
  };

  const expanded = expand(root, [], '#');
  return sawRef ? { schema: expanded, requiresDefinitions } : { schema, requiresDefinitions: false };
}

// ===================== OpenAI =====================

/** OpenAI 需要删除的 schema 关键字（仅 schema 节点层级） */
const OPENAI_DROPPED_KEYWORDS = ['$defs', 'definitions', '$schema'];

/**
 * 为 OpenAI 降级 schema：
 *   1. 先展开本地 `$ref`（防循环）；
 *   2. 删除 `$defs` / `definitions`（已展开）与 `$schema`；仍有递归引用时保留定义，保证引用可解析；
 *   3. enum 原样保留（OpenAI 支持非字符串 enum），其余关键字保留。
 */
export function sanitizeSchemaForOpenAI(schema: unknown): unknown {
  const dereferenced = dereferenceLocalSchemaRefs(schema);
  return sanitizeOpenAISchemaNode(dereferenced.schema, dereferenced.requiresDefinitions);
}

function sanitizeOpenAISchemaNode(schema: unknown, keepDefinitions: boolean): unknown {
  if (schema === null || schema === undefined || typeof schema !== 'object') {
    return schema;
  }

  if (Array.isArray(schema)) {
    return schema.map(item => sanitizeOpenAISchemaNode(item, keepDefinitions));
  }

  const obj = schema as Record<string, unknown>;
  const result: Record<string, unknown> = {};
  const sanitizeChild = (child: unknown) => sanitizeOpenAISchemaNode(child, keepDefinitions);

  for (const [key, value] of Object.entries(obj)) {
    // 仍有保留的 $ref 时，定义表必须留下，否则引用悬空
    if (keepDefinitions && (key === '$defs' || key === 'definitions')) {
      result[key] = sanitizePropertyMap(value, sanitizeChild);
      continue;
    }

    // 删除已展开的残留（schema 关键字层级；properties 里的同名属性不受影响）
    // $schema: MCP 工具常携带，对 LLM API 无意义
    if (OPENAI_DROPPED_KEYWORDS.includes(key)) continue;

    // properties / patternProperties：key 是属性名，原样保留
    if (key === 'properties' || key === 'patternProperties') {
      result[key] = sanitizePropertyMap(value, sanitizeChild);
      continue;
    }

    // enum / const / default 等是实例数据，原样保留（不再把数字 enum 转成字符串）
    if (SCHEMA_DATA_KEYWORDS.has(key)) {
      result[key] = value;
      continue;
    }

    result[key] = sanitizeChild(value);
  }

  return result;
}

// ===================== Claude =====================

/** Claude 需要删除的 schema 关键字（仅 schema 节点层级） */
const CLAUDE_DROPPED_KEYWORDS = ['$defs', 'definitions', '$schema'];

/**
 * 为 Claude 降级 schema。中等处理：
 *   1. 顶层 oneOf/allOf/anyOf → 取第一个分支（嵌套内保留）
 *   2. 删除 $defs/definitions
 *   3. enum 数字值 → 字符串
 */
export function sanitizeSchemaForClaude(schema: unknown, isTopLevel = true): unknown {
  if (schema === null || schema === undefined || typeof schema !== 'object') {
    return schema;
  }

  if (Array.isArray(schema)) {
    return schema.map(v => sanitizeSchemaForClaude(v, false));
  }

  const obj = schema as Record<string, unknown>;
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    // 删除已展开的残留（schema 关键字层级；properties 里的同名属性不受影响）
    // $schema: MCP 工具常携带，对 LLM API 无意义
    if (CLAUDE_DROPPED_KEYWORDS.includes(key)) continue;

    // properties / patternProperties：key 是属性名，原样保留
    if (key === 'properties' || key === 'patternProperties') {
      result[key] = sanitizePropertyMap(value, v => sanitizeSchemaForClaude(v, false));
      continue;
    }

    // 顶层的 anyOf/oneOf/allOf → 取第一个分支
    if (isTopLevel && (key === 'anyOf' || key === 'oneOf' || key === 'allOf') && Array.isArray(value) && value.length > 0) {
      const first = sanitizeSchemaForClaude(value[0], false);
      if (isPlainObject(first)) {
        Object.assign(result, first);
      }
      continue;
    }

    // enum: 统一转字符串
    if (key === 'enum' && Array.isArray(value)) {
      result[key] = value.map(v => String(v));
      continue;
    }

    result[key] = sanitizeSchemaForClaude(value, false);
  }

  return result;
}
