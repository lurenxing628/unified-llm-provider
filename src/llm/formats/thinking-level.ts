export type NormalizedThinkingLevel = 'not-set' | 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export type GeminiThinkingLevel = Extract<NormalizedThinkingLevel, 'minimal' | 'low' | 'medium' | 'high'>;
export type ClaudeThinkingLevel = Extract<NormalizedThinkingLevel, 'none' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'>;
export type OpenAIThinkingLevel = Extract<NormalizedThinkingLevel, 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'>;
export type DeepSeekThinkingLevel = Extract<NormalizedThinkingLevel, 'none' | 'low' | 'high' | 'max'>;

const NON_SET_LEVELS = new Set(['not-set', 'non-set', 'not_set', 'non_set', 'notset', 'nonset', 'unset']);

export function normalizeThinkingLevel(value: unknown): NormalizedThinkingLevel | undefined {
  if (typeof value !== 'string') return undefined;
  const normalized = value.trim().toLowerCase().replace(/\s+/g, '-');
  if (!normalized) return undefined;
  if (NON_SET_LEVELS.has(normalized)) return 'not-set';

  switch (normalized.replace(/_/g, '-')) {
    case 'none':
    case 'minimal':
    case 'low':
    case 'medium':
    case 'high':
    case 'max':
      return normalized as NormalizedThinkingLevel;
    case 'xhigh':
    case 'x-high':
    case 'extra-high':
      return 'xhigh';
    default:
      return undefined;
  }
}

export function mapGeminiThinkingLevel(value: unknown): GeminiThinkingLevel | undefined {
  const level = normalizeThinkingLevel(value);
  switch (level) {
    case 'minimal':
    case 'low':
    case 'medium':
    case 'high':
      return level;
    default:
      return undefined;
  }
}

export function mapClaudeThinkingLevel(value: unknown): ClaudeThinkingLevel | undefined {
  const level = normalizeThinkingLevel(value);
  switch (level) {
    case 'none':
    case 'low':
    case 'medium':
    case 'high':
    case 'xhigh':
    case 'max':
      return level;
    default:
      return undefined;
  }
}

export function mapOpenAIThinkingLevel(value: unknown): OpenAIThinkingLevel | undefined {
  const level = normalizeThinkingLevel(value);
  switch (level) {
    case 'none':
    case 'minimal':
    case 'low':
    case 'medium':
    case 'high':
    case 'xhigh':
    case 'max':
      return level;
    default:
      return undefined;
  }
}

export const mapOpenAIResponsesThinkingLevel = mapOpenAIThinkingLevel;

/**
 * DeepSeek reasoning_effort 只接受 none / low / high / max（默认开启思考、effort=high）。
 * 官方把其他常见等级映射为：minimal → low，medium / xhigh → high
 * （https://api-docs.deepseek.com/api/create-chat-completion 的 reasoning_effort 说明，
 *  https://api-docs.deepseek.com/guides/thinking_mode 的 effort 对照表）。
 * 这里按同样的映射直接发送合法取值，避免这些等级被当成未设置而落到服务端默认的 high。
 */
export function mapDeepSeekThinkingLevel(value: unknown): DeepSeekThinkingLevel | undefined {
  const level = normalizeThinkingLevel(value);
  switch (level) {
    case 'none':
      return 'none';
    case 'minimal':
    case 'low':
      return 'low';
    case 'medium':
    case 'high':
    case 'xhigh':
      return 'high';
    case 'max':
      return 'max';
    default:
      return undefined;
  }
}

export type ReasoningMode = 'pro' | 'standard';

export function normalizeReasoningMode(value: unknown): ReasoningMode | undefined {
  if (typeof value !== 'string') return undefined;
  const normalized = value.trim().toLowerCase();
  switch (normalized) {
    case 'pro':
    case 'standard':
      return normalized;
    default:
      return undefined;
  }
}
