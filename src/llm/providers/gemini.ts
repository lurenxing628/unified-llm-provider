/**
 * Gemini Provider
 */

import { DEFAULTS } from '../../config/llm.js';
import type { LLMConfig } from '../../config/types.js';
import { GeminiFormat } from '../formats/gemini.js';
import { LLMProvider } from './base.js';

export function createGeminiProvider(config: LLMConfig): LLMProvider {
  const defaults = DEFAULTS.gemini;
  const model = config.model || String(defaults.model ?? 'gemini-2.0-flash');
  const baseUrl = (config.baseUrl || defaults.baseUrl || '').replace(/\/+$/, '');
  const modelPath = encodeGeminiModelPath(model);

  return new LLMProvider(
    new GeminiFormat(),
    {
      url: config.endpoint?.url || `${baseUrl}/models/${modelPath}:generateContent`,
      streamUrl: config.endpoint?.streamUrl || `${baseUrl}/models/${modelPath}:streamGenerateContent?alt=sse`,
      headers: {
        'x-goog-api-key': config.apiKey ?? '',
        ...config.headers,
        ...config.endpoint?.headers,
      },
      fetch: config.fetch,
      debug: config.debug,
      proxy: config.endpoint?.proxy ?? config.proxy,
    },
    config.name ?? 'Gemini',
    config.requestBody,
    'gemini',
  );
}


/**
 * 把模型 id 放进 URL 路径前做百分号编码。
 *
 * 官方 Gemini 模型 id 只含字母、数字、`-`、`.`（如 gemini-2.5-flash），encodeURIComponent 后不变，
 * 官方请求 URL 逐字节不变；网关风格 id（如 `[v]gemini-3.5-flash`）里的方括号、空格等会被编码成
 * `%5Bv%5Dgemini-3.5-flash`，网关实测同样接受。按 `/` 分段编码、保留 `/` 本身，避免改变带斜杠 id
 * （如 `publisher/model`）原本的路径语义。
 */
function encodeGeminiModelPath(model: string): string {
  return model.split('/').map(segment => encodeURIComponent(segment)).join('/');
}
