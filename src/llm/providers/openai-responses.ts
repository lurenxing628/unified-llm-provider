/**
 * OpenAI Responses Provider
 */

import { DEFAULTS } from '../../config/llm.js';
import type { LLMConfig, LLMTransportMode } from '../../config/types.js';
import { OpenAIResponsesFormat } from '../formats/openai-responses.js';
import { LLMProvider } from './base.js';

export function createOpenAIResponsesProvider(config: LLMConfig): LLMProvider {
  const defaults = DEFAULTS['openai-responses'];
  const model = config.model || String(defaults.model ?? 'gpt-4o');
  const baseUrl = (config.baseUrl || defaults.baseUrl || '').replace(/\/+$/, '');
  const url = config.endpoint?.url || `${baseUrl}/responses`;
  const transport = resolveTransport(config);
  const webSocketSessionKey = typeof config.webSocketSessionKey === 'string' && config.webSocketSessionKey.trim()
    ? config.webSocketSessionKey.trim()
    : typeof config.endpoint?.webSocketSessionKey === 'string' && config.endpoint.webSocketSessionKey.trim()
      ? config.endpoint.webSocketSessionKey.trim()
      : undefined;
  const webSocketOptions = config.webSocketOptions ?? config.endpoint?.webSocketOptions;

  return new LLMProvider(
    // LimCode GPT-6 家族：HTTP/SSE 原生解码事件（codec 内部按精确模型族门禁；WS 由 LimCode 会话自产事件）。
    new OpenAIResponsesFormat(model, config.promptCache, true),
    {
      url,
      streamUrl: config.endpoint?.streamUrl,
      compactUrl: config.endpoint?.compactUrl || `${baseUrl}/responses/compact`,
      ...(transport === 'websocket' ? { transport, webSocketUrl: config.endpoint?.webSocketUrl || toWebSocketUrl(url) } : {}),
      ...(webSocketSessionKey ? { webSocketSessionKey } : {}),
      ...(transport === 'websocket' && webSocketOptions ? { webSocketOptions } : {}),
      headers: {
        Authorization: `Bearer ${config.apiKey ?? ''}`,
        ...config.headers,
        ...config.endpoint?.headers,
      },
      fetch: config.fetch,
      debug: config.debug,
      proxy: config.endpoint?.proxy ?? config.proxy,
    },
    config.name ?? `OpenAIResponses(${model})`,
    config.requestBody,
    'openai-responses',
  );
}

function resolveTransport(config: LLMConfig): LLMTransportMode {
  return config.transport === 'websocket' || config.endpoint?.transport === 'websocket' ? 'websocket' : 'http';
}

function toWebSocketUrl(url: string): string {
  const parsed = new URL(url);
  if (parsed.protocol === 'https:') parsed.protocol = 'wss:';
  else if (parsed.protocol === 'http:') parsed.protocol = 'ws:';
  return parsed.toString();
}
