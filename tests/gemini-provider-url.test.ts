/**
 * B7 Gemini provider：URL 里的模型 id 做百分号编码。
 * 官方模型 id 编码后不变（请求 URL 与修复前逐字节一致）；网关 id 如 `[v]gemini-3.5-flash` 编码后网关实测同样接受。
 */
import { describe, expect, it } from 'vitest';

import { createGeminiProvider } from '../src/index.js';
import type { LLMRequest } from '../src/index.js';

const request: LLMRequest = { contents: [{ role: 'user', parts: [{ text: 'hi' }] }] };
const BASE = 'https://generativelanguage.googleapis.com/v1beta';

async function urls(model: string, extra: Record<string, unknown> = {}): Promise<{ url: string; streamUrl: string }> {
  const provider = createGeminiProvider({ provider: 'gemini', model, apiKey: 'k', baseUrl: BASE, ...extra } as any);
  const [chat, stream] = await Promise.all([
    provider.dryRun(request, { stream: false }),
    provider.dryRun(request, { stream: true }),
  ]);
  return { url: chat.url, streamUrl: stream.url };
}

describe('B7 Gemini provider URL 对模型 id 编码', () => {
  it('官方模型 id 编码后不变，URL 与修复前逐字节一致', async () => {
    for (const model of ['gemini-2.5-flash', 'gemini-3-pro-preview', 'gemini-2.0-flash-001', 'gemini-2.5-flash-lite-preview-09-2025']) {
      expect(await urls(model)).toEqual({
        url: `${BASE}/models/${model}:generateContent`,
        streamUrl: `${BASE}/models/${model}:streamGenerateContent?alt=sse`,
      });
    }
  });

  it('网关 id `[v]gemini-3.5-flash` 的方括号被编码', async () => {
    expect(await urls('[v]gemini-3.5-flash')).toEqual({
      url: `${BASE}/models/%5Bv%5Dgemini-3.5-flash:generateContent`,
      streamUrl: `${BASE}/models/%5Bv%5Dgemini-3.5-flash:streamGenerateContent?alt=sse`,
    });
  });

  it('空格、问号、井号等会破坏 URL 的字符被编码', async () => {
    const { url } = await urls('my model?x#y');
    expect(url).toBe(`${BASE}/models/my%20model%3Fx%23y:generateContent`);
  });

  it('带斜杠的 id 按段编码，保留斜杠的路径语义（与修复前一致）', async () => {
    const { url } = await urls('publisher/gemini-2.5-flash');
    expect(url).toBe(`${BASE}/models/publisher/gemini-2.5-flash:generateContent`);
  });

  it('显式配置的 endpoint.url / streamUrl 原样使用，不做编码', async () => {
    const result = await urls('[v]gemini-3.5-flash', {
      endpoint: { url: 'https://gw.test/custom/[v]:generate', streamUrl: 'https://gw.test/custom/[v]:stream' },
    });
    expect(result).toEqual({ url: 'https://gw.test/custom/[v]:generate', streamUrl: 'https://gw.test/custom/[v]:stream' });
  });
});
