import { describe, expect, it } from 'vitest';

import {
  createClaudeProvider,
  createDeepSeekProvider,
  createGeminiProvider,
  createOpenAICompatibleProvider,
  createOpenAIResponsesProvider,
} from '../src/index.js';
import type { LLMProviderLike, LLMRequest } from '../src/index.js';

const request: LLMRequest = {
  contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
};

function createProviders(requestBody?: Record<string, unknown>): Array<{ name: string; provider: LLMProviderLike }> {
  return [
    {
      name: 'OpenAI Responses',
      provider: createOpenAIResponsesProvider({
        provider: 'openai-responses',
        model: 'gpt-test',
        apiKey: 'sk-test',
        baseUrl: 'https://api.openai.test/v1',
        ...(requestBody ? { requestBody } : {}),
      }),
    },
    {
      name: 'OpenAI Compatible',
      provider: createOpenAICompatibleProvider({
        provider: 'openai-compatible',
        model: 'gpt-test',
        apiKey: 'sk-test',
        baseUrl: 'https://api.openai.test/v1',
        ...(requestBody ? { requestBody } : {}),
      }),
    },
    {
      name: 'Claude',
      provider: createClaudeProvider({
        provider: 'claude',
        model: 'claude-test',
        apiKey: 'sk-test',
        baseUrl: 'https://api.anthropic.test/v1',
        ...(requestBody ? { requestBody } : {}),
      }),
    },
    {
      name: 'Gemini',
      provider: createGeminiProvider({
        provider: 'gemini',
        model: 'gemini-test',
        apiKey: 'gemini-test',
        baseUrl: 'https://generativelanguage.googleapis.test/v1beta',
        ...(requestBody ? { requestBody } : {}),
      }),
    },
    {
      name: 'DeepSeek',
      provider: createDeepSeekProvider({
        provider: 'deepseek',
        model: 'deepseek-test',
        apiKey: 'sk-test',
        baseUrl: 'https://api.deepseek.test/v1',
        ...(requestBody ? { requestBody } : {}),
      }),
    },
  ];
}

function createOpenAIResponsesProviderForTest(requestBody?: Record<string, unknown>) {
  return createOpenAIResponsesProvider({
    provider: 'openai-responses',
    model: 'gpt-test',
    apiKey: 'sk-test',
    baseUrl: 'https://api.openai.test/v1',
    ...(requestBody ? { requestBody } : {}),
  });
}

describe('requestBody.tools normalization', () => {
  it('所有 provider 都把单个工具对象规范为单元素数组', async () => {
    for (const { name, provider } of createProviders({ tools: { type: 'web_search' } })) {
      const dry = await provider.dryRun(request, { stream: true });
      expect((dry.body as any).tools, name).toEqual([{ type: 'web_search' }]);
    }
  });

  it('所有 provider 都保持合法数组顺序并跳过非对象项', async () => {
    const requestBody = {
      tools: [
        { type: 'web_search' },
        'invalid-tool',
        null,
        42,
        { type: 'web_search_preview' },
      ],
    };
    for (const { name, provider } of createProviders(requestBody)) {
      const dry = await provider.dryRun(request, { stream: true });
      expect((dry.body as any).tools, name).toEqual([
        { type: 'web_search' },
        { type: 'web_search_preview' },
      ]);
    }
  });

  it('所有 provider 都忽略完全非法的 tools 形状', async () => {
    for (const { name, provider } of createProviders({ tools: 'web_search' })) {
      const dry = await provider.dryRun(request, { stream: true });
      expect((dry.body as any).tools, name).toBeUndefined();
    }
  });

  it('与统一函数工具合并后仍保持标准数组', async () => {
    const provider = createOpenAIResponsesProviderForTest({ tools: { type: 'web_search' } });
    const requestWithFunctionTool: LLMRequest = {
      ...request,
      tools: [{
        functionDeclarations: [{
          name: 'read_file',
          description: 'Read a file',
          parameters: {
            type: 'object',
            properties: { path: { type: 'string' } },
            required: ['path'],
          },
        }],
      }],
    };

    const dry = await provider.dryRun(requestWithFunctionTool, { stream: true });

    expect((dry.body as any).tools).toEqual([
      {
        type: 'function',
        name: 'read_file',
        description: 'Read a file',
        parameters: {
          type: 'object',
          properties: { path: { type: 'string' } },
          required: ['path'],
        },
      },
      { type: 'web_search' },
    ]);
  });

  it('非法 tools override 不会误删统一编码生成的函数工具', async () => {
    const provider = createOpenAIResponsesProviderForTest({ tools: 'invalid-tools-shape' });
    const requestWithFunctionTool: LLMRequest = {
      ...request,
      tools: [{
        functionDeclarations: [{
          name: 'read_file',
          description: 'Read a file',
          parameters: { type: 'object', properties: {} },
        }],
      }],
    };

    const dry = await provider.dryRun(requestWithFunctionTool, { stream: true });

    expect((dry.body as any).tools).toEqual([{
      type: 'function',
      name: 'read_file',
      description: 'Read a file',
      parameters: { type: 'object', properties: {} },
    }]);
  });

  it('runtime requestBody override 的单对象 tools 也会被规范', async () => {
    const provider = createOpenAIResponsesProviderForTest();
    provider.patchRequestBodyOverrides?.({ tools: { type: 'web_search' } });

    const dry = await provider.dryRun(request, { stream: true });

    expect((dry.body as any).tools).toEqual([{ type: 'web_search' }]);
  });

  it('compact requestBody 的单对象 tools 同样规范为数组', async () => {
    const provider = createOpenAIResponsesProviderForTest();

    const dry = await provider.compactDryRun(request, {
      requestBody: { tools: { type: 'web_search' } },
    });

    expect((dry.body as any).tools).toEqual([{ type: 'web_search' }]);
  });
});
