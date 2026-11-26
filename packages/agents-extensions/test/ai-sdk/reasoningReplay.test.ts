import { describe, expect, test } from 'vitest';
import { Agent, MemorySession, Runner, tool } from '@openai/agents';
import { ReadableStream } from 'node:stream/web';
import { z } from 'zod';
import { AiSdkModel } from '../../src/ai-sdk/index';
import { stubModel } from './fixtures';

// AI SDK providers number reasoning blocks per response, so Anthropic and Bedrock send `0` every turn.
const PROVIDER_BLOCK_ID = '0';

function turnParts(turn: number) {
  if (turn === 3) {
    return [
      { type: 'text-delta', id: '1', delta: 'Done.' },
      { type: 'response-metadata', id: 'resp_3' },
      {
        type: 'finish',
        finishReason: 'stop',
        usage: { inputTokens: 1, outputTokens: 1 },
      },
    ];
  }
  return [
    { type: 'reasoning-start', id: PROVIDER_BLOCK_ID },
    {
      type: 'reasoning-delta',
      id: PROVIDER_BLOCK_ID,
      delta: `thought ${turn}`,
    },
    {
      type: 'reasoning-delta',
      id: PROVIDER_BLOCK_ID,
      delta: '',
      providerMetadata: { anthropic: { signature: `sig-${turn}` } },
    },
    { type: 'reasoning-end', id: PROVIDER_BLOCK_ID },
    {
      type: 'tool-call',
      toolCallId: `call_${turn}`,
      toolName: 'lookup',
      input: '{}',
    },
    { type: 'response-metadata', id: `resp_${turn}` },
    {
      type: 'finish',
      finishReason: 'tool-calls',
      usage: { inputTokens: 1, outputTokens: 1 },
    },
  ];
}

function turnContent(turn: number) {
  return turnParts(turn).flatMap((part: any) => {
    if (part.type === 'reasoning-delta' && part.delta) {
      return [
        {
          type: 'reasoning',
          text: part.delta,
          providerMetadata: {
            anthropic: { signature: `sig-${turn}` },
          },
        },
      ];
    }
    if (part.type === 'tool-call') {
      return [part];
    }
    if (part.type === 'text-delta') {
      return [{ type: 'text', text: part.delta }];
    }
    return [];
  });
}

// Reduce each prompt part to what identifies it, so the order of turns is easy to compare.
function describePrompt(prompt: any[]): string[] {
  return prompt
    .filter((message) => message.role !== 'system' && message.role !== 'user')
    .flatMap((message) =>
      message.content.map((part: any) => {
        if (part.type === 'reasoning') {
          return `reasoning ${part.text} ${part.providerOptions?.anthropic?.signature}`;
        }
        return `${part.type} ${part.toolCallId}`;
      }),
    );
}

describe('AI SDK reasoning replay', () => {
  test.each([true, false])(
    'keeps each turn reasoning in its own place (stream=%s)',
    async (stream) => {
      const prompts: any[][] = [];
      const model = new AiSdkModel(
        stubModel({
          async doStream(options) {
            prompts.push(options.prompt);
            const parts = turnParts(prompts.length);
            return {
              stream: ReadableStream.from(
                (async function* () {
                  yield* parts;
                })(),
              ),
            } as any;
          },
          async doGenerate(options) {
            prompts.push(options.prompt);
            const turn = prompts.length;
            return {
              content: turnContent(turn),
              finishReason: turn === 3 ? 'stop' : 'tool-calls',
              usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
              response: { id: `resp_${turn}` },
              providerMetadata: {},
              warnings: [],
            } as any;
          },
        }),
      );
      const agent = new Agent({
        name: 'Reasoner',
        model,
        tools: [
          tool({
            name: 'lookup',
            description: 'Looks something up.',
            parameters: z.object({}),
            execute: async () => 'ok',
          }),
        ],
      });
      const session = new MemorySession();
      const runner = new Runner({ tracingDisabled: true });

      const result = stream
        ? await runner.run(agent, 'Think twice.', { stream: true, session })
        : await runner.run(agent, 'Think twice.', { session });
      if ('completed' in result) await result.completed;

      expect(result.finalOutput).toBe('Done.');
      expect(prompts).toHaveLength(3);
      expect(describePrompt(prompts[2])).toEqual([
        'reasoning thought 1 sig-1',
        'tool-call call_1',
        'tool-result call_1',
        'reasoning thought 2 sig-2',
        'tool-call call_2',
        'tool-result call_2',
      ]);

      const savedReasoning = (await session.getItems())
        .filter((item) => item.type === 'reasoning')
        .map((item) => item.content[0]?.text);
      expect(savedReasoning).toEqual(['thought 1', 'thought 2']);
    },
  );
});
