import { describe, expect, it } from 'vitest';
import OpenAI from 'openai';
import { Agent, Runner, tool } from '@openai/agents-core';
import { z } from 'zod';
import { OpenAIProvider } from '../src/openaiProvider';

describe('Chat Completions tool calls without an index', () => {
  it('runs each parallel call that Gemini streams without an index', async () => {
    // Chunk shapes from a recorded Gemini OpenAI-compatible stream: each call arrives whole, with an id and no index.
    const turns = [
      [
        {
          index: 0,
          delta: {
            role: 'assistant',
            tool_calls: [
              {
                extra_content: { google: { thought_signature: 'sig' } },
                function: {
                  arguments: '{"city":"Taipei"}',
                  name: 'get_weather',
                },
                id: 'call_108766',
                type: 'function',
              },
            ],
          },
        },
        {
          index: 0,
          delta: {
            role: 'assistant',
            tool_calls: [
              {
                function: {
                  arguments: '{"city":"Kaohsiung"}',
                  name: 'get_weather',
                },
                id: 'call_108772',
                type: 'function',
              },
            ],
          },
        },
        { index: 0, delta: { role: 'assistant' }, finish_reason: 'stop' },
      ],
      [
        {
          index: 0,
          delta: { role: 'assistant', content: 'Sunny in both cities.' },
        },
        { index: 0, delta: { role: 'assistant' }, finish_reason: 'stop' },
      ],
    ];
    const bodies: any[] = [];
    // Capture real SSE parsing and request serialization; ScriptedModel skips both.
    const client = new OpenAI({
      apiKey: 'test-key',
      baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai/',
      fetch: async (_url, init) => {
        bodies.push(JSON.parse(init!.body as string));
        const chunks = turns[bodies.length - 1].map(
          (choice) =>
            `data: ${JSON.stringify({
              id: `chatcmpl_${bodies.length}`,
              object: 'chat.completion.chunk',
              created: 0,
              model: 'gemini-3.1-flash-lite',
              choices: [choice],
            })}\n\n`,
        );
        return new Response(`${chunks.join('')}data: [DONE]\n\n`, {
          headers: { 'content-type': 'text/event-stream' },
        });
      },
    });
    const cities: string[] = [];
    const agent = new Agent({
      name: 'Weather',
      model: 'gemini-3.1-flash-lite',
      tools: [
        tool({
          name: 'get_weather',
          description: 'Gets the weather for a city.',
          parameters: z.object({ city: z.string() }),
          execute: async ({ city }) => {
            cities.push(city);
            return `Sunny in ${city}.`;
          },
        }),
      ],
    });
    const runner = new Runner({
      modelProvider: new OpenAIProvider({
        openAIClient: client,
        useResponses: false,
      }),
      tracingDisabled: true,
    });

    const result = await runner.run(
      agent,
      'What is the weather in Taipei and Kaohsiung?',
      { stream: true },
    );
    await result.completed;

    expect(cities.sort()).toEqual(['Kaohsiung', 'Taipei']);
    expect(result.finalOutput).toBe('Sunny in both cities.');
    expect(bodies).toHaveLength(2);
    const replayed = bodies[1].messages.filter(
      (message: any) => message.role !== 'user',
    );
    expect(replayed).toEqual([
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            id: 'call_108766',
            type: 'function',
            function: { name: 'get_weather', arguments: '{"city":"Taipei"}' },
          },
          {
            id: 'call_108772',
            type: 'function',
            function: {
              name: 'get_weather',
              arguments: '{"city":"Kaohsiung"}',
            },
          },
        ],
      },
      {
        role: 'tool',
        tool_call_id: 'call_108766',
        content: 'Sunny in Taipei.',
      },
      {
        role: 'tool',
        tool_call_id: 'call_108772',
        content: 'Sunny in Kaohsiung.',
      },
    ]);
  });
});
