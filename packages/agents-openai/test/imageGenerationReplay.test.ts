import { describe, expect, it } from 'vitest';
import OpenAI from 'openai';
import { Agent, Runner } from '@openai/agents-core';
import { OpenAIProvider } from '../src/openaiProvider';
import { imageGenerationTool } from '../src/tools';

describe('image generation history replay', () => {
  it.each([false, true])(
    'sends generated images back in the next request (stream=%s)',
    async (stream) => {
      const generated = {
        type: 'image_generation_call' as const,
        id: 'ig_1',
        status: 'completed' as const,
        result: 'iVBORw0KGgoAAAANSUhEUg==',
        revised_prompt: 'A cat sitting on a windowsill.',
        output_format: 'png' as const,
        size: '1024x1024',
        quality: 'low' as const,
        background: 'opaque' as const,
      };
      const failed = {
        type: 'image_generation_call' as const,
        id: 'ig_2',
        status: 'failed' as const,
        result: null,
      };
      const message = {
        type: 'message' as const,
        id: 'msg_1',
        role: 'assistant' as const,
        status: 'completed' as const,
        content: [
          {
            type: 'output_text' as const,
            text: 'Here is your cat.',
            annotations: [],
          },
        ],
      };
      const outputs = [
        [generated, failed, message],
        [{ ...message, id: 'msg_2' }],
      ];
      const bodies: OpenAI.Responses.ResponseCreateParams[] = [];
      // Capture real wire conversion and JSON serialization; ScriptedModel skips both.
      const client = new OpenAI({
        apiKey: 'test-key',
        fetch: async (_url, init) => {
          bodies.push(JSON.parse(init!.body as string));
          const response = {
            id: `resp_${bodies.length}`,
            object: 'response',
            status: 'completed',
            output: outputs[bodies.length - 1],
            usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
          };
          return new Response(
            stream
              ? `event: response.completed\ndata: ${JSON.stringify({ type: 'response.completed', sequence_number: 0, response })}\n\n`
              : JSON.stringify(response),
            {
              headers: {
                'content-type': stream
                  ? 'text/event-stream'
                  : 'application/json',
              },
            },
          );
        },
      });
      const runner = new Runner({
        modelProvider: new OpenAIProvider({ openAIClient: client }),
        tracingDisabled: true,
      });
      const agent = new Agent({
        name: 'Image replay',
        model: 'gpt-5',
        modelSettings: { store: false },
        tools: [imageGenerationTool()],
      });

      const first = stream
        ? await runner.run(agent, 'Draw a cat.', { stream: true })
        : await runner.run(agent, 'Draw a cat.');
      if ('completed' in first) await first.completed;
      const savedCall = first.history.find(
        (item) => item.type === 'hosted_tool_call' && item.id === 'ig_1',
      );
      expect(savedCall).toHaveProperty('output', generated.result);
      expect(savedCall?.providerData).not.toHaveProperty('result');

      const nextInput = [
        ...first.history,
        { role: 'user' as const, content: 'Make it blue.' },
      ];
      const next = stream
        ? await runner.run(agent, nextInput, { stream: true })
        : await runner.run(agent, nextInput);
      if ('completed' in next) await next.completed;
      expect(bodies).toHaveLength(2);
      expect(bodies[1].store).toBe(false);
      expect(bodies[1].previous_response_id).toBeUndefined();
      expect(bodies[1].input).toEqual(
        expect.arrayContaining([generated, failed]),
      );
    },
  );
});
