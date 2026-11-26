import { describe, expect, it } from 'vitest';
import {
  Agent,
  applyPatchTool,
  RunContext,
  RunState,
  Runner,
  shellTool,
} from '../src';
import { ScriptedModel, assistantMessage } from '../src/testing';
import { FakeEditor, FakeShell } from './stubs';

describe('shell and apply_patch tool call ids', () => {
  it('pass the approved call id to the shell and the editor after resuming', async () => {
    const shell = new FakeShell();
    const editor = new FakeEditor();
    const approvalCallIds: (string | undefined)[] = [];
    const model = new ScriptedModel([
      [
        {
          type: 'shell_call',
          callId: 'call_shell',
          status: 'completed',
          action: { commands: ['echo hi'] },
        },
        {
          type: 'apply_patch_call',
          callId: 'call_patch',
          status: 'completed',
          operation: { type: 'delete_file', path: 'notes.txt' },
        },
      ],
      [assistantMessage('Done.')],
    ]);
    const agent = new Agent({
      name: 'Builder',
      model,
      tools: [
        shellTool({
          shell,
          needsApproval: async (_runContext, _action, callId) => {
            approvalCallIds.push(callId);
            return true;
          },
        }),
        applyPatchTool({
          editor,
          needsApproval: async (_runContext, _operation, callId) => {
            approvalCallIds.push(callId);
            return true;
          },
        }),
      ],
    });
    const runner = new Runner({ tracingDisabled: true });

    const first = await runner.run(agent, 'Clean up.');
    expect(approvalCallIds).toEqual(['call_shell', 'call_patch']);
    expect(shell.calls).toHaveLength(0);
    expect(editor.operations).toHaveLength(0);

    // Approve from a serialized state, as a human-in-the-loop flow would.
    const state = await RunState.fromString(agent, first.state.toString());
    for (const interruption of state.getInterruptions()) {
      state.approve(interruption);
    }
    const resumed = await runner.run(agent, state);

    expect(resumed.finalOutput).toBe('Done.');
    expect(shell.contexts).toEqual([
      { runContext: expect.any(RunContext), toolCallId: 'call_shell' },
    ]);
    expect(editor.contexts).toEqual([
      { runContext: expect.any(RunContext), toolCallId: 'call_patch' },
    ]);
    expect(shell.contexts[0]?.runContext).toBe(resumed.state._context);
  });
});
