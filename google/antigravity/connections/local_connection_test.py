# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for LocalConnection."""

import asyncio
import json
import unittest
from unittest import mock

from absl.testing import absltest
from google.protobuf import json_format

from google.antigravity import types
from google.antigravity.connections import local_connection
from google.antigravity.connections import localharness_pb2
from google.antigravity.hooks import hook_runner
from google.antigravity.hooks import hooks as hooks_base
from google.antigravity.tools import tool_runner
from google.antigravity.types import QuestionResponse


class FakeWebSocket:

  def __init__(self):
    self.queue = asyncio.Queue()
    self.sent_messages = []

  async def send(self, message):
    self.sent_messages.append(message)

  async def put_event(self, event):
    await self.queue.put(json_format.MessageToJson(event))

  def __aiter__(self):

    async def _gen():
      while True:
        msg = await self.queue.get()
        if msg is None:  # Sentinel for close
          break
        yield msg

    return _gen()

  async def close(self):
    await self.queue.put(None)


class LocalConnectionTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()
    self.tool_runner = tool_runner.ToolRunner()

  async def test_receive_steps_basic(self):
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="Hello world",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )

    await self.mock_ws.put_event(event)
    await self.mock_ws.close()

    # Simulate that a turn is active (send clears this in reality)
    conn._is_idle.clear()

    steps = []
    async for step in conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].content, "Hello world")
    self.assertEqual(steps[0].status, types.StepStatus.ACTIVE)
    self.assertEqual(steps[0].source, types.StepSource.MODEL)

  def test_local_connection_step_from_dict(self):
    """Tests that LocalConnectionStep maps fields correctly."""
    step_dict = {
        "step_index": 1,
        "text": "Hello world",
        "state": "STATE_ACTIVE",
        "source": "SOURCE_MODEL",
        "target": "TARGET_USER",
    }
    step = local_connection.LocalConnectionStep.from_dict(step_dict)
    self.assertEqual(step.id, "1")
    self.assertEqual(step.content, "Hello world")
    self.assertEqual(step.status, types.StepStatus.ACTIVE)
    self.assertEqual(step.source, types.StepSource.MODEL)
    self.assertEqual(step.target, "TARGET_USER")

  async def test_turn_hook_deny(self):
    hr = hook_runner.HookRunner()

    class DenyingTurnHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=False, message="Denied by hook")

    hr.pre_turn_hooks.append(DenyingTurnHook())

    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await conn.send("Hello")

    steps = []
    async for step in conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(steps[0].status, types.StepStatus.CANCELED)
    self.assertEqual(steps[0].error, "Denied by hook")

  async def test_tool_hook_deny(self):
    hr = hook_runner.HookRunner()

    class DenyingToolHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=False, message="Denied tool")

    hr.pre_tool_call_decide_hooks.append(DenyingToolHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_1",
            name="some_tool",
            arguments_json="{}",
        )
    )

    await self.mock_ws.put_event(event)

    # Allow reader loop to process
    await asyncio.sleep(0.1)

    # Verify that ToolResponse was sent back to harness denying it
    self.assertEqual(len(self.mock_ws.sent_messages), 1)
    sent_data = json.loads(self.mock_ws.sent_messages[0])
    self.assertIn("toolResponse", sent_data)
    resp = sent_data["toolResponse"]
    self.assertEqual(resp["id"], "call_1")
    self.assertIn("Denied tool", resp["responseJson"])

  async def test_tool_confirmation_request_integration(self):
    hr = hook_runner.HookRunner()

    class DenyingToolHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=False)

    hr.pre_tool_call_decide_hooks.append(DenyingToolHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )

    await self.mock_ws.put_event(event)

    await asyncio.sleep(0.1)

    self.assertEqual(len(self.mock_ws.sent_messages), 1)
    sent_data = json.loads(self.mock_ws.sent_messages[0])
    self.assertIn("toolConfirmation", sent_data)
    self.assertEqual(sent_data["toolConfirmation"]["trajectoryId"], "test_traj")
    self.assertFalse(sent_data["toolConfirmation"]["accepted"])

  async def test_tool_confirmation_uses_enum_value_for_view_file(self):
    """Verifies that hooks receive the BuiltinTools enum value as the tool name.

    Why: Hooks should see stable, semantic names (e.g. "view_file") rather
    than raw proto field names. For view_file these happen to match, but the
    test locks in the contract.
    """
    captured_tool_names = []

    class CapturingToolHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured_tool_names.append(data.name)
        return hooks_base.HookResult(allow=True)

    hr = hook_runner.HookRunner()
    hr.pre_tool_call_decide_hooks.append(CapturingToolHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )
    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    self.assertEqual(captured_tool_names, [types.BuiltinTools.VIEW_FILE.value])

  async def test_tool_confirmation_uses_enum_value_for_find_file(self):
    """Verifies that a find_file step update is correctly recognized.

    Why: find_file is a harness builtin tool that must be correctly identified
    in _BUILTIN_TOOL_PROTO_FIELDS so hooks receive the right name.
    """
    captured_tool_names = []

    class CapturingToolHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured_tool_names.append(data.name)
        return hooks_base.HookResult(allow=True)

    hr = hook_runner.HookRunner()
    hr.pre_tool_call_decide_hooks.append(CapturingToolHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            find_file=localharness_pb2.ActionFindFile(
                directory_path="file:///home/user",
                query="*.py",
            ),
        )
    )
    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    self.assertEqual(captured_tool_names, [types.BuiltinTools.FIND_FILE.value])

  async def test_question_hook_integration(self):
    hr = hook_runner.HookRunner()

    class AutoAnswerQuestionHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.QuestionHookResult(
            responses=[
                QuestionResponse(selected_option_ids=["1"]),
            ]
        )

    hr.on_interaction_hooks.append(AutoAnswerQuestionHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            questions_request=localharness_pb2.UserQuestionsRequest(
                questions=[
                    localharness_pb2.UserQuestion(
                        multiple_choice=localharness_pb2.MultipleChoice(
                            question="Do you agree?",
                            choices=["Yes", "No"],
                        )
                    )
                ]
            ),
        )
    )

    await self.mock_ws.put_event(event)

    await asyncio.sleep(0.1)

    self.assertEqual(len(self.mock_ws.sent_messages), 1)
    sent_data = json.loads(self.mock_ws.sent_messages[0])
    self.assertIn("questionResponse", sent_data)
    self.assertEqual(sent_data["questionResponse"]["trajectoryId"], "test_traj")

  async def test_deduplication_of_wait_requests(self):
    """Verifies that multiple updates for the same wait state don't duplicate."""
    hr = hook_runner.HookRunner()

    class CountingHook:

      def __init__(self):
        self.call_count = 0

      async def run(self, context, data):  # pylint: disable=unused-argument
        del context, data
        self.call_count += 1
        return hooks_base.HookResult(allow=True)

    hook_instance = CountingHook()
    hr.pre_tool_call_decide_hooks.append(hook_instance)

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
        )
    )

    # Send the exact same wait event three times (e.g. keepalives)
    await self.mock_ws.put_event(event)
    await self.mock_ws.put_event(event)
    await self.mock_ws.put_event(event)

    await asyncio.sleep(0.2)  # Give reader loop and async tasks time to process

    # Hook should only be called ONCE despite 3 events, thanks to _handled_waits
    self.assertEqual(hook_instance.call_count, 1)
    self.assertEqual(len(self.mock_ws.sent_messages), 1)

  async def test_async_non_blocking_dispatch(self):
    """Verifies that wait handlers run concurrently without blocking loop."""
    hr = hook_runner.HookRunner()

    class BlockingHook:

      def __init__(self):
        self.started = False
        self.finished = False

      async def run(self, context, data):  # pylint: disable=unused-argument
        del context, data
        self.started = True
        await asyncio.sleep(0.5)  # Simulate a slow human interaction
        self.finished = True
        return hooks_base.HookResult(allow=True)

    hook_instance = BlockingHook()
    hr.pre_tool_call_decide_hooks.append(hook_instance)

    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    wait_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="traj_1",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
            view_file=localharness_pb2.ActionViewFile(file_path="/foo"),
        )
    )

    # An event from another subagent that should not be blocked
    active_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="traj_2",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            text="I am another agent running concurrently",
        )
    )

    await self.mock_ws.put_event(wait_event)
    await self.mock_ws.put_event(active_event)

    # Wait just a tiny bit to let the reader loop process both events
    await asyncio.sleep(0.1)

    # The hook should have started, but not finished (because of the 0.5s sleep)
    self.assertTrue(hook_instance.started)
    self.assertFalse(hook_instance.finished)

    # The reader loop SHOULD NOT be blocked! It should have processed traj_2
    # and put both events into the step queue.
    step1 = await conn._step_queue.get()
    step2 = await conn._step_queue.get()

    self.assertEqual(step1.trajectory_id, "traj_1")
    self.assertEqual(step2.trajectory_id, "traj_2")
    self.assertEqual(step2.content, "I am another agent running concurrently")

    # Cleanup: Wait for hook to finish so we don't get pending task errors
    await asyncio.sleep(0.5)

  async def test_state_transition_clears_handled_requests(self):
    """Verifies WAITING -> ACTIVE -> WAITING transitions re-trigger handlers."""
    hr = hook_runner.HookRunner()

    class CountingHook:

      def __init__(self):
        self.call_count = 0

      async def run(self, context, data):  # pylint: disable=unused-argument
        del context, data
        self.call_count += 1
        return hooks_base.HookResult(allow=True)

    hook_instance = CountingHook()
    hr.pre_tool_call_decide_hooks.append(hook_instance)

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    def create_wait_event():
      return localharness_pb2.OutputEvent(
          step_update=localharness_pb2.StepUpdate(
              step_index=1,
              trajectory_id="test_traj",
              state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
              tool_confirmation_request=localharness_pb2.ToolConfirmationRequest(),
              view_file=localharness_pb2.ActionViewFile(file_path="/foo/bar"),
          )
      )

    active_event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            trajectory_id="test_traj",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
        )
    )

    # 1. First wait
    await self.mock_ws.put_event(create_wait_event())
    await asyncio.sleep(0.1)
    self.assertEqual(hook_instance.call_count, 1)

    # 2. Transition back to active (this should clear the handled_requests)
    await self.mock_ws.put_event(active_event)
    await asyncio.sleep(0.1)

    # 3. Second wait on the SAME step
    await self.mock_ws.put_event(create_wait_event())
    await asyncio.sleep(0.1)

    # The hook should be called a second time!
    self.assertEqual(hook_instance.call_count, 2)
    self.assertEqual(len(self.mock_ws.sent_messages), 2)

  async def test_yielding_wait_state_to_queue(self):
    """Verifies that wait states are correctly yielded to the step queue for the UI to render."""
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=5,
            trajectory_id="ui_traj",
            state=localharness_pb2.StepUpdate.STATE_WAITING_FOR_USER,
            text="Waiting for confirmation",
        )
    )

    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    # We should be able to retrieve this step from the queue
    step_obj = await conn._step_queue.get()
    self.assertEqual(step_obj.trajectory_id, "ui_traj")
    self.assertEqual(step_obj.id, "ui_traj-5")
    self.assertEqual(step_obj.status, types.StepStatus.WAITING_FOR_USER)
    self.assertEqual(step_obj.content, "Waiting for confirmation")

  async def test_cancel(self):
    """Verifies that cancel sends a halt request."""
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
    )

    await conn.cancel()

    self.assertEqual(len(self.mock_ws.sent_messages), 1)
    sent_data = json.loads(self.mock_ws.sent_messages[0])
    self.assertTrue(sent_data.get("haltRequest"))


class LocalConnectionStepFromDictTest(unittest.TestCase):
  """Tests for LocalConnectionStep.from_dict derivation logic.

  Specifically targets the is_final_response calculation (lines 134-137) and
  edge cases in step type detection.
  """

  def test_is_final_response_true(self):
    """Verifies is_final_response is True when source=MODEL, state=DONE, and text is present.

    Why: This is the canonical "agent finished speaking" signal that callers
    rely on to surface the final answer. The derivation at lines 134-137 must
    produce True when all three conditions hold.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "text": "Here is my answer.",
    })
    self.assertTrue(step.is_final_response)

  def test_is_final_response_false_when_source_not_model(self):
    """Verifies is_final_response is False when source is not MODEL.

    Why: System or user steps that are done and have text should not be
    treated as the model's final answer.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_USER",
        "state": "STATE_DONE",
        "text": "Some user text.",
    })
    self.assertFalse(step.is_final_response)

  def test_is_final_response_false_when_not_done(self):
    """Verifies is_final_response is False when state is not DONE.

    Why: An active model step is still streaming; it should not be treated
    as final until the harness marks it done.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "text": "Partial response...",
    })
    self.assertFalse(step.is_final_response)

  def test_is_final_response_false_when_no_text(self):
    """Verifies is_final_response is False when text is empty.

    Why: A done model step with no text is a structural step (e.g. tool use
    completion), not a final textual response.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
    })
    self.assertFalse(step.is_final_response)

  def test_is_final_response_false_when_error_state(self):
    """Verifies is_final_response is False when state is ERROR."""
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ERROR",
        "text": "Something went wrong",
        "error_message": "internal error",
    })
    self.assertFalse(step.is_final_response)

  def test_step_type_tool_call_with_builtin(self):
    """Verifies that a step with a builtin tool proto field is typed TOOL_CALL."""
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_ACTIVE",
        "view_file": {"file_path": "/foo"},
    })
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

  def test_structured_output_extracted_from_finish(self):
    """Verifies that structured output is extracted when finish payload is present.

    Why: The connection layer is responsible for extracting and parsing
    the final structured output from the wire format so Layer 2 and E2E tests
    can access it natively.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (
                '{"total_revenue": 386.0, "top_selling_product": "Widget A"}'
            ),
        },
    })
    self.assertEqual(
        step.structured_output,
        {"total_revenue": 386.0, "top_selling_product": "Widget A"},
    )

  def test_structured_output_extracted_from_finish_handles_invalid_json(self):
    """Verifies that invalid JSON in finish payload defaults to None.

    Why: The connection layer should handle malformed JSON payloads gracefully
    by returning None instead of raising a fatal exception.
    """
    step = local_connection.LocalConnectionStep.from_dict({
        "source": "SOURCE_MODEL",
        "state": "STATE_DONE",
        "finish": {
            "output_string": (  # Invalid JSON
                '{"total_revenue": 386.0, "top_selling_product": }'
            ),
        },
    })
    self.assertIsNone(step.structured_output)


class LocalConnectionToolCallNoRunnerTest(unittest.IsolatedAsyncioTestCase):
  """Tests for tool call handling when no ToolRunner is configured."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()

  async def test_tool_call_without_runner_yields_step(self):
    """Verifies that a tool call with no ToolRunner queues a step for the user.

    Why: When no ToolRunner is configured, the connection should surface the
    tool call to the caller so they can handle it manually, rather than
    silently dropping it.
    """
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=None,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_99",
            name="custom_tool",
            arguments_json='{"key": "value"}',
        )
    )

    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    step_obj = await conn._step_queue.get()
    self.assertEqual(step_obj.type, types.StepType.TOOL_CALL)
    self.assertEqual(step_obj.tool_calls[0].name, "custom_tool")
    self.assertEqual(step_obj.tool_calls[0].args, {"key": "value"})
    self.assertEqual(step_obj.tool_calls[0].id, "call_99")
    # No messages should have been sent back to the harness.
    self.assertEqual(len(self.mock_ws.sent_messages), 0)


class LocalConnectionStrategyConfigTest(unittest.TestCase):
  """Tests for config-to-proto translation in LocalConnectionStrategy.

  These tests exercise _build_harness_config() directly, without mocking
  any internal logic. Only the strategy constructor and config builder run;
  no subprocess or websocket I/O is triggered.
  """

  def _make_strategy(self, **kwargs):
    """Creates a LocalConnectionStrategy with the given kwargs and a dummy binary_path."""
    defaults = {"binary_path": "/fake/binary"}
    defaults.update(kwargs)
    return local_connection.LocalConnectionStrategy(**defaults)

  def test_default_config_produces_valid_harness_config(self):
    """Verifies that a strategy with all defaults produces a well-formed proto.

    Why: The default path is the most common case. Callers should be able to
    construct a strategy with only binary_path and get a valid HarnessConfig.
    How: Build the config and assert the proto has expected default structure.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertIsInstance(config, localharness_pb2.HarnessConfig)
    # Default: all harness side tools enabled.
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    self.assertTrue(config.harness_side_tools.user_questions.enabled)
    self.assertTrue(config.harness_side_tools.run_command.enabled)
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertTrue(config.harness_side_tools.generate_image.enabled)
    # No gemini config, system instructions, workspaces, or skills by default.
    self.assertFalse(config.HasField("gemini_config"))
    self.assertFalse(config.HasField("system_instructions"))
    self.assertEqual(len(config.workspaces), 0)
    self.assertEqual(len(config.skills_paths), 0)

  def test_capabilities_config_finish_tool_schema_json_to_proto(self):
    """Verifies capabilities config propagates finish tool schema to the proto config.

    Why: The user's custom schema must be delivered to the Go harness so it can
    be appropriately injected into the finish tool declaration.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            finish_tool_schema_json='{"type": "object"}',
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.finish_tool_schema_json, '{"type": "object"}')

  def test_gemini_config_to_proto(self):
    """Verifies GeminiConfig fields translate to the correct proto fields.

    Why: The proto's field names must match the Pydantic model's semantics
    exactly, or the Go harness will receive incorrect configuration.
    How: Set all GeminiConfig fields and assert proto field values.
    """
    strategy = self._make_strategy(
        gemini_config=types.GeminiConfig(
            api_key="test-key",
            models=types.ModelConfig(
                default=types.ModelEntry(name="gemini-2.5-pro"),
            ),
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.api_key, "test-key")
    self.assertEqual(config.gemini_config.model_name, "gemini-2.5-pro")

  def test_gemini_config_none_fields_omitted(self):
    """Verifies that None fields on GeminiConfig are not set on the proto.

    Why: The Go harness uses proto field presence to determine whether to
    apply overrides. Setting empty strings would be semantically wrong.
    How: Create a GeminiConfig with defaults (api_key=None), build proto,
    and assert api_key is not populated.
    """
    strategy = self._make_strategy(gemini_config=types.GeminiConfig())
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.model_name, "gemini-3-flash-preview")
    # api_key should not be set (proto default empty string).
    self.assertEqual(config.gemini_config.api_key, "")

  def test_gemini_config_default_model_name(self):
    """Verifies the default model name propagates correctly.

    Why: The default model name is a critical fallback; if it changes
    unintentionally, agents would use the wrong model.
    How: Create default GeminiConfig and check model_name in proto.
    """
    strategy = self._make_strategy(gemini_config=types.GeminiConfig())
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.model_name, "gemini-3-flash-preview")

  def test_gemini_config_string_shorthand(self):
    """Verifies that a bare model name string creates a proper GeminiConfig."""
    strategy = self._make_strategy(gemini_config="custom-model-name")
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.model_name, "custom-model-name")
    # No API key set in shorthand path.
    self.assertEqual(config.gemini_config.api_key, "")

  def test_system_instructions_string_shorthand(self):
    """Verifies that a plain string normalizes to AppendedSystemInstructions.

    Why: The str shorthand is an ergonomic convenience. It defaults to
    appending.
    How: Pass a string, build proto, and assert the appended field is set.
    """
    strategy = self._make_strategy(system_instructions="Be concise.")
    config = strategy._build_harness_config()
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].content,
        "Be concise.",
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title,
        "user_system_instructions",
    )

  def test_system_instructions_model_custom(self):
    """Verifies that CustomSystemInstructions sets custom on the proto."""
    strategy = self._make_strategy(
        system_instructions=types.CustomSystemInstructions(
            text="Override everything."
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.custom.part[0].text, "Override everything."
    )

  def test_system_instructions_model_templated(self):
    """Verifies that TemplatedSystemInstructions sets appended on the proto."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            identity="New Identity", sections=[section]
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.appended.custom_identity, "New Identity"
    )
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title, "extra"
    )

  def test_system_instructions_model_templated_only_identity(self):
    """Verifies that TemplatedSystemInstructions with only identity maps correctly."""
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            identity="Only Identity"
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(
        config.system_instructions.appended.custom_identity, "Only Identity"
    )
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 0
    )

  def test_system_instructions_model_templated_only_sections(self):
    """Verifies that TemplatedSystemInstructions with only sections maps correctly."""
    section = types.SystemInstructionSection(
        title="extra", content="More instructions"
    )
    strategy = self._make_strategy(
        system_instructions=types.TemplatedSystemInstructions(
            sections=[section]
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.system_instructions.appended.custom_identity, "")
    self.assertEqual(
        len(config.system_instructions.appended.appended_sections), 1
    )
    self.assertEqual(
        config.system_instructions.appended.appended_sections[0].title, "extra"
    )

  def test_system_instructions_none(self):
    """Verifies that no system_instructions field is set when not provided.

    Why: The harness should use its own defaults when no instructions are given.
    How: Build with system_instructions=None and assert no proto field is set.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertFalse(config.HasField("system_instructions"))

  def test_workspaces_to_proto(self):
    """Verifies workspace paths translate to Workspace protos correctly.

    Why: The harness uses a structured Workspace proto with FilesystemWorkspace;
    plain strings must be wrapped correctly.
    How: Pass two paths, build proto, and assert each workspace directory.
    """
    strategy = self._make_strategy(
        workspaces=["/home/user/project", "/tmp/scratch"]
    )
    config = strategy._build_harness_config()
    self.assertEqual(len(config.workspaces), 2)
    self.assertEqual(
        config.workspaces[0].filesystem_workspace.directory,
        "/home/user/project",
    )
    self.assertEqual(
        config.workspaces[1].filesystem_workspace.directory,
        "/tmp/scratch",
    )

  def test_workspaces_none(self):
    """Verifies that no workspaces are set when not provided.

    Why: The harness should not receive spurious workspace entries.
    How: Build with default workspaces=None and assert empty repeated field.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertEqual(len(config.workspaces), 0)

  def test_empty_workspaces_list(self):
    """Verifies that an empty list produces an empty repeated field.

    Why: workspaces=[] is a valid explicit choice meaning "no workspaces",
    distinct from None (which also means no workspaces but is implicit).
    How: Pass empty list and assert empty repeated field.
    """
    strategy = self._make_strategy(workspaces=[])
    config = strategy._build_harness_config()
    self.assertEqual(len(config.workspaces), 0)

  def test_skills_paths_to_proto(self):
    """Verifies skills_paths translate directly to the proto repeated field.

    Why: Skills paths are simple strings that map 1:1 to the proto field.
    How: Pass a list and assert proto field contents.
    """
    strategy = self._make_strategy(skills_paths=["/skills/a", "/skills/b"])
    config = strategy._build_harness_config()
    self.assertEqual(list(config.skills_paths), ["/skills/a", "/skills/b"])

  def test_capabilities_config_disabled_tools(self):
    """Verifies that disabling tools produces the correct proto.

    Why: Each BuiltinTool with a proto toggle should map to its config field.
    How: Disable RUN_COMMAND and ASK_QUESTION and assert each sub-proto's
    enabled field, plus check that other tools remain enabled.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            disabled_tools=[
                types.BuiltinTools.RUN_COMMAND,
                types.BuiltinTools.ASK_QUESTION,
                types.BuiltinTools.GENERATE_IMAGE,
            ],
        )
    )
    config = strategy._build_harness_config()
    self.assertFalse(config.harness_side_tools.run_command.enabled)
    self.assertFalse(config.harness_side_tools.user_questions.enabled)
    self.assertFalse(config.harness_side_tools.generate_image.enabled)
    # Subagents are not in BuiltinTools; should still be enabled by default.
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    # Tools that were not disabled should still be enabled.
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertTrue(config.harness_side_tools.file_edit.enabled)
    self.assertTrue(config.harness_side_tools.view_file.enabled)
    self.assertTrue(config.harness_side_tools.write_to_file.enabled)
    self.assertTrue(config.harness_side_tools.grep_search.enabled)
    self.assertTrue(config.harness_side_tools.list_dir.enabled)

  def test_capabilities_config_enabled_tools(self):
    """Verifies that enabled_tools allowlist excludes non-listed tools.

    Why: When an explicit allowlist is provided, only those tools should be
    active; all others should be disabled at the proto level.
    How: Enable only VIEW_FILE and assert all other tools are disabled.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(
            enabled_tools=[types.BuiltinTools.VIEW_FILE],
        )
    )
    config = strategy._build_harness_config()
    self.assertTrue(config.harness_side_tools.view_file.enabled)
    self.assertFalse(config.harness_side_tools.run_command.enabled)
    self.assertFalse(config.harness_side_tools.user_questions.enabled)
    self.assertFalse(config.harness_side_tools.find.enabled)
    self.assertFalse(config.harness_side_tools.file_edit.enabled)
    self.assertFalse(config.harness_side_tools.write_to_file.enabled)
    self.assertFalse(config.harness_side_tools.grep_search.enabled)
    self.assertFalse(config.harness_side_tools.list_dir.enabled)

  def test_capabilities_config_compaction_threshold(self):
    """Verifies compaction_threshold maps to HarnessConfig.compaction_threshold.

    Why: This controls context window compaction behavior in the harness.
    How: Set a threshold and assert it appears on the proto.
    """
    strategy = self._make_strategy(
        capabilities_config=types.CapabilitiesConfig(compaction_threshold=50000)
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.compaction_threshold, 50000)

  def test_capabilities_config_none_uses_defaults(self):
    """Verifies that capabilities_config=None produces default-enabled tools.

    Why: The most common case is no explicit CapabilitiesConfig; all tools
    should be enabled and compaction_threshold unset.
    How: Build with no capabilities_config and assert defaults.
    """
    strategy = self._make_strategy()
    config = strategy._build_harness_config()
    self.assertTrue(config.harness_side_tools.subagents.enabled)
    self.assertTrue(config.harness_side_tools.user_questions.enabled)
    self.assertTrue(config.harness_side_tools.run_command.enabled)
    self.assertTrue(config.harness_side_tools.find.enabled)
    self.assertEqual(config.compaction_threshold, 0)

  def test_cascade_id_passed_through(self):
    """Verifies that cascade_id appears in HarnessConfig.cascade_id.

    Why: cascade_id is used for session resumption; if it's lost, the
    harness creates a new session instead of resuming.
    How: Set cascade_id and assert it appears on the proto.
    """
    strategy = self._make_strategy(cascade_id="resume-123")
    config = strategy._build_harness_config()
    self.assertEqual(config.cascade_id, "resume-123")

  def test_gemini_config_thinking_level_set(self):
    """Verifies that thinking_level on ModelEntry maps to the proto field."""
    strategy = self._make_strategy(
        gemini_config=types.GeminiConfig(
            models=types.ModelConfig(
                default=types.ModelEntry(
                    name=types.DEFAULT_MODEL,
                    generation=types.GenerationConfig(
                        thinking_level=types.ThinkingLevel.HIGH,
                    ),
                ),
            ),
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.thinking_level, "high")

  def test_gemini_config_thinking_level_none_omitted(self):
    """Verifies that thinking_level=None leaves the proto field at its default."""
    strategy = self._make_strategy(gemini_config=types.GeminiConfig())
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.thinking_level, "")

  def test_gemini_config_thinking_level_all_values(self):
    """Verifies all ThinkingLevel enum values produce correct proto strings."""
    for level in types.ThinkingLevel:
      strategy = self._make_strategy(
          gemini_config=types.GeminiConfig(
              models=types.ModelConfig(
                  default=types.ModelEntry(
                      name=types.DEFAULT_MODEL,
                      generation=types.GenerationConfig(
                          thinking_level=level,
                      ),
                  ),
              ),
          )
      )
      config = strategy._build_harness_config()
      self.assertEqual(
          config.gemini_config.thinking_level,
          level.value,
          f"ThinkingLevel.{level.name} should produce proto string"
          f" '{level.value}'",
      )

  def test_per_model_api_key_takes_priority(self):
    """Verifies that a per-model API key overrides the shared GeminiConfig key."""
    strategy = self._make_strategy(
        gemini_config=types.GeminiConfig(
            api_key="shared-key",
            models=types.ModelConfig(
                default=types.ModelEntry(
                    name=types.DEFAULT_MODEL,
                    api_key="per-model-key",
                ),
            ),
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.api_key, "per-model-key")

  def test_shared_api_key_used_when_per_model_is_none(self):
    """Verifies that the shared GeminiConfig api_key is used as fallback."""
    strategy = self._make_strategy(
        gemini_config=types.GeminiConfig(
            api_key="shared-key",
            models=types.ModelConfig(
                default=types.ModelEntry(name=types.DEFAULT_MODEL),
            ),
        )
    )
    config = strategy._build_harness_config()
    self.assertEqual(config.gemini_config.api_key, "shared-key")


class LocalConnectionStrategyApiKeyTest(unittest.IsolatedAsyncioTestCase):
  """Tests for API key validation in LocalConnectionStrategy."""

  def _make_strategy(self, **kwargs):
    defaults = {"binary_path": "/fake/binary"}
    defaults.update(kwargs)
    return local_connection.LocalConnectionStrategy(**defaults)

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_without_api_key(self):
    """Verifies entry raises when no API key is available.

    Why: The Go localharness binary silently returns empty responses when no
    API key is provided. An explicit error at startup is much more actionable.
    How: Create a strategy with no api_key and no GEMINI_API_KEY env var and
    assert AntigravityValidationError is raised.
    """
    strategy = self._make_strategy()
    with self.assertRaises(types.AntigravityValidationError) as ctx:
      async with strategy:
        pass
    self.assertIn("API key", str(ctx.exception))

  @mock.patch.dict("os.environ", {}, clear=True)
  async def test_raises_with_empty_gemini_config(self):
    """Verifies entry raises when GeminiConfig has no api_key and env is unset.

    Why: GeminiConfig() defaults api_key to None. The check must not be
    fooled by the presence of a GeminiConfig object with no key.
    """
    strategy = self._make_strategy(gemini_config=types.GeminiConfig())
    with self.assertRaises(types.AntigravityValidationError):
      async with strategy:
        pass

  @mock.patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_env_var_api_key(
      self, mock_popen
  ):
    """Verifies entry does not raise when GEMINI_API_KEY env var is set.

    Why: The env var fallback is the most common path for 3P developers.
    How: Set GEMINI_API_KEY, enter the context manager, and verify it proceeds
    past the validation check (it will fail later at subprocess I/O, which is
    expected).

    Args:
      mock_popen: Mocked subprocess.Popen to prevent actual process launch.
    """
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc
    strategy = self._make_strategy()
    # Should not raise AntigravityValidationError; it will raise RuntimeError
    # from the subprocess read failure, which proves we passed the check.
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch("subprocess.Popen")
  async def test_accepts_gemini_config_api_key(
      self, mock_popen
  ):
    """Verifies entry does not raise when GeminiConfig.api_key is set.

    Why: Explicit API key in config is the recommended path.
    How: Set api_key in GeminiConfig, enter the context manager, and verify
    it proceeds past the validation check.

    Args:
      mock_popen: Mocked subprocess.Popen to prevent actual process launch.
    """
    mock_proc = mock.MagicMock()
    mock_proc.stdin = mock.MagicMock()
    mock_proc.stdout = mock.MagicMock()
    mock_proc.stderr = mock.MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc
    strategy = self._make_strategy(
        gemini_config=types.GeminiConfig(api_key="explicit-key")
    )
    with self.assertRaises(RuntimeError):
      async with strategy:
        pass


class GetDefaultBinaryPathTest(unittest.TestCase):

  @mock.patch.dict("os.environ", {"ANTIGRAVITY_HARNESS_PATH": "/env/path"})
  def test_returns_env_path(self):
    path = local_connection._get_default_binary_path()
    self.assertEqual(path, "/env/path")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch.object(local_connection, "resources", None)
  @mock.patch("importlib.resources.files")
  @mock.patch("os.path.exists")
  def test_returns_external_wheel_path(self, mock_exists, mock_files):
    mock_path = mock.MagicMock()
    mock_path.joinpath.return_value.__str__.return_value = "/wheel/path"
    mock_files.return_value = mock_path
    mock_exists.return_value = True

    path = local_connection._get_default_binary_path()
    self.assertEqual(path, "/wheel/path")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch.object(local_connection, "resources", None)
  @mock.patch("importlib.resources.files")
  @mock.patch("shutil.which")
  def test_returns_system_path(self, mock_which, mock_files):
    mock_files.side_effect = ImportError
    mock_which.return_value = "/system/path"

    path = local_connection._get_default_binary_path()
    self.assertEqual(path, "/system/path")
    mock_which.assert_called_once_with("localharness")

  @mock.patch.dict("os.environ", {}, clear=True)
  @mock.patch.object(local_connection, "resources", None)
  @mock.patch("importlib.resources.files")
  @mock.patch("shutil.which")
  def test_raises_when_not_found(self, mock_which, mock_files):
    mock_files.side_effect = ImportError
    mock_which.return_value = None

    with self.assertRaises(RuntimeError) as ctx:
      local_connection._get_default_binary_path()
    self.assertIn(
        "Could not find default localharness binary", str(ctx.exception)
    )


class LocalConnectionSessionHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for session start/end hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()
    self.tool_runner = tool_runner.ToolRunner()

  async def test_session_start_hook_dispatched_on_init(self):
    """Verifies OnSessionStartHook fires when LocalConnection is created."""
    called = []

    class SessionStartHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        called.append("started")

    hr = hook_runner.HookRunner()
    hr.on_session_start_hooks.append(SessionStartHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await asyncio.sleep(0.1)
    self.assertEqual(called, ["started"])

  async def test_session_end_hook_dispatched_on_disconnect(self):
    """Verifies OnSessionEndHook fires when disconnect() is called."""
    called = []

    class SessionEndHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        called.append("ended")

    hr = hook_runner.HookRunner()
    hr.on_session_end_hooks.append(SessionEndHook())

    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    await conn.disconnect()
    self.assertEqual(called, ["ended"])


class LocalConnectionPostTurnHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for post-turn hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()
    self.tool_runner = tool_runner.ToolRunner()

  async def test_post_turn_hook_dispatched_on_final_step(self):
    """Verifies PostTurnHook fires when a terminal model step is received."""
    captured = []

    class PostTurnHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)

    hr = hook_runner.HookRunner()
    hr.post_turn_hooks.append(PostTurnHook())

    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=self.tool_runner,
        hook_runner=hr,
    )

    # Simulate a send to create turn context.
    await conn.send("hello")

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="test_traj",
            trajectory_id="test_traj",
            step_index=1,
            text="Final answer",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )

    await self.mock_ws.put_event(event)

    # The real harness sends STATE_IDLE after the final step. The
    # connection waits for this before returning from receive_steps().
    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="test_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )
    await self.mock_ws.put_event(idle_event)

    # Drain receive_steps to trigger terminal detection + hook dispatch.
    steps = []
    async for step in conn.receive_steps():
      steps.append(step)

    self.assertEqual(len(steps), 1)
    self.assertEqual(captured, ["Final answer"])


class LocalConnectionCompactionHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for compaction hook dispatch."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()

  async def test_compaction_step_dispatches_hook(self):
    """Verifies OnCompactionHook fires when a compaction step is received."""
    captured = []

    class CompactionHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)

    hr = hook_runner.HookRunner()
    hr.on_compaction_hooks.append(CompactionHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            step_index=1,
            text="Context compaction",
            state=localharness_pb2.StepUpdate.STATE_DONE,
            source=localharness_pb2.StepUpdate.SOURCE_SYSTEM,
            target=localharness_pb2.StepUpdate.TARGET_USER,
            compaction=localharness_pb2.ActionCompaction(),
        )
    )

    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    self.assertEqual(len(captured), 1)
    self.assertIsInstance(captured[0], local_connection.LocalConnectionStep)
    self.assertEqual(captured[0].content, "Context compaction")


class LocalConnectionSubagentHookTest(unittest.IsolatedAsyncioTestCase):
  """Tests for subagent hook dispatch via tool hooks.

  Subagent invocations are treated as tool calls with the name
  START_SUBAGENT. Pre- and post-tool-call hooks receive the subagent
  data using standard tool hook dispatch.
  """

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()

  async def test_invoke_subagent_step_classified_as_tool_call(self):
    """Verifies invoke_subagent steps are classified as TOOL_CALL."""
    hr = hook_runner.HookRunner()

    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )

    await conn.send("hello")

    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=1,
            text="Invoking subagent",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            invoke_subagent=localharness_pb2.ActionInvokeSubagent(),
        )
    )

    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    # Drain the queue to inspect the step.
    step = conn._step_queue.get_nowait()
    self.assertEqual(step.type, types.StepType.TOOL_CALL)

  async def test_post_tool_hook_on_subagent_trajectory_idle(self):
    """Verifies post-tool-call hook fires when a non-main trajectory goes idle."""
    captured = []

    class PostToolHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured.append(data)

    hr = hook_runner.HookRunner()
    hr.post_tool_call_hooks.append(PostToolHook())

    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )

    # Establish the cascade_id via a parent trajectory step
    # (cascade_id == trajectory_id).
    main_step = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main_traj",
            step_index=0,
            trajectory_id="main_traj",
            text="Main step",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )
    await self.mock_ws.put_event(main_step)
    await asyncio.sleep(0.1)

    self.assertEqual(conn._cascade_id, "main_traj")

    # Simulate a subagent model step with text (may arrive as ACTIVE first).
    sub_active = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main_traj",
            trajectory_id="sub_traj",
            step_index=0,
            text="Here is a poem about nature.",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
            target=localharness_pb2.StepUpdate.TARGET_USER,
        )
    )
    await self.mock_ws.put_event(sub_active)
    await asyncio.sleep(0.1)

    # Now simulate the subagent trajectory going idle.
    idle_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="sub_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )
    await self.mock_ws.put_event(idle_event)
    await asyncio.sleep(0.1)

    self.assertEqual(len(captured), 1)
    self.assertIsInstance(captured[0], types.ToolResult)
    self.assertEqual(captured[0].name, types.BuiltinTools.START_SUBAGENT.value)
    # The result should contain the subagent's final response, not just
    # the trajectory ID.
    self.assertEqual(captured[0].result, "Here is a poem about nature.")

    # Main trajectory idle should NOT fire post-tool hook for subagent.
    main_idle = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="main_traj",
            state=localharness_pb2.TrajectoryStateUpdate.STATE_IDLE,
        )
    )
    await self.mock_ws.put_event(main_idle)
    await asyncio.sleep(0.1)

    # Still only 1 capture.
    self.assertEqual(len(captured), 1)

  async def test_subagent_running_tracked(self):
    """Verifies STATE_RUNNING adds subagent to active set."""
    hr = hook_runner.HookRunner()
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )

    # Establish cascade_id.
    main_step = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=0,
            text="hi",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )
    await self.mock_ws.put_event(main_step)
    await asyncio.sleep(0.1)

    # Subagent starts running.
    running_event = localharness_pb2.OutputEvent(
        trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
            trajectory_id="sub_1",
            state=(localharness_pb2.TrajectoryStateUpdate.STATE_RUNNING),
        )
    )
    await self.mock_ws.put_event(running_event)
    await asyncio.sleep(0.1)

    self.assertIn("sub_1", conn._active_subagent_ids)

  async def test_connection_waits_for_subagents_before_idle(self):
    """Verifies receive_steps blocks until subagents complete."""
    hr = hook_runner.HookRunner()
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )

    await conn.send("hello")

    # Establish cascade_id + a step.
    main_step = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            cascade_id="main",
            trajectory_id="main",
            step_index=0,
            text="response",
            state=localharness_pb2.StepUpdate.STATE_ACTIVE,
            source=localharness_pb2.StepUpdate.SOURCE_MODEL,
        )
    )
    await self.mock_ws.put_event(main_step)

    # Subagent starts.
    await self.mock_ws.put_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="sub_1",
                state=(localharness_pb2.TrajectoryStateUpdate.STATE_RUNNING),
            )
        )
    )

    # Parent goes idle, but subagent still running.
    await self.mock_ws.put_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="main",
                state=(localharness_pb2.TrajectoryStateUpdate.STATE_IDLE),
            )
        )
    )
    await asyncio.sleep(0.1)

    # _is_idle should NOT be set yet.
    self.assertFalse(conn._is_idle.is_set())

    # Subagent completes.
    await self.mock_ws.put_event(
        localharness_pb2.OutputEvent(
            trajectory_state_update=localharness_pb2.TrajectoryStateUpdate(
                trajectory_id="sub_1",
                state=(localharness_pb2.TrajectoryStateUpdate.STATE_IDLE),
            )
        )
    )
    await asyncio.sleep(0.1)

    # NOW idle should be set.
    self.assertTrue(conn._is_idle.is_set())

  async def test_send_resets_subagent_tracking(self):
    """Verifies send() clears subagent tracking state."""
    hr = hook_runner.HookRunner()
    conn = local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )

    # Pollute tracking state.
    conn._active_subagent_ids.add("leftover")
    conn._subagent_responses["leftover"] = "stale response"
    conn._parent_idle = True

    await conn.send("new turn")

    self.assertEqual(conn._active_subagent_ids, set())
    self.assertEqual(conn._subagent_responses, {})
    self.assertFalse(conn._parent_idle)
    self.assertFalse(conn._is_idle.is_set())


class LocalConnectionToolCallHooksTest(unittest.IsolatedAsyncioTestCase):
  """Tests for post-tool-call and on-tool-error hooks."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()

  async def test_post_tool_call_hook_dispatched(self):
    """Verifies PostToolCallHook fires after successful tool execution."""
    captured_results = []

    class PostToolHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        captured_results.append(data)

    tr = tool_runner.ToolRunner()

    async def echo_handler(**kwargs):
      return json.dumps({"echo": kwargs})

    tr.register(echo_handler, "echo_tool")

    hr = hook_runner.HookRunner()
    hr.post_tool_call_hooks.append(PostToolHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=tr,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_1",
            name="echo_tool",
            arguments_json='{"msg": "hi"}',
        )
    )

    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    self.assertEqual(len(captured_results), 1)

  async def test_on_tool_error_hook_with_recovery(self):
    """Verifies OnToolErrorHook can provide recovery values on tool failure."""

    class RecoveringErrorHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        return "recovered_value"

    tr = tool_runner.ToolRunner()

    async def failing_handler(**kwargs):
      raise RuntimeError("Intentional failure")

    tr.register(failing_handler, "failing_tool")

    hr = hook_runner.HookRunner()
    hr.on_tool_error_hooks.append(RecoveringErrorHook())

    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        tool_runner=tr,
        hook_runner=hr,
    )

    event = localharness_pb2.OutputEvent(
        tool_call=localharness_pb2.ToolCall(
            id="call_fail",
            name="failing_tool",
            arguments_json="{}",
        )
    )

    await self.mock_ws.put_event(event)
    await asyncio.sleep(0.1)

    # The recovery value should have been sent back.
    self.assertEqual(len(self.mock_ws.sent_messages), 1)
    sent_data = json.loads(self.mock_ws.sent_messages[0])
    self.assertIn("toolResponse", sent_data)
    self.assertIn("recovered_value", sent_data["toolResponse"]["responseJson"])


class LocalConnectionModelHooksErrorTest(unittest.TestCase):
  """Verifies that model hooks still raise NotImplementedError."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()

  def test_pre_model_call_hook_raises(self):
    """Pre-model-call hooks are not supported by LocalConnection."""

    class DummyHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        pass

    hr = hook_runner.HookRunner()
    hr.pre_model_call_hooks.append(DummyHook())

    with self.assertRaises(NotImplementedError):
      local_connection.LocalConnection(
          process=self.mock_process,
          ws=self.mock_ws,
          hook_runner=hr,
      )

  def test_post_model_call_hook_raises(self):
    """Post-model-call hooks are not supported by LocalConnection."""

    class DummyHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        pass

    hr = hook_runner.HookRunner()
    hr.post_model_call_hooks.append(DummyHook())

    with self.assertRaises(NotImplementedError):
      local_connection.LocalConnection(
          process=self.mock_process,
          ws=self.mock_ws,
          hook_runner=hr,
      )

  def test_on_model_chunk_hook_raises(self):
    """On-model-chunk hooks are not supported by LocalConnection."""

    class DummyHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        pass

    hr = hook_runner.HookRunner()
    hr.on_model_chunk_hooks.append(DummyHook())

    with self.assertRaises(NotImplementedError):
      local_connection.LocalConnection(
          process=self.mock_process,
          ws=self.mock_ws,
          hook_runner=hr,
      )


class LocalConnectionHookAcceptanceTest(unittest.IsolatedAsyncioTestCase):
  """Verifies that previously-unsupported hooks are now accepted."""

  def setUp(self):
    super().setUp()
    self.mock_process = mock.MagicMock()
    self.mock_ws = FakeWebSocket()

  async def test_subagent_tool_hooks_accepted(self):
    """Subagent lifecycle is handled by tool hooks; no special subagent lists."""

    class DummyHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        return hooks_base.HookResult(allow=True)

    hr = hook_runner.HookRunner()
    hr.pre_tool_call_decide_hooks.append(DummyHook())

    # Should NOT raise.
    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )

  async def test_compaction_hooks_no_longer_raise(self):
    """Compaction hooks should be accepted now."""

    class DummyHook:

      async def run(self, context, data):  # pylint: disable=unused-argument
        pass

    hr = hook_runner.HookRunner()
    hr.on_compaction_hooks.append(DummyHook())

    # Should NOT raise.
    local_connection.LocalConnection(
        process=self.mock_process,
        ws=self.mock_ws,
        hook_runner=hr,
    )




if __name__ == "__main__":
  absltest.main()
