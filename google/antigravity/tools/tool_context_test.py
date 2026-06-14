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

"""Tests for tool_context module."""

from unittest import mock

from absl.testing import absltest

from google.antigravity.conversation import conversation as conversation_module
from google.antigravity.tools import tool_context


def _make_mock_conversation(**overrides) -> mock.MagicMock:
  """Creates a mock Conversation with sensible defaults.

  Args:
    **overrides: Attribute overrides for the mock.

  Returns:
    A MagicMock with spec=Conversation.
  """
  conv = mock.MagicMock(spec=conversation_module.Conversation)
  conv.conversation_id = "test-conv-123"
  for k, v in overrides.items():
    setattr(conv, k, v)
  return conv


class ToolContextPropertyTest(absltest.TestCase):
  """Validates ToolContext property accessors.

  Ensures that conversation_id delegates correctly to
  the underlying Conversation.
  """

  def test_conversation_id(self):
    """Verifies conversation_id delegates to Conversation.conversation_id.

    What: Checks that the property returns the conversation's ID.
    Why: ToolContext must expose identity for tool-level state management.
    How: Creates a ToolContext with a mock conversation and asserts equality.
    """
    conv = _make_mock_conversation(conversation_id="abc-123")
    ctx = tool_context.ToolContext(conv)
    self.assertEqual(ctx.conversation_id, "abc-123")


class ToolContextStateTest(absltest.TestCase):
  """Validates per-conversation state management.

  Ensures that get_state/set_state provide a simple key-value store
  scoped to the ToolContext lifetime.
  """

  def test_get_state_missing_returns_default(self):
    """Verifies get_state returns the default for missing keys.

    What: Checks the default return behavior.
    Why: Tools should not crash when accessing unset state.
    How: Calls get_state for an absent key and asserts the default.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    self.assertIsNone(ctx.get_state("missing"))
    self.assertEqual(ctx.get_state("missing", "fallback"), "fallback")

  def test_set_and_get_state(self):
    """Verifies set_state stores values retrievable by get_state.

    What: Checks round-trip state persistence.
    Why: Core state store functionality must work correctly.
    How: Sets a value and asserts it's returned by get_state.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    ctx.set_state("counter", 42)
    self.assertEqual(ctx.get_state("counter"), 42)

  def test_set_state_overwrites(self):
    """Verifies set_state overwrites existing values.

    What: Checks that re-setting a key updates the stored value.
    Why: State must be mutable for accumulating tool results.
    How: Sets a key twice and asserts the latest value is returned.
    """
    conv = _make_mock_conversation()
    ctx = tool_context.ToolContext(conv)
    ctx.set_state("key", "old")
    ctx.set_state("key", "new")
    self.assertEqual(ctx.get_state("key"), "new")

  def test_state_isolation_between_instances(self):
    """Verifies that separate ToolContext instances have independent state.

    What: Checks that state does not leak between instances.
    Why: Each session must have its own state namespace.
    How: Creates two contexts, sets state on one, and asserts the other
    does not see it.
    """
    conv = _make_mock_conversation()
    ctx1 = tool_context.ToolContext(conv)
    ctx2 = tool_context.ToolContext(conv)
    ctx1.set_state("shared_key", "value1")
    self.assertIsNone(ctx2.get_state("shared_key"))


if __name__ == "__main__":
  absltest.main()
