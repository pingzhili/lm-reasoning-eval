# MIT License

# Copyright (c) 2025 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest
from unittest.mock import Mock

from lighteval.models.model_output import ModelResponse
from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig
from lighteval.tasks.requests import Doc


class TestSelfJudgeThinking(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config with self_judge_thinking enabled
        self.config = VLLMModelConfig(
            model_name="test-model",
            enable_thinking=True,
            self_judge_thinking=True,
            data_parallel_size=1,
            tensor_parallel_size=1,
        )

        # Create test docs
        self.test_docs = [
            Doc(
                query="What is 2 + 2?",
                choices=["3", "4", "5"],
                gold_index=1,  # 4 is the correct answer
                task_name="math",
                instruction="",
                unconditioned_query="",
                num_asked_few_shots=0,
                num_effective_few_shots=0,
                original_query="What is 2 + 2?",
                id="test_1",
                num_samples=1,
                generation_size=100,
                stop_sequences=None,
            ),
            Doc(
                query="Solve the differential equation: dy/dx = 2x + 3y",
                choices=["solution"],
                gold_index=0,
                task_name="math",
                instruction="",
                unconditioned_query="",
                num_asked_few_shots=0,
                num_effective_few_shots=0,
                original_query="Solve the differential equation: dy/dx = 2x + 3y",
                id="test_2",
                num_samples=1,
                generation_size=200,
                stop_sequences=None,
            ),
        ]

    def test_create_judge_doc(self):
        """Test that judge docs are created correctly."""
        # Create a mock model instance
        model = VLLMModel.__new__(VLLMModel)
        model._config = self.config

        # Test creating judge doc
        judge_doc = model._create_judge_doc(self.test_docs[0])

        # Verify judge doc properties
        self.assertIn("determine if it requires step-by-step thinking", judge_doc.query)
        self.assertIn("What is 2 + 2?", judge_doc.query)
        self.assertEqual(judge_doc.choices, ["YES", "NO"])
        self.assertEqual(judge_doc.num_samples, 1)
        self.assertEqual(judge_doc.generation_size, 10)

    def test_parse_thinking_judgment(self):
        """Test parsing of thinking judgment responses."""
        model = VLLMModel.__new__(VLLMModel)
        model._config = self.config

        # Test various response formats
        self.assertTrue(model._parse_thinking_judgment("YES"))
        self.assertTrue(model._parse_thinking_judgment("yes"))
        self.assertTrue(model._parse_thinking_judgment("  YES  "))
        self.assertTrue(model._parse_thinking_judgment("YES, this requires thinking"))
        self.assertTrue(model._parse_thinking_judgment(["YES"]))

        self.assertFalse(model._parse_thinking_judgment("NO"))
        self.assertFalse(model._parse_thinking_judgment("no"))
        self.assertFalse(model._parse_thinking_judgment(""))
        self.assertFalse(model._parse_thinking_judgment("Maybe"))
        self.assertFalse(model._parse_thinking_judgment([]))

    def test_greedy_until_self_judge_disabled(self):
        """Test that regular greedy_until is used when self_judge_thinking is disabled."""
        # Setup config with self_judge_thinking disabled
        config = VLLMModelConfig(
            model_name="test-model",
            enable_thinking=True,
            self_judge_thinking=False,
        )

        # Create model mock
        model = VLLMModel.__new__(VLLMModel)
        model._config = config
        model.greedy_until = Mock(
            return_value=[ModelResponse(text=["4"], logprobs=None, output_tokens=None, input_tokens=None)]
        )

        # Call greedy_until_self_judge
        results = model.greedy_until_self_judge(self.test_docs[:1])

        # Verify regular greedy_until was called
        model.greedy_until.assert_called_once_with(self.test_docs[:1])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, ["4"])

    def test_greedy_until_self_judge_workflow(self):
        """Test the complete self-judging workflow."""
        # Create model mock
        model = VLLMModel.__new__(VLLMModel)
        model._config = self.config
        model.prompt_manager = Mock()
        model.prompt_manager.enable_thinking = True

        # Mock logger to avoid import issues
        import logging

        model.logger = logging.getLogger(__name__)

        # Mock responses for judgment and actual generation
        judge_responses = [
            ModelResponse(text=["NO"], logprobs=None, output_tokens=None, input_tokens=None),  # Simple question
            ModelResponse(text=["YES"], logprobs=None, output_tokens=None, input_tokens=None),  # Complex question
        ]

        actual_responses = [
            ModelResponse(text=["4"], logprobs=None, output_tokens=None, input_tokens=None),
            ModelResponse(text=["The solution is..."], logprobs=None, output_tokens=None, input_tokens=None),
        ]

        # Mock greedy_until to return different responses based on call count
        call_count = 0

        def mock_greedy_until(docs):
            nonlocal call_count
            if call_count == 0:  # First call is for all judgments at once
                result = judge_responses[: len(docs)]
            else:  # Next calls are for actual generation (one per doc)
                result = [actual_responses[call_count - 1]]
            call_count += 1
            return result

        model.greedy_until = Mock(side_effect=mock_greedy_until)

        # Call greedy_until_self_judge
        results = model.greedy_until_self_judge(self.test_docs)

        # Verify the workflow
        self.assertEqual(model.greedy_until.call_count, 3)  # 1 for all judgments, 2 for individual generations
        self.assertEqual(len(results), 2)

        # Check results were generated
        self.assertEqual(results[0].text, ["4"])  # Simple question response
        self.assertEqual(results[1].text, ["The solution is..."])  # Complex question response

    def test_thinking_judge_template(self):
        """Test that the thinking judge template is properly formatted."""
        model = VLLMModel.__new__(VLLMModel)

        # Check template exists and has required placeholders
        self.assertIn("{question}", model.THINKING_JUDGE_TEMPLATE)
        self.assertIn("YES", model.THINKING_JUDGE_TEMPLATE)
        self.assertIn("NO", model.THINKING_JUDGE_TEMPLATE)

        # Test formatting
        test_question = "What is the meaning of life?"
        formatted = model.THINKING_JUDGE_TEMPLATE.format(question=test_question)
        self.assertIn(test_question, formatted)

    def test_prompt_manager_thinking_state_restoration(self):
        """Test that prompt manager's enable_thinking state is properly restored."""
        # Create model mock
        model = VLLMModel.__new__(VLLMModel)
        model._config = self.config
        model.prompt_manager = Mock()
        model.prompt_manager.enable_thinking = True
        original_thinking_state = model.prompt_manager.enable_thinking

        # Mock logger
        import logging

        model.logger = logging.getLogger(__name__)

        # Mock greedy_until to track thinking state
        thinking_states = []

        def track_thinking_state(docs):
            thinking_states.append(model.prompt_manager.enable_thinking)
            return [ModelResponse(text=["response"], logprobs=None, output_tokens=None, input_tokens=None)]

        model.greedy_until = Mock(side_effect=track_thinking_state)

        # Call greedy_until_self_judge
        try:
            model.greedy_until_self_judge(self.test_docs[:1])
        except Exception:
            # It's OK if it fails, we're just testing state restoration
            pass

        # Verify thinking state was restored
        self.assertEqual(model.prompt_manager.enable_thinking, original_thinking_state)

    def test_judgment_storage(self):
        """Test that judgments are correctly stored in the model."""
        # Create model with self_judge_thinking enabled
        model = VLLMModel.__new__(VLLMModel)
        model._config = self.config
        model._self_judgments = {}
        model.prompt_manager = Mock()
        model.prompt_manager.enable_thinking = True

        # Mock logger
        import logging

        model.logger = logging.getLogger(__name__)

        # Mock greedy_until to return different responses
        def mock_greedy_until(docs):
            if "determine if it requires step-by-step thinking" in docs[0].query:
                # Judgment calls
                return [
                    ModelResponse(text=["NO"], logprobs=None, output_tokens=None, input_tokens=None),
                    ModelResponse(text=["YES"], logprobs=None, output_tokens=None, input_tokens=None),
                ]
            else:
                # Actual generation
                return [ModelResponse(text=["answer"], logprobs=None, output_tokens=None, input_tokens=None)]

        model.greedy_until = Mock(side_effect=mock_greedy_until)

        # Call greedy_until_self_judge
        model.greedy_until_self_judge(self.test_docs)

        # Verify judgments were stored
        self.assertIn("test_1", model._self_judgments)
        self.assertIn("test_2", model._self_judgments)
        self.assertFalse(model._self_judgments["test_1"])  # "NO" -> False
        self.assertTrue(model._self_judgments["test_2"])  # "YES" -> True

    async def test_async_greedy_until_self_judge_workflow(self):
        """Test the complete async self-judging workflow."""
        # Import AsyncVLLMModel
        from lighteval.models.vllm.vllm_model import AsyncVLLMModel

        # Create model mock
        model = AsyncVLLMModel.__new__(AsyncVLLMModel)
        model._config = self.config
        model.prompt_manager = Mock()
        model.prompt_manager.enable_thinking = True

        # Mock logger to avoid import issues
        import logging

        model.logger = logging.getLogger(__name__)

        # Mock responses for judgment and actual generation
        judge_responses = [
            ModelResponse(text=["NO"], logprobs=None, output_tokens=None, input_tokens=None),  # Simple question
            ModelResponse(text=["YES"], logprobs=None, output_tokens=None, input_tokens=None),  # Complex question
        ]

        actual_responses = [
            ModelResponse(text=["4"], logprobs=None, output_tokens=None, input_tokens=None),
            ModelResponse(text=["The solution is..."], logprobs=None, output_tokens=None, input_tokens=None),
        ]

        # Mock async greedy_until
        async def mock_async_greedy_until(docs):
            if len(docs) == 2 and all("determine if it requires step-by-step thinking" in doc.query for doc in docs):
                # This is the judgment call
                return judge_responses[: len(docs)]
            else:
                # This is the actual generation call
                if "2 + 2" in docs[0].query:
                    return [actual_responses[0]]
                else:
                    return [actual_responses[1]]

        model.greedy_until = mock_async_greedy_until

        # Call greedy_until_self_judge
        import asyncio

        results = asyncio.run(model.greedy_until_self_judge(self.test_docs))

        # Verify the workflow
        self.assertEqual(len(results), 2)

        # Check results were generated
        self.assertEqual(results[0].text, ["4"])  # Simple question response
        self.assertEqual(results[1].text, ["The solution is..."])  # Complex question response

        # Verify judgments were stored
        self.assertIn("test_1", model._self_judgments)
        self.assertIn("test_2", model._self_judgments)
        self.assertFalse(model._self_judgments["test_1"])  # "NO" -> False
        self.assertTrue(model._self_judgments["test_2"])  # "YES" -> True


if __name__ == "__main__":
    unittest.main()
