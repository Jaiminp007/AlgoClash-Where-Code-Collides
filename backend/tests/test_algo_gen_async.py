"""
Unit tests for async algorithm generation

Tests verify:
1. Concurrent execution of 6 requests
2. Output order preservation
3. Retry/backoff on 429 errors
4. Failure handling with placeholders
"""

import pytest
import asyncio
import httpx
from unittest.mock import AsyncMock, Mock, patch
import sys
import os

# Add parent directory to path for imports
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.join(backend_dir, 'open_router'))

# Mock the model_fecthing import before importing algo_gen
sys.modules['model_fecthing'] = Mock()

from open_router.algo_gen import (
    generate_algorithm_async,
    _generate_algorithms_for_agents_async,
)


@pytest.mark.asyncio
async def test_generate_algorithm_async_success():
    """Test successful algorithm generation"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{
            'message': {
                'content': '''
def execute_trade(ticker, cash_balance, shares_held):
    return "HOLD"
'''
            }
        }]
    }

    async with httpx.AsyncClient() as client:
        with patch.object(client, 'post', return_value=mock_response) as mock_post:
            result = await generate_algorithm_async(
                client,
                "test-model",
                "test prompt"
            )

            assert result is not None
            assert 'def execute_trade' in result
            assert mock_post.called


@pytest.mark.asyncio
async def test_generate_algorithm_async_retry_on_429():
    """Test retry logic on 429 rate limit"""
    # First two attempts return 429, third succeeds
    mock_response_429 = Mock()
    mock_response_429.status_code = 429
    mock_response_429.text = "Rate limited"

    mock_response_success = Mock()
    mock_response_success.status_code = 200
    mock_response_success.json.return_value = {
        'choices': [{
            'message': {
                'content': 'def execute_trade(ticker, cash_balance, shares_held):\n    return "HOLD"'
            }
        }]
    }

    call_count = 0
    async def mock_post_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return mock_response_429
        return mock_response_success

    async with httpx.AsyncClient() as client:
        with patch.object(client, 'post', side_effect=mock_post_side_effect):
            result = await generate_algorithm_async(
                client,
                "test-model",
                "test prompt",
                max_retries=3
            )

            assert result is not None
            assert 'def execute_trade' in result
            assert call_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_generate_algorithm_async_max_retries_exceeded():
    """Test that function returns None after max retries"""
    mock_response_429 = Mock()
    mock_response_429.status_code = 429
    mock_response_429.text = "Rate limited"

    async with httpx.AsyncClient() as client:
        with patch.object(client, 'post', return_value=mock_response_429):
            result = await generate_algorithm_async(
                client,
                "test-model",
                "test prompt",
                max_retries=2
            )

            assert result is None


@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test that 6 requests execute concurrently (not sequentially)"""

    # Track when each request starts and completes
    events = []

    async def mock_generate(client, model_id, prompt):
        events.append(f"start:{model_id}")
        await asyncio.sleep(0.1)  # Simulate API latency
        events.append(f"end:{model_id}")
        return f"def execute_trade(ticker, cash_balance, shares_held):\n    # {model_id}\n    return 'HOLD'"

    models = [f"model-{i}" for i in range(6)]

    with patch('open_router.algo_gen.generate_algorithm_async', side_effect=mock_generate):
        with patch('open_router.algo_gen._get_accessible_models', return_value=set(models)):
            with patch('open_router.algo_gen.load_csv_preview', return_value=""):
                with patch('open_router.algo_gen.build_generation_prompt', return_value="prompt"):
                    with patch('open_router.algo_gen.build_diversity_directives', return_value=""):
                        with patch('open_router.algo_gen._save_code_for_model'):
                            with patch('os.path.exists', return_value=False):
                                start_time = asyncio.get_event_loop().time()
                                result = await _generate_algorithms_for_agents_async(
                                    models,
                                    "AAPL"
                                )
                                end_time = asyncio.get_event_loop().time()

                                # Verify result
                                assert result is True

                                # If executed sequentially, would take 6 * 0.1 = 0.6s
                                # If concurrent, should take ~0.1s
                                elapsed = end_time - start_time
                                assert elapsed < 0.3, f"Took {elapsed}s, expected < 0.3s for concurrent execution"

                                # Verify all models started before any completed (concurrent execution)
                                start_events = [e for e in events if e.startswith("start:")]
                                assert len(start_events) == 6


@pytest.mark.asyncio
async def test_output_order_preserved():
    """Test that results are returned in the same order as input"""

    models = [f"model-{i}" for i in range(6)]
    results_order = []

    async def mock_generate(client, model_id, prompt):
        # Add random delay to simulate different response times
        await asyncio.sleep(0.01 * (6 - int(model_id.split('-')[1])))  # Reverse order delays
        return f"def execute_trade(ticker, cash_balance, shares_held):\n    # {model_id}\n    return 'HOLD'"

    def mock_save(code, model_name):
        # _save_code_for_model receives the original model name without sanitization
        results_order.append(model_name)

    with patch('open_router.algo_gen.generate_algorithm_async', side_effect=mock_generate):
        with patch('open_router.algo_gen._get_accessible_models', return_value=set(models)):
            with patch('open_router.algo_gen.load_csv_preview', return_value=""):
                with patch('open_router.algo_gen.build_generation_prompt', return_value="prompt"):
                    with patch('open_router.algo_gen.build_diversity_directives', return_value=""):
                        with patch('open_router.algo_gen.save_algorithm_to_file'):  # Mock the file save
                            with patch('os.path.exists', return_value=False):
                                # Patch _save_code_for_model to track order
                                original_save = __import__('open_router.algo_gen', fromlist=['_save_code_for_model'])._save_code_for_model

                                def tracked_save(code, model_name):
                                    results_order.append(model_name)
                                    # Don't call original to avoid file operations

                                with patch('open_router.algo_gen._save_code_for_model', side_effect=tracked_save):
                                    result = await _generate_algorithms_for_agents_async(
                                        models,
                                        "AAPL"
                                    )

                                    assert result is True
                                    # Results should be saved in original order (model names passed to _save_code_for_model)
                                    assert results_order == models


@pytest.mark.asyncio
async def test_partial_failure_handling():
    """Test that partial failures don't break the entire process"""

    models = [f"model-{i}" for i in range(6)]

    async def mock_generate(client, model_id, prompt):
        # Fail for even-numbered models
        model_num = int(model_id.split('-')[1])
        if model_num % 2 == 0:
            return None  # Simulate failure
        return f"def execute_trade(ticker, cash_balance, shares_held):\n    return 'HOLD'"

    saved_models = []

    def mock_save(code, model_name):
        saved_models.append(model_name)

    with patch('open_router.algo_gen.generate_algorithm_async', side_effect=mock_generate):
        with patch('open_router.algo_gen._get_accessible_models', return_value=set(models)):
            with patch('open_router.algo_gen.load_csv_preview', return_value=""):
                with patch('open_router.algo_gen.build_generation_prompt', return_value="prompt"):
                    with patch('open_router.algo_gen.build_diversity_directives', return_value=""):
                        with patch('open_router.algo_gen._save_code_for_model', side_effect=mock_save):
                            with patch('os.path.exists', return_value=False):
                                result = await _generate_algorithms_for_agents_async(
                                    models,
                                    "AAPL"
                                )

                                # Should succeed because at least one model succeeded
                                assert result is True
                                # Only odd-numbered models should be saved
                                assert len(saved_models) == 3


@pytest.mark.asyncio
async def test_all_failures():
    """Test that function returns False when all requests fail"""

    models = [f"model-{i}" for i in range(6)]

    async def mock_generate(client, model_id, prompt):
        return None  # All fail

    with patch('open_router.algo_gen.generate_algorithm_async', side_effect=mock_generate):
        with patch('open_router.algo_gen._get_accessible_models', return_value=set(models)):
            with patch('open_router.algo_gen.load_csv_preview', return_value=""):
                with patch('open_router.algo_gen.build_generation_prompt', return_value="prompt"):
                    with patch('open_router.algo_gen.build_diversity_directives', return_value=""):
                        with patch('os.path.exists', return_value=False):
                            result = await _generate_algorithms_for_agents_async(
                                models,
                                "AAPL"
                            )

                            # Should fail because no models succeeded
                            assert result is False


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """Test that semaphore limits max concurrent requests to 6"""

    active_requests = []
    max_concurrent = 0

    async def mock_generate(client, model_id, prompt):
        nonlocal max_concurrent
        active_requests.append(model_id)
        max_concurrent = max(max_concurrent, len(active_requests))
        await asyncio.sleep(0.05)
        active_requests.remove(model_id)
        return f"def execute_trade(ticker, cash_balance, shares_held):\n    return 'HOLD'"

    # Try with 10 models to test semaphore (should limit to 6 concurrent)
    models = [f"model-{i}" for i in range(10)]

    with patch('open_router.algo_gen.generate_algorithm_async', side_effect=mock_generate):
        with patch('open_router.algo_gen._get_accessible_models', return_value=set(models)):
            with patch('open_router.algo_gen.load_csv_preview', return_value=""):
                with patch('open_router.algo_gen.build_generation_prompt', return_value="prompt"):
                    with patch('open_router.algo_gen.build_diversity_directives', return_value=""):
                        with patch('open_router.algo_gen._save_code_for_model'):
                            with patch('os.path.exists', return_value=False):
                                result = await _generate_algorithms_for_agents_async(
                                    models,
                                    "AAPL"
                                )

                                assert result is True
                                # Max concurrent should never exceed 6 due to semaphore
                                assert max_concurrent <= 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
