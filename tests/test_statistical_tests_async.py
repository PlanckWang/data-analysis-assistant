import unittest
from unittest.mock import patch, AsyncMock
import pandas as pd
import asyncio
import numpy as np # For sample data

# Make sure Python can find the src directory for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.statistical_tests import (
    normality_test, _normality_test_sync,
    t_test, _t_test_sync,
    anova_test, _anova_test_sync,
    chi_square_test, _chi_square_test_sync,
    correlation_test, _correlation_test_sync
)

class TestStatisticalTestsAsync(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.sample_df = pd.DataFrame({
            'col1': np.random.normal(0, 1, 100),
            'col2': np.random.normal(5, 1, 100),
            'group': np.random.choice(['A', 'B', 'C'], 100),
            'category1': np.random.choice(['X', 'Y'], 100),
            'category2': np.random.choice(['M', 'N'], 100)
        })
        self.mock_success_return = {"test_passed": True, "p_value": 0.5}
        self.mock_error_return = {"error": "Test error from sync function"}

    async def test_normality_test_uses_to_thread(self):
        with patch('src.statistical_tests.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.statistical_tests._normality_test_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await normality_test(self.sample_df, columns=['col1'], alpha=0.05)

            mock_to_thread.assert_called_once_with(
                _normality_test_sync, self.sample_df, ['col1'], 0.05
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_normality_test_handles_thread_error(self):
        with patch('src.statistical_tests.asyncio.to_thread', new_callable=AsyncMock, side_effect=Exception("Threading issue")) as mock_to_thread:
            result = await normality_test(self.sample_df, columns=['col1'])
            self.assertTrue("error" in result)
            self.assertEqual(result["error"], "Threading issue")

    async def test_normality_test_handles_sync_error_via_to_thread_return(self):
        # This tests if the sync function returns an error dict and to_thread passes it through
        async def to_thread_passthrough(func, *args, **kwargs):
            return func(*args, **kwargs) # Directly call the (mocked) sync function

        with patch('src.statistical_tests.asyncio.to_thread', new_callable=AsyncMock, side_effect=to_thread_passthrough) as mock_to_thread, \
             patch('src.statistical_tests._normality_test_sync', return_value=self.mock_error_return) as mock_sync_func:

            result = await normality_test(self.sample_df, columns=['col1'])

            mock_to_thread.assert_called_once()
            mock_sync_func.assert_called_once()
            self.assertEqual(result, self.mock_error_return)


    async def test_t_test_uses_to_thread(self):
        with patch('src.statistical_tests.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.statistical_tests._t_test_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await t_test(self.sample_df, column1='col1', column2='col2', paired=False)

            mock_to_thread.assert_called_once_with(
                _t_test_sync, self.sample_df, 'col1', 'col2', False, 'two-sided' # default alternative
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_anova_test_uses_to_thread(self):
        with patch('src.statistical_tests.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.statistical_tests._anova_test_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await anova_test(self.sample_df, dependent_var='col1', independent_var='group')

            mock_to_thread.assert_called_once_with(
                _anova_test_sync, self.sample_df, 'col1', 'group'
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_chi_square_test_uses_to_thread(self):
        with patch('src.statistical_tests.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.statistical_tests._chi_square_test_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await chi_square_test(self.sample_df, column1='category1', column2='category2')

            mock_to_thread.assert_called_once_with(
                _chi_square_test_sync, self.sample_df, 'category1', 'category2'
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_correlation_test_uses_to_thread(self):
        with patch('src.statistical_tests.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.statistical_tests._correlation_test_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await correlation_test(self.sample_df, column1='col1', column2='col2', method='pearson')

            mock_to_thread.assert_called_once_with(
                _correlation_test_sync, self.sample_df, 'col1', 'col2', 'pearson'
            )
            self.assertEqual(result, self.mock_success_return)

if __name__ == '__main__':
    unittest.main()
