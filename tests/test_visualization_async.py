import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd
import asyncio

# Make sure Python can find the src directory for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization import (
    create_histogram, _create_histogram_sync,
    create_scatter_plot, _create_scatter_plot_sync,
    create_box_plot, _create_box_plot_sync,
    create_correlation_heatmap, _create_correlation_heatmap_sync,
    create_time_series_plot, _create_time_series_plot_sync,
    create_interactive_plot, _create_interactive_plot_sync
)

class TestVisualizationAsync(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': ['x', 'y', 'x', 'y', 'x'],
            'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        })
        self.mock_success_return = {"type": "test_plot", "image": "base64_image_string"}
        self.mock_error_return = {"error": "Test error"}

    async def test_create_histogram_uses_to_thread(self):
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.visualization._create_histogram_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await create_histogram(self.sample_df, 'A', bins=10, title='Test Histo')

            mock_to_thread.assert_called_once()
            # Check that the first argument to to_thread is the sync function
            self.assertEqual(mock_to_thread.call_args[0][0], _create_histogram_sync)
            # Check remaining args passed to to_thread (which are then passed to the sync func)
            self.assertEqual(mock_to_thread.call_args[0][1], self.sample_df)
            self.assertEqual(mock_to_thread.call_args[0][2], 'A')
            self.assertEqual(mock_to_thread.call_args[0][3], 10)
            self.assertEqual(mock_to_thread.call_args[0][4], 'Test Histo')

            self.assertEqual(result, self.mock_success_return)

    async def test_create_histogram_handles_error(self):
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock, side_effect=Exception("Thread error")) as mock_to_thread:
            result = await create_histogram(self.sample_df, 'A')
            self.assertTrue("error" in result)
            self.assertEqual(result["error"], "Thread error")

    async def test_create_scatter_plot_uses_to_thread(self):
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.visualization._create_scatter_plot_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await create_scatter_plot(self.sample_df, 'A', 'B', color_column='C', title='Test Scatter')

            mock_to_thread.assert_called_once_with(
                _create_scatter_plot_sync, self.sample_df, 'A', 'B', 'C', 'Test Scatter'
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_create_box_plot_uses_to_thread(self):
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.visualization._create_box_plot_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await create_box_plot(self.sample_df, ['A', 'B'], title='Test BoxPlot')

            mock_to_thread.assert_called_once_with(
                _create_box_plot_sync, self.sample_df, ['A', 'B'], 'Test BoxPlot'
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_create_correlation_heatmap_uses_to_thread(self):
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.visualization._create_correlation_heatmap_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await create_correlation_heatmap(self.sample_df, columns=['A', 'B'], method='pearson')

            mock_to_thread.assert_called_once_with(
                _create_correlation_heatmap_sync, self.sample_df, ['A', 'B'], 'pearson'
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_create_time_series_plot_uses_to_thread(self):
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.visualization._create_time_series_plot_sync', return_value=self.mock_success_return) as mock_sync_func:

            result = await create_time_series_plot(self.sample_df, date_column='Date', value_columns=['A', 'B'], title='Test TimeSeries')

            mock_to_thread.assert_called_once_with(
                _create_time_series_plot_sync, self.sample_df, 'Date', ['A', 'B'], 'Test TimeSeries'
            )
            self.assertEqual(result, self.mock_success_return)

    async def test_create_interactive_plot_uses_to_thread(self):
        mock_interactive_success = {"type": "interactive_scatter", "plot_json": "{}"}
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
             patch('src.visualization._create_interactive_plot_sync', return_value=mock_interactive_success) as mock_sync_func:

            kwargs = {'x': 'A', 'y': 'B', 'title': 'Test Interactive'}
            result = await create_interactive_plot(self.sample_df, plot_type='scatter', **kwargs)

            mock_to_thread.assert_called_once_with(
                _create_interactive_plot_sync, self.sample_df, 'scatter', **kwargs
            )
            self.assertEqual(result, mock_interactive_success)

    async def test_interactive_plot_handles_error_from_sync(self):
        # Test error propagation when the sync function itself returns an error dict
        with patch('src.visualization.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # Make to_thread return the result of the (mocked) sync function directly
            async def side_effect_for_to_thread(sync_func, *args, **kwargs):
                # Simulate sync_func returning an error
                if sync_func == _create_interactive_plot_sync:
                    return self.mock_error_return
                return self.mock_success_return # Default for other sync funcs if any

            mock_to_thread.side_effect = side_effect_for_to_thread

            kwargs = {'x': 'A', 'y': 'B', 'title': 'Test Interactive Error'}
            result = await create_interactive_plot(self.sample_df, plot_type='scatter', **kwargs)

            self.assertTrue("error" in result)
            self.assertEqual(result["error"], "Test error")

if __name__ == '__main__':
    unittest.main()
