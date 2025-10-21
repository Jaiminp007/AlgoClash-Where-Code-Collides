"""
Unit tests for algorithm preview/management API endpoints.
Tests the /api/algos blueprint for security, functionality, and error handling.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))

from app import app
from api.algos import is_safe_filename, safe_path, get_model_name_from_filename


class TestAlgosAPI(unittest.TestCase):
    """Test suite for algorithm preview API"""

    def setUp(self):
        """Set up test client and temporary directory"""
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Patch OUTPUT_DIR to use test directory
        import api.algos
        self.original_output_dir = api.algos.OUTPUT_DIR
        api.algos.OUTPUT_DIR = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        # Restore original OUTPUT_DIR
        import api.algos
        api.algos.OUTPUT_DIR = self.original_output_dir

        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_test_file(self, filename, content="test content"):
        """Helper to create a test algorithm file"""
        filepath = Path(self.test_dir) / filename
        with open(filepath, 'w') as f:
            f.write(content)
        return filepath

    # Security tests
    def test_is_safe_filename_valid(self):
        """Test valid filenames pass security check"""
        self.assertTrue(is_safe_filename('generated_algo_gpt-4.py'))
        self.assertTrue(is_safe_filename('generated_algo_claude-3.py'))
        self.assertTrue(is_safe_filename('generated_algo_model_123.py'))

    def test_is_safe_filename_invalid(self):
        """Test invalid filenames fail security check"""
        # Directory traversal attempts
        self.assertFalse(is_safe_filename('../generated_algo_test.py'))
        self.assertFalse(is_safe_filename('../../etc/passwd'))
        self.assertFalse(is_safe_filename('generated_algo/../test.py'))

        # Wrong pattern
        self.assertFalse(is_safe_filename('malicious.py'))
        self.assertFalse(is_safe_filename('algo_generated_test.py'))
        self.assertFalse(is_safe_filename('generated_algo_test.txt'))

        # Path separators
        self.assertFalse(is_safe_filename('test/generated_algo_test.py'))
        self.assertFalse(is_safe_filename('test\\generated_algo_test.py'))

    def test_safe_path_valid(self):
        """Test safe_path returns valid path for legitimate files"""
        is_valid, path = safe_path('generated_algo_test.py')
        self.assertTrue(is_valid)
        self.assertIsNotNone(path)

    def test_safe_path_invalid(self):
        """Test safe_path rejects malicious paths"""
        is_valid, path = safe_path('../../../etc/passwd')
        self.assertFalse(is_valid)
        self.assertIsNone(path)

    def test_get_model_name_from_filename(self):
        """Test model name extraction from filename"""
        self.assertEqual(
            get_model_name_from_filename('generated_algo_gpt-4.py'),
            'gpt-4'
        )
        self.assertEqual(
            get_model_name_from_filename('generated_algo_claude-3-opus.py'),
            'claude-3-opus'
        )

    # API endpoint tests
    def test_list_algorithms_empty(self):
        """Test listing algorithms when directory is empty"""
        response = self.client.get('/api/algos')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 0)

    def test_list_algorithms_with_files(self):
        """Test listing algorithms returns file metadata"""
        # Create test files
        self.create_test_file('generated_algo_model1.py', 'content1')
        self.create_test_file('generated_algo_model2.py', 'content2')

        response = self.client.get('/api/algos')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(len(data), 2)
        self.assertIn('filename', data[0])
        self.assertIn('modelName', data[0])
        self.assertIn('sizeBytes', data[0])
        self.assertIn('createdAt', data[0])
        self.assertIn('modifiedAt', data[0])

    def test_get_algorithm_success(self):
        """Test retrieving algorithm content"""
        content = "def execute_trade():\n    return 'BUY'"
        self.create_test_file('generated_algo_test.py', content)

        response = self.client.get('/api/algos/generated_algo_test.py')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), content)
        self.assertIn('text/plain', response.content_type)

    def test_get_algorithm_not_found(self):
        """Test 404 when algorithm doesn't exist"""
        response = self.client.get('/api/algos/generated_algo_nonexistent.py')
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertIn('error', data)

    def test_get_algorithm_invalid_filename(self):
        """Test 400 for invalid filename"""
        response = self.client.get('/api/algos/../../../etc/passwd')
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_delete_algorithm_success(self):
        """Test deleting a single algorithm"""
        self.create_test_file('generated_algo_test.py', 'test')

        response = self.client.delete('/api/algos/generated_algo_test.py')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get('deleted'))

        # Verify file is deleted
        filepath = Path(self.test_dir) / 'generated_algo_test.py'
        self.assertFalse(filepath.exists())

    def test_delete_algorithm_not_found(self):
        """Test 404 when trying to delete non-existent file"""
        response = self.client.delete('/api/algos/generated_algo_nonexistent.py')
        self.assertEqual(response.status_code, 404)

    def test_delete_all_algorithms(self):
        """Test bulk delete of all algorithms"""
        # Create multiple test files
        self.create_test_file('generated_algo_model1.py', 'test1')
        self.create_test_file('generated_algo_model2.py', 'test2')
        self.create_test_file('generated_algo_model3.py', 'test3')

        response = self.client.delete('/api/algos')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('deleted'), 3)

        # Verify all files are deleted
        remaining = list(Path(self.test_dir).glob('generated_algo_*.py'))
        self.assertEqual(len(remaining), 0)

    def test_download_algorithm(self):
        """Test downloading algorithm as attachment"""
        content = "def execute_trade():\n    return 'BUY'"
        self.create_test_file('generated_algo_test.py', content)

        response = self.client.get('/api/algos/generated_algo_test.py/download')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), content)
        self.assertIn('attachment', response.headers.get('Content-Disposition', ''))


class TestSecurityEdgeCases(unittest.TestCase):
    """Test security edge cases and attack vectors"""

    def test_no_directory_traversal(self):
        """Test various directory traversal attempts are blocked"""
        attacks = [
            '../generate_algo/test.py',
            '../../etc/passwd',
            'test/../../../etc/passwd',
            '..\\..\\windows\\system32',
            'generated_algo_test.py/../../../etc/passwd',
        ]
        for attack in attacks:
            self.assertFalse(is_safe_filename(attack), f"Failed to block: {attack}")

    def test_null_byte_injection(self):
        """Test null byte injection attempts are blocked"""
        self.assertFalse(is_safe_filename('generated_algo_test.py\x00.txt'))

    def test_empty_and_none_filenames(self):
        """Test empty and None filenames are rejected"""
        self.assertFalse(is_safe_filename(''))
        self.assertFalse(is_safe_filename(None))


if __name__ == '__main__':
    unittest.main()
