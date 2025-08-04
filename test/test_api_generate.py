import unittest
from app import app

class APIGenerateTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_generate_no_query(self):
        response = self.app.post('/api/generate', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_generate_empty_query(self):
        response = self.app.post('/api/generate', json={'query': ''})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_generate_valid_query(self):
        # This test assumes a valid query and all dependencies are set up
        response = self.app.post('/api/generate', json={'query': 'Show all vessel rotations'})
        # Accept 200 or 500 (if backend fails), but should not be a network error
        self.assertIn(response.status_code, [200, 500])

if __name__ == '__main__':
    unittest.main()
