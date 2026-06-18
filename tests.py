import unittest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestApp(unittest.TestCase):
    def test_sandbox_route(self):
        response = client.get("/sandbox")
        self.assertEqual(response.status_code, 200)
        self.assertIn("StoryVoyage Adaptive Learning Sandbox", response.text)

    def test_unauthorized_access(self):
        # Using a model-based endpoint to test auth
        response = client.post("/generate-scene", json={
            "category": "test",
            "target_words": ["word1", "word2", "word3", "word4", "word5", "word6"],
            "student_state": {
                "current_estimated_level": "A1-Sub1"
            }
        })
        self.assertEqual(response.status_code, 401)

if __name__ == "__main__":
    unittest.main()
