import unittest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestApp(unittest.TestCase):
    def test_sandbox_route(self):
        response = client.get("/sandbox")
        self.assertEqual(response.status_code, 200)
        self.assertIn("StoryVoyage Advanced Sandbox", response.text)

    def test_unauthorized_access(self):
        # Using the new setup endpoint to test auth
        response = client.post("/adventure/setup", json={
            "rank_index": 100,
            "genre": "Space"
        })
        self.assertEqual(response.status_code, 401)

if __name__ == "__main__":
    unittest.main()
