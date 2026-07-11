import unittest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestApp(unittest.TestCase):
    def test_sandbox_route(self):
        response = client.get("/sandbox")
        self.assertEqual(response.status_code, 200)
        self.assertIn("StoryVoyage DAG Sandbox", response.text)

    def test_unauthorized_access(self):
        response = client.post("/adventure/setup", json={})
        self.assertEqual(response.status_code, 401)

if __name__ == "__main__":
    unittest.main()
