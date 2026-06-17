import unittest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestApp(unittest.TestCase):
    def test_sandbox_route(self):
        response = client.get("/sandbox")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ESL Adaptive Learning Sandbox", response.text)

    def test_unauthorized_access(self):
        response = client.post("/generate-scene", json={})
        self.assertEqual(response.status_code, 401)

if __name__ == "__main__":
    unittest.main()
