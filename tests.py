import unittest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestApp(unittest.TestCase):
    def test_sandbox_route(self):
        response = client.get("/sandbox")
        self.assertEqual(response.status_code, 200)
        self.assertIn("StoryVoyage Wizard Sandbox", response.text)

    def test_unauthorized_access(self):
        # Using a model-based endpoint to test auth
        response = client.post("/story/generate-arc", json={
            "story_elements": {"hero_name": "Leo", "setting": "Space", "goal": "Explore"},
            "student_state": {"current_estimated_level": "A1-Sub1"}
        })
        self.assertEqual(response.status_code, 401)

if __name__ == "__main__":
    unittest.main()
