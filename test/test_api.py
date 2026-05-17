from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")

    assert response.status_code == 200


def test_predict_text():
    payload = {
        "text": "Experienced data scientist with Python and machine learning skills"
    }

    response = client.post(
        "/predict_text",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data