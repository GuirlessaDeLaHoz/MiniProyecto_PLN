from fastapi import FastAPI

from app.schemas import TextRequest
from app.predictor import predict_text

app = FastAPI()

@app.get("/")
def home():

    return {
        "message": "NLP Resume Classification API"
    }

@app.post("/predict_text")
def predict(request: TextRequest):

    prediction = predict_text(
        request.text
    )

    return {
        "prediction": prediction
    }