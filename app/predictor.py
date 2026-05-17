import time
import torch
import pickle

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

MODEL_PATH = "Models/distilbert_result/model"
TOKENIZER_PATH = "Models/distilbert_result/tokenizer"

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

tokenizer = DistilBertTokenizerFast.from_pretrained(
    TOKENIZER_PATH
)

with open(
    "Models/distilbert_result/label_encoder.pkl",
    "rb"
) as f:

    label_encoder = pickle.load(f)

model.eval()

def predict_text(text):

    start_time = time.time()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = torch.max(
        probabilities
    ).item()

    category = label_encoder.inverse_transform(
        [prediction]
    )[0]

    inference_time = round(
        time.time() - start_time,
        4
    )

    return {
        "prediction": category,
        "confidence": round(confidence, 4),
        "inference_time_seconds": inference_time
    }