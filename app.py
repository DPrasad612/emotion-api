from fastapi import FastAPI, UploadFile, File
from fer import FER
from PIL import Image
import numpy as np
import io

app = FastAPI()

detector = FER(mtcnn=False)

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/detect-emotion")
async def detect_emotion(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = detector.detect_emotions(np.array(image))

    if result:
        emotions = result[0]["emotions"]
        emotion = max(emotions, key=emotions.get)
    else:
        emotion = "neutral"

    return {"emotion": emotion}
