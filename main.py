from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("Mental_Health_Model.pkl")


class PatientData(BaseModel):
    Study_Hours: float
    Age: int
    Avg_Daily_Usage_Hours: float
    Daily_Unlocks: int
    Physical_Activity_Hours: float
    Sleep_Hours_Per_Night: float
    Stress_Level: str
    Gender: str
    Academic_Level: str
    Most_Used_Platform: str
    Purpose_Of_Use: str
    Grouped_country: str


@app.get("/")
def home():
    return {"message": "Mental Health Model API is running"}


@app.post("/predict")
def predict(data: PatientData):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)

    return {
        "predicted_mental_health_score": float(prediction[0])
    }