from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from fastapi.encoders import jsonable_encoder

from train import BikePredictor

app = FastAPI()


@app.get("/")
async def root():
    return {"status": True}


class BikeStation(BaseModel):
    location: dict
    count: int


class Item(BaseModel):
    time: float
    bikes: List[BikeStation]


predictor = BikePredictor()
predictor.load_model('bike_model_1min_v1.pth')


@app.post('/predict')
def predict(items: List[Item]):
    items = jsonable_encoder(items)
    return predictor.predict(items)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9901)