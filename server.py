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
    last_real_time = items[-1]['time']

    predictions = []

    while len(predictions) < 120:
        last_real_time += 60 * 1000

        new_predict = {
            "time": last_real_time,
            "bikes": predictor.predict(items)
        }

        predictions.append(new_predict)

        items = items[1:] + [new_predict]

    return predictions

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9901)