import os
import json

from train import BikePredictor
from util import convert_interval

data_entry = []
data_path = ['data/' + x for x in os.listdir('data')]

for path in data_path:
    with open(path, 'r') as f:
        entry = json.load(f)

    data_entry.append({
        "time": float(path.split('.')[0].split('/')[1]),
        "bikes": [{
            "location": {
                "lat": x['location']['lat'],
                "lng": x['location']['lng'],
            },
            "count": x['availableBikes'],
        } for x in entry]
    })

    print(f"Load {len(data_entry)}/{len(data_path)}")

data_entry = convert_interval(data_entry)[-16:-1]

import requests
response = requests.post('http://127.0.0.1:9901/predict', json=data_entry)
print(response.json())