import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Set
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from util import convert_interval


@dataclass
class Location:
    lng: float
    lat: float

    def __hash__(self):
        return hash((self.lng, self.lat))

    def __eq__(self, other):
        return self.lng == other.lng and self.lat == other.lat

class BikeDataPreprocessor:
    def __init__(self, max_gap_minutes: int = 5):
        self.max_gap_minutes = max_gap_minutes
        self.location_stats = {}
        self.locations: List[Location] = []

    def _extract_locations(self, data: List[Dict]) -> List[Location]:
        """Extract unique locations from the dataset"""
        unique_locations: Set[Location] = set()

        for item in data:
            for bike in item['bikes']:
                loc = bike['location']
                unique_locations.add(Location(lng=loc['lng'], lat=loc['lat']))

        # Sort locations to maintain consistent ordering
        self.locations = sorted(list(unique_locations), key=lambda x: (x.lng, x.lat))
        return self.locations

    def _create_location_index(self) -> Dict[Tuple[float, float], int]:
        """Create mapping from location coordinates to index"""
        return {(loc.lng, loc.lat): i for i, loc in enumerate(self.locations)}

    def preprocess_data(self, data: List[Dict]) -> np.ndarray:
        # Extract unique locations if not already done
        if not self.locations:
            self._extract_locations(data)

        location_index = self._create_location_index()
        num_locations = len(self.locations)

        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x['time'])

        # Create time series matrix
        timestamps = []
        values = []

        for item in sorted_data:
            timestamps.append(item['time'] / 1000)  # Convert milliseconds to seconds

            # Initialize counts for this timestamp
            counts = np.zeros(num_locations)

            # Fill in available counts
            for bike in item['bikes']:
                loc = bike['location']
                idx = location_index[(loc['lng'], loc['lat'])]
                counts[idx] = bike['count']

            values.append(counts)

        # Convert to numpy array
        X = np.array(values)

        # Handle gaps and normalize
        X = self._handle_gaps(X, timestamps)
        X = self._normalize_data(X)

        return X

    def _handle_gaps(self, X: np.ndarray, timestamps: List[float]) -> np.ndarray:
        gaps = []
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]) / 60  # Convert to minutes
            if gap > self.max_gap_minutes:
                gaps.append(i)

        # Split data at gaps and interpolate within valid segments
        valid_segments = []
        start = 0
        for gap in gaps:
            if gap - start >= 15:  # Minimum sequence length
                segment = X[start:gap]
                valid_segments.append(self._interpolate_segment(segment))
            start = gap

        if len(X) - start >= 15:
            segment = X[start:]
            valid_segments.append(self._interpolate_segment(segment))

        return np.concatenate(valid_segments, axis=0)

    def _interpolate_segment(self, segment: np.ndarray) -> np.ndarray:
        """Linear interpolation for missing values within a segment"""
        # Implement more sophisticated interpolation if needed
        return segment

    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize data for each location"""
        for i, loc in enumerate(self.locations):
            mean = X[:, i].mean()
            std = X[:, i].std()
            X[:, i] = (X[:, i] - mean) / (std + 1e-8)
            self.location_stats[i] = {'mean': mean, 'std': std}
        return X

class BikePredictionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()

        # Spatial embedding for locations
        self.spatial_embedding = nn.Linear(2, 16)  # Convert lng/lat to embedding

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, spatial_features=None):
        # x shape: (batch_size, sequence_length, input_size)

        # Process spatial features if provided
        if spatial_features is not None:
            spatial_embed = self.spatial_embedding(spatial_features)
            # Add spatial information to input (implement as needed)

        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class BikeDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int = 15):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.FloatTensor(X), torch.FloatTensor(y)

class BikePredictor:
    def __init__(self, model_path: str = None):
        self.model = None
        self.preprocessor = BikeDataPreprocessor()
        if model_path:
            self.load_model(model_path)

    def train(self, data: List[Dict], epochs: int = 100, batch_size: int = 32):
        # Preprocess data
        X = self.preprocessor.preprocess_data(data)

        # Create dataset and dataloader
        dataset = BikeDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Prepare spatial features
        spatial_features = torch.tensor([
            [loc.lng, loc.lat] for loc in self.preprocessor.locations
        ], dtype=torch.float32)

        # Initialize model
        self.model = BikePredictionModel(input_size=len(self.preprocessor.locations))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch, spatial_features)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')

    def predict(self, recent_data: List[Dict]) -> List[Dict]:
        """Predict bike counts for the next 15 minutes"""
        # Preprocess recent data
        X = self.preprocessor.preprocess_data(recent_data)

        # Prepare input sequence
        X = torch.FloatTensor(X[-15:]).unsqueeze(0)  # Add batch dimension

        # Prepare spatial features
        spatial_features = torch.tensor([
            [loc.lng, loc.lat] for loc in self.preprocessor.locations
        ], dtype=torch.float32)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X, spatial_features)

        # Denormalize predictions
        prediction = prediction.numpy()[0]
        result = []

        for i, loc in enumerate(self.preprocessor.locations):
            stats = self.preprocessor.location_stats[i]
            count = prediction[i] * stats['std'] + stats['mean']
            result.append({
                'location': {'lng': loc.lng, 'lat': loc.lat},
                'count': max(0, int(round(count)))
            })

        return result

    def save_model(self, path: str):
        """Save model and preprocessing parameters"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'location_stats': self.preprocessor.location_stats,
            'locations': [(loc.lng, loc.lat) for loc in self.preprocessor.locations]
        }
        torch.save(model_state, path)

    def load_model(self, path: str):
        """Load model and preprocessing parameters"""
        model_state = torch.load(path)
        self.preprocessor.locations = [
            Location(lng=lng, lat=lat)
            for lng, lat in model_state['locations']
        ]
        self.preprocessor.location_stats = model_state['location_stats']
        self.model = BikePredictionModel(input_size=len(self.preprocessor.locations))
        self.model.load_state_dict(model_state['model_state_dict'])
        self.model.eval()

if __name__ == '__main__':
    # Load your data
    import os
    data_entry =[]
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
                "count": x['availableBikes']
            } for x in entry]
        })

        print(f"Load {len(data_entry)}/{len(data_path)}")

    data_entry = convert_interval(data_entry)

    train_data_entry = data_entry[:-6]

    predictor = BikePredictor()
    predictor.load_model('bike_model_1min_v1.pth')
    # predictor.save_model('bike_model_1min_v1.pth')

    # Make predictions
    recent_data = data_entry[-16:-1]  # Get last 5 entries

    prediction = predictor.predict(recent_data)
    validation_point = [
        {'lat': 42.407381, 'lng': -71.055356},
        {'lat': 42.363564, 'lng': -71.100441},
        {'lat': 42.338431, 'lng': -71.08169},
        {'lat': 42.357119, 'lng': -71.186045}
    ]
    for v in validation_point:
        print('Predict -----')
        print([x for x in prediction if x['location']['lat'] == v['lat'] and x['location']['lng'] == v['lng']][0]['count'])


        print('\n')

        print('Real -----')

        print([x for x in data_entry[-1]['bikes'] if x['location']['lat'] == v['lat'] and x['location']['lng'] == v['lng']][0]['count'])

        print('\n\n')