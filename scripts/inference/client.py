# Run: `python scripts/inference/client.py` to test server running locally
import numpy as np
import requests

URL = "https://helios-74045595887.us-central1.run.app"

S2_bands = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
START_MONTH = 3
NUM_TIMESTEPS = 12
months = np.fmod(np.arange(START_MONTH - 1, START_MONTH - 1 + NUM_TIMESTEPS), 12)
timestamp_mock_data = np.stack(
    [np.ones_like(months), months, np.ones_like(months)], axis=-1
).tolist()
s2_mock_data = np.zeros([1, 1, NUM_TIMESTEPS, len(S2_bands)]).tolist()
mock_json = {"sentinel2_l2a": [s2_mock_data], "timestamps": [timestamp_mock_data]}

response = requests.post(f"{URL}/predict", json=mock_json)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
