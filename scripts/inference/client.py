# Run: `python scripts/inference/client.py` to test server running locally
import requests

URL = "https://helios-74045595887.us-central1.run.app"

event = {
    "bucket": "ai2-ivan-helios-input-data",
    "name": "Togo_v20250925_ts1000/2019-03_2020-03/tile1.tif"
}
response = requests.post(f"{URL}/predict", json=event)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
