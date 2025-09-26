import os
import requests


def trigger(event, context):
    requests.post(os.environ.get('INFERENCE_HOST') + "/predict", data=event)