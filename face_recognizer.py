import json
import requests
import io


# My super secret Azure Key
KEY = "d09620f2c7624e7b9b88e4cca11694e0"
# My  Azure cognitive services endpoint
ENDPOINT = "https://west-eu.cognitiveservices.azure.com"
face_api_url = ENDPOINT + '/face/v1.0/detect'

headers = {'Ocp-Apim-Subscription-Key': KEY,
           'Content-Type':'application/octet-stream'}
params = {
	'detectionModel': 'detection_01',
    'returnFaceAttributes': 'emotion'
}

def get_emotion_from_json(data):
    emotion = ('None', 0)
    for key, val in data['faceAttributes']['emotion'].items():
        if val > emotion[1]:
            emotion = (key,val)
    emotion = (emotion[0], str(emotion[1] * 100))
    return emotion

def get_emotion(imagePath):
    img = open(imagePath,mode="rb")  
    res = requests.post(face_api_url, params=params,
                         headers=headers, data=img)
    return get_emotion_from_json(res.json()[0])