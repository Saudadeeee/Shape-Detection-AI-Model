import requests

url = 'http://localhost:5000/predict'
image_path = 'test.png'

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print(response.json())
