import requests

url = 'http://localhost:8000/solve-captcha/'
with open('captchas/captcha_325.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
print(response.json())