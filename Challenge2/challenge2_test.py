import requests

res = requests.get("https://en.wikipedia.org/wiki/Microsoft")

print((res.text))
