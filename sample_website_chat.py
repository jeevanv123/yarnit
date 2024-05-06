# this is a sample API call for the website chat

import requests

url = 'http://127.0.0.1:6000/website_chat'    # change the url to your link
data = {
    'url': 'https://en.wikipedia.org/wiki/Football',
    'question': 'How to play football?'
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=data, headers=headers)

print(response.json())