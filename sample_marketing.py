# this is a sample API call for the website chat

import requests

url = 'http://127.0.0.1:7000/marketing'       # change the url to your link
data = {
    'format': 'Instagram',
    'topic': 'Football'
}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=data, headers=headers)

print(response.json())