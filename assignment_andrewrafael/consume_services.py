import requests

HOST = 'localhost'
PORT = 8009

# Request to check the status of the service
response = requests.get(url=f'http://{HOST}:{PORT}/status')
print(response.json())

# Request to score headlines
headlines = [
    "Running the ball. Catching the ball. Drew MacPherson does it all as Loyola wins state title. ‘He’s one of a kind.’",
    "The best Black Friday TV deals still available",
    "U.S. Faces Stiff Test Against Chinese Dominance in Africa. When President Biden visits Angola today, he will promote a rail project meant to show America’s commitment and to counter Chinese influence."
]

response = requests.post(url=f'http://{HOST}:{PORT}/score_headlines',
                         json={'headlines': headlines})

print(response.json())