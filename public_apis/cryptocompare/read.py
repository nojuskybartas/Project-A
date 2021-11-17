import json
import datetime

dates = []
with open('data.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
    for reading in (d['Data']['Data']):
        dates.append(datetime.datetime.utcfromtimestamp(reading['time']))

print(dates[0], dates[-1])
