import urllib.request, json

KEY = '7344725fc3097b05921fea2d5a5218cb1090739397df0098b8e1aee025760a1c'
URL = f'https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=EUR&limit=2000&aggregate=1&e=Kraken&extraParams=your_app_name&api_key={KEY}'

with urllib.request.urlopen(URL) as url:
    data = json.loads(url.read().decode())
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)