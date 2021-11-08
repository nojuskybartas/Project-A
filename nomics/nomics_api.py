import urllib.request, json
KEY = '9c2429676003996506e3117d549d0fd3d1a052ac'


HISTORY_URL = f"https://api.nomics.com/v1/exchange-rates/history?key={KEY}&currency=BTC&start=2021-01-01T00%3A00%3A00Z&end=2021-11-08T00%3A00%3A00Z"
VOLUME_HISTORY_URL = f"https://api.nomics.com/v1/volume/history?key={KEY}&currency=BTC&start=2021-01-01T00%3A00%3A00Z&end=2021-11-08T00%3A00%3A00Z"
CURRENCY_URL = f"https://api.nomics.com/v1/currencies/ticker?key={KEY}&ids=BTC,ETH,XRP&interval=1h,30d&convert=EUR&per-page=100&page=1"

#VOLUME_HISTORY_URL = f"https://api.nomics.com/v1/volume/history?key={key}&start=2018-04-14T00%3A00%3A00Z&end=2018-05-14T00%3A00%3A00Z&convert=EUR"

with urllib.request.urlopen(HISTORY_URL) as url:
    data = json.loads(url.read().decode())
with open('data2.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
