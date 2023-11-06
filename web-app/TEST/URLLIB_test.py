import urllib.request

url = 'https://qiita.com/hoto17296/items/8fcf55cc6cd823a18217'

req = urllib.request.Request(url)
with urllib.request.urlopen(req) as res:
    body = res.read()
print(body)