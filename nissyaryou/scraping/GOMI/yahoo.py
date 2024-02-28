from urllib import response
import requests
from bs4 import BeautifulSoup
# urlのサイトの情報をもらってくる
# urlのサイトの情報をもらってくる
# urlのサイトの情報をもらってくる
url = 'https://www.yahoo.co.jp/'
html = requests.get(url)

# BeautifulSoupで使えるように変換
soup = BeautifulSoup(html.content, 'html.parser')

# h1タグを全て取り出す
h1 = soup.find_all('h1')

# # 最初の1つを表示
# print(h1[0].text)
# print('\n')

# # 全部表示
# for element in h1:
#     print(element.text)
