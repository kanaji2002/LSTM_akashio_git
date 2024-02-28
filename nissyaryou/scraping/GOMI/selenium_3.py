from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
# nissyaryou_sc.py


# Chromeのオプションを設定
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Chromeドライバーを初期化
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
# driver.implicitly_wait(3)

# ターゲットのウェブページにアクセス
driver.get('https://www.data.jma.go.jp/gmd/risk/obsdl/')
driver.implicitly_wait(2)

# # 要素がクリック可能になるまで待機するWebDriverWaitを作成
# wait = WebDriverWait(driver, 5)

# XPathを使用して要素を見つける
e1 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[1]/table/tbody/tr[14]/td[7]/div")
# 要素をクリック
e1.click()


# 高松を選択
e2 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[1]/div[2]/div[3]/div")
# 要素をクリック
e2.click()

# 項目を選ぶ
e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[1]/img[2]")
# 要素をクリック
e3.click()

# 日照/日射
e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div[3]/div[1]/ul/li[3]/a")
# 要素をクリック
e3.click()

# 日合計全天日射量を選択
e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div[3]/div[1]/div[4]/table/tbody/tr[1]/td[2]")
# 要素をクリック
e3.click()

# 期間を選ぶ
e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[1]/img[3]")
# 要素をクリック
e3.click()

# 最近一か月
e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[3]/div/div/div[1]/div[2]/div[1]/input[2]")
# 要素をクリック
e3.click()


# 表示
e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[2]/div/div/div[1]/div[1]/span/img")
# 要素をクリック
e3.click()



# #要素をxpatで絞る
# element = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[1]/div[2]/div[2]/div[2]/div[5]/div[3]/div')
# #スクロールをして一番下する
# driver.execute_script("return arguments[0].scrollIntoView(true);", element)
# time.sleep(5)
#textを取得する
text = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[1]/div[2]/div[2]/div[2]/div[5]/div[3]/div/div[32]/div/span').text
print(text)






# ここで次のページに移動した後の処理を追加できます
time.sleep(10)
