from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ChromeDriverのパスを指定
driver_path = 'C:/Users/Kanaji Rinntarou/chromedriver-win64/chromedriver.exe'

# ChromeOptionsを作成し、実行可能なパスを設定
chrome_options = webdriver.ChromeOptions()
chrome_options.binary_location = driver_path

# WebDriverの初期化
driver = webdriver.Chrome(options=chrome_options)

# ウェブサイトにアクセス

driver.get('https://www.data.jma.go.jp/gmd/risk/obsdl/')



# ページが読み込まれるまで待機する（必要に応じて）
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="result"]')))

# ウェブページのHTMLを取得（確認用）
print(driver.page_source)

# ブラウザを閉じる
driver.quit()