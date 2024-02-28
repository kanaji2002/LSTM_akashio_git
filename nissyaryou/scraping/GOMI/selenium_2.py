from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from elements_manager import *

# 手動でChromeDriverのパスを指定
chrome_driver_path = 'C:/Users/Kanaji Rinntarou/chromedriver-win64/chromedriver.exe'
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


options = webdriver.ChromeOptions()
# options.add_argument('--headless')  #ヘッドレスモードを使用する場合は有効にします
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.implicitly_wait(3)
# to click on the element(ホーム >          各種データ...) found
# //*[@id="pr72"]

## ------------------------------ここから処理
driver.get('https://www.data.jma.go.jp/gmd/risk/obsdl/')
driver.implicitly_wait(3)

# e1 = driver.find_element(By.XPATH, get_xpath(driver, '/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[1]/table/tbody/tr[14]/td[7]/div')).click()

# ここでページが遷移するかもしれないので再度要素を検索
# e1 = driver.find_element(By.XPATH, get_xpath(driver, '/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[1]/table/tbody/tr[14]/td[7]/div/input')).click()
# e1 = driver.find_element_by_css_selector('input[type="hidden"][value="72"]')
# e1 =driver.find_element_by_xpath("//input[@value='72'] ")

# e1.click()

# e1 = driver.find_element(By.XPATH, "//input[@value='72']")
e1 = driver.find_element(By.XPATH, "//input[@name='prid' and @value='72']")
driver.implicitly_wait(3)


driver.execute_script("arguments[0].click();", e1)
e1.click()
