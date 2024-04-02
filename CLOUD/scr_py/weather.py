from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# weather/weather.py
def scr_weather():
    # Chromeのオプションを設定
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Chromeドライバーを初期化
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.set_page_load_timeout(10)  # 10秒

    try:
        driver.get('https://www.jma.go.jp/bosai/forecast/#area_type=class20s&area_code=3720100')
        print("The page was loaded in time！")
        
    except:
        print("time out!")

    wait = WebDriverWait(driver, 10)
    # XPathを使用して要素を見つける
    today = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[7]/div[1]/div/div/div[2]/table/tr[3]/td[1]/div")))
    print(today.text)

    e1 = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[7]/div[1]/div/div/div[2]/table/tr[3]/td[2]/div")))
    print(e1.text)

    e2 = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[7]/div[1]/div/div/div[2]/table/tr[3]/td[3]/div")))
    print(e2.text)

    e3 = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[7]/div[1]/div/div/div[2]/table/tr[3]/td[4]/div")))
    print(e3.text)

    e4 = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[7]/div[1]/div/div/div[2]/table/tr[3]/td[5]/div")))
    print(e4.text)

    e5 = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[7]/div[1]/div/div/div[2]/table/tr[3]/td[6]/div")))
    print(e5.text)

    e6 = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[7]/div[1]/div/div/div[2]/table/tr[3]/td[7]/div")))
    print(e6.text)


    data=[
        ["今日",today.text],
        ["1日後",e1.text],
        ["2日後",e2.text],
        ["3日後",e3.text],
        ["4日後",e4.text],
        ["5日後",e5.text],
        ["6日後",e6.text],
    ]

    df=pd.DataFrame(data,columns=["datetime","weather"])
    df['sun']=0
    df['rain']=0
    df['cloud']=0
    df['kousuiryou']=0
    df['nissyaryou']=0
    

    for index, row in df.iterrows():
        if "晴" in row['weather']:
            df.at[index, 'sun']=1
            df.at[index, 'kousuiryou']+=0
            df.at[index, 'nissyaryou']+=5

        if "雨" in row['weather']:
            df.at[index, 'rain']=1
            df.at[index, 'kousuiryou']+=15
            df.at[index, 'nissyaryou']-=10

        if "曇" in row['weather']:
            df.at[index, 'cloud']=1
            df.at[index, 'kousuiryou']+=0
            df.at[index, 'nissyaryou']-=5

        if "雨" in row['weather'] and "晴" in row['weather']:
            df.at[index, 'kousuiryou']=3
            df.at[index, 'nissyaryou']=-5
        if "雨" in row['weather'] and "曇" in row['weather']:
            df.at[index, 'kousuiryou']=5
            df.at[index, 'nissyaryou']=-8
    print(df)


    driver.close()
    return df

# scr_weather()


