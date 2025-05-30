from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from pathlib import Path
import time
import random

URL = "https://bsr.twse.com.tw/bshtm/bsMenu.aspx"
chrome_options = Options()
chrome_options.add_argument("--headless")


def find_next_img_number():
    output_dir = Path("./dataset/origin")
    txt_files = [f for f in output_dir.iterdir() if f.is_file()]
    return len(txt_files)


def collect_data():
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(URL)
    driver.implicitly_wait(5)
    img = driver.find_element(
        By.XPATH,
        '//*[@id="Panel_bshtm"]/table/tbody/tr/td/table/tbody/tr[1]/td/div/div[1]/img',
    )
    img = img.screenshot(f"dataset/origin/{find_next_img_number() + 1}.png")
    driver.close()


for i in range(500):
    collect_data()
    time.sleep(random.randint(3, 10))
