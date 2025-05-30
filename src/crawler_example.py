import os
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from pathlib import Path
import clean_data
import cv2
import torch
from PIL import Image
from torchvision import transforms
from train_model import CaptchaCRNN, idx_to_char
import time
from selenium.common.exceptions import NoSuchElementException

URL = "https://bsr.twse.com.tw/bshtm/bsMenu.aspx"
chrome_options = Options()
# chrome_options.add_argument("--headless")
preferences = {
    "download.default_directory": os.getcwd(),  # pass the variable
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True,
}
chrome_options.add_experimental_option("prefs", preferences)
model_path = "captcha_crnn_best_model.pth"
model = CaptchaCRNN()
model.load_state_dict(torch.load(model_path, map_location="cpu"))


def find_next_img_number():
    output_dir = Path("./dataset/origin")
    txt_files = [f for f in output_dir.iterdir() if f.is_file()]
    return len(txt_files)


def scarpying(stock_list: list[str]):
    driver = webdriver.Chrome(options=chrome_options)
    counter = 0
    while counter < len(stock_list):
        driver.get(URL)
        driver.implicitly_wait(5)
        img = driver.find_element(
            By.XPATH,
            '//*[@id="Panel_bshtm"]/table/tbody/tr/td/table/tbody/tr[1]/td/div/div[1]/img',
        )
        img_bytes = img.screenshot_as_png
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_mat = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cleaned_img = clean_data.clean_image(img_mat)
        cleaned_img = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cleaned_img).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = output.max(2)
            predicted_chars = [idx_to_char[idx.item()] for idx in predicted.squeeze(0)]
            predicted_string = "".join(predicted_chars)
        captcha_input = driver.find_element(
            By.XPATH,
            '//*[@id="Panel_bshtm"]/table/tbody/tr/td/table/tbody/tr[1]/td/div/div[2]/input',
        )
        captcha_input.clear()
        captcha_input.send_keys(predicted_string)
        print(predicted_string)
        time.sleep(2)
        stock_symbol = driver.find_element(By.XPATH, '//*[@id="TextBox_Stkno"]')
        stock_symbol.clear()
        stock_symbol.send_keys(stock_list[counter])
        time.sleep(2)
        driver.find_element(By.XPATH, '//*[@id="btnOK"]').click()
        try:
            stock_data = driver.find_element(
                By.XPATH, '//*[@id="HyperLink_DownloadCSV"]'
            )
            stock_data.click()
            counter += 1
        except NoSuchElementException:
            continue


scarpying(["0050", "0051"])
