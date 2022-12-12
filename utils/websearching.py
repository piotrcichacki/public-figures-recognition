import time
import io
import PIL
import numpy as np
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.webdriver import FirefoxOptions

class ErrorDuringDownloadingImage(Exception):
    def __init__(self, message=None):
        super().__init__(message)


class WebDriver:
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        service = Service(executable_path=self.file_path)
        opts = FirefoxOptions()
        opts.add_argument("--headless")
        self.web_driver = webdriver.Firefox(service=service, options=opts)
        self.web_driver.maximize_window()
        self.web_driver.get("https://www.google.com/")
        WebDriverWait(self.web_driver, 10).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="L2AGLb"]'))).click()
        return self.web_driver

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Web driver error: {exc_val}")
        self.web_driver.close()


def search_in_google(driver, search_query, delay=2.0):
    search_query = search_query.replace(" ", "+")
    url = f"https://www.google.com/search?tbm=isch&q={search_query}"
    driver.get(url)
    time.sleep(delay)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


def switch_to_google_graphics(driver, delay=2.0):
    graphics_button = driver.find_element(by=By.LINK_TEXT, value="Images")
    url = graphics_button.get_attribute("href")
    driver.get(url)
    time.sleep(delay)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


def search_images(driver, max_index=100, delay=1.0):
    thumbnails = driver.find_elements(by=By.CLASS_NAME, value="Q4LuWd")
    current_index = 0
    while current_index < max_index:
        try:
            thumbnails[current_index].click()
        except:
            return
        time.sleep(delay)

        elements = driver.find_elements(by=By.CLASS_NAME, value="n3VNCb")
        for element in elements:
            if element.get_attribute("src") and "http" in element.get_attribute("src"):
                image_url = element.get_attribute("src")
                yield image_url
        current_index += 1


def download_image(image_url):
    print(f"Downloading image: {image_url}")
    try:
        img_content = requests.get(image_url).content
    except requests.exceptions.InvalidSchema as err:
        raise ErrorDuringDownloadingImage(str(err))

    img_file = io.BytesIO(img_content)
    try:
        img = np.array(Image.open(img_file))
    except PIL.UnidentifiedImageError as err:
        raise ErrorDuringDownloadingImage(str(err))

    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ErrorDuringDownloadingImage(message="Incorrect image data format")


def save_image(image, save_path):
    img = Image.fromarray(image)
    with open(save_path, mode="w") as img_file:
        img.save(img_file, "JPEG")

