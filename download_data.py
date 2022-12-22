import os

import yaml
import numpy as np
from utils.detection import detect_faces_from_image, crop_image
from utils.websearching import WebDriver, search_in_google, search_images, download_image, \
    ErrorDuringDownloadingImage, save_image
import func_timeout

def load_catalog(catalog_path):
    with open(catalog_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)
            return {"footballers": [], "keywords": []}


def create_faces(image_url, person_folder_path):
    image = download_image(image_url)
    detected_faces = detect_faces_from_image(image)
    for face_box in detected_faces:
        face_crop = crop_image(image, face_box)
        save_path = os.path.join(person_folder_path, f"{len(os.listdir(person_folder_path))}.jpg")
        save_image(face_crop, save_path)


if __name__ == '__main__':
    # used to download retinaface before doing anything
    detect_faces_from_image(np.zeros([100, 100, 3])) 

    # loading football players and keywords
    catalog = load_catalog(catalog_path="conf/catalog.yml")

    # downloading process
    with WebDriver(file_path="utils/chromedriver.exe") as web_driver:
        for person in catalog["footballers"]:
            for keyword in catalog["keywords"]:
                print(f"Searching for: '{person} {keyword}'...")
                person_folder_name = person.replace(" ", "_").lower()
                person_folder_path = f"data/01_raw/{person_folder_name}"
                if not os.path.isdir(person_folder_path):
                    os.mkdir(person_folder_path)
                if len(os.listdir(person_folder_path)) >= 500:
                    break
                

                search_query = f"{person} {keyword}"
                try:
                    search_in_google(web_driver, search_query)
                    for image_url in search_images(web_driver):
                        if len(os.listdir(person_folder_path)) >= 500:
                            print("Limit reached")
                            break
                        try:
                            func_timeout.func_timeout(
                                20, 
                                create_faces, 
                                args=[image_url, person_folder_path]
                            )
                        except:
                            continue
                except Exception as err:
                    print(str(err))
                    continue
                print(f"Done")

