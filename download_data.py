import os

from utils.detection import detect_faces_from_image, crop_image
from utils.utils import load_yaml
from utils.websearching import WebDriver, search_in_google, search_images, download_image, \
    ErrorDuringDownloadingImage, save_image

if __name__ == '__main__':

    catalog = load_yaml(file_path="conf/catalog.yml")

    with WebDriver(file_path="utils/chromedriver.exe") as web_driver:
        for person in catalog["footballers"]:
            for keyword in catalog["keywords"]:
                person_folder_name = person.replace(" ", "_").lower()
                person_folder_path = f"data/01_raw/{person_folder_name}"
                if not os.path.isdir(person_folder_path):
                    os.mkdir(person_folder_path)

                search_query = f"{person} {keyword}"
                search_in_google(web_driver, search_query)
                for image_url in search_images(web_driver):
                    try:
                        image = download_image(image_url)
                    except ErrorDuringDownloadingImage as error:
                        continue

                    detected_faces = detect_faces_from_image(image)
                    for face_box in detected_faces:
                        face_crop = crop_image(image, face_box)
                        save_path = os.path.join(person_folder_path, f"{len(os.listdir(person_folder_path))}.jpg")
                        save_image(face_crop, save_path)
