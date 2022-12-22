# public-figures-recognition

Description

## Requirements

- Python 3.8


- Download the Google Chrome and the appropriate version of [ChromeDriver](https://chromedriver.chromium.org/downloads) according to your version of Google Chrome and place the "chromedriver.exe" file in the "utils" dir.


- Install needed packages


`pip install -r requirements.txt`


## Usage

### download_data.py

This script runs a Web Scraper that searches in google images for combinations of a football player names and some keywords, for example "Cristiano Ronaldo face". Then it downloads found images, and uses RetinaFace library to crop faces from the images, which are saved as separate files in folder named by fotball player's name and surname. 

NOTE: The script runs in headless mode. If you want to see web browser in action looking for images, then comment the line that adds `--headless` option in file `utils/websearching.py`
