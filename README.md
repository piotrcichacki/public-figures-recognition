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


### clean_data.py

When faces are downloaded, this script can be run to clean the dataset. It involves looking for duplicates as well as outliers (which may be faces of a different person) and deleting them from the dataset.


### prepare_data.py

This step converets images to data ready to be supplied as an input for a model training, testing and validation. Two files are saved: `data/output/X.npy` which represents a matrix with input data, each row representing one photo), and `data/output/y.npy` which are names and surnames (labels) for photos.
