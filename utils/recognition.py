import os

import PIL
import numpy as np
from PIL import Image
from arcface import ArcFace

arcface = ArcFace.ArcFace()


def extract_feature_vector(face_crop):
    features_vector = np.array(arcface.calc_emb(face_crop), dtype=np.float32)
    features_vector = np.expand_dims(features_vector, axis=0)
    return features_vector


def extract_person_feature_vectors(image_folder_path):
    output_feature_vectors = []

    for image_file in sorted(os.listdir(image_folder_path), key=lambda file_name: int(file_name.split(".")[0])):
        image_file_path = os.path.join(image_folder_path, image_file)
        try:
            image = np.array(Image.open(image_file_path))
        except PIL.UnidentifiedImageError:
            os.remove(image_file_path)
            continue
        feature_vector = extract_feature_vector(image)
        output_feature_vectors.append(feature_vector)

    return np.squeeze(np.stack(output_feature_vectors), axis=1)
