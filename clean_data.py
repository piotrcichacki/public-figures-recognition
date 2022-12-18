import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from utils.recognition import extract_person_feature_vectors


def search_for_duplicates(matrix, threshold):
    
    duplicates_indices = np.argwhere(matrix <= threshold).tolist()
    duplicates_images_indices = [higher_index for lower_index, higher_index in duplicates_indices if higher_index > lower_index]
    duplicates_images_indices_unique = list(set(duplicates_images_indices))
    
    return duplicates_images_indices_unique


def search_for_outliers(matrix, threshold):

    outliers_indices = np.where(np.mean(matrix, axis=1) > threshold)[0].tolist()

    return outliers_indices


def remove_images(image_folder_path, indices):
    for idx, image_file in enumerate(sorted(os.listdir(image_folder_path), key=lambda file_name: int(file_name.split(".")[0]))):
        image_file_path = os.path.join(image_folder_path, image_file)

        if idx in indices:
            os.remove(image_file_path)


def collect_dataset_statistics(folder_path, df, phase):
    for person_folder in os.listdir(folder_path):
        person_folder_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            df.loc[len(df), :] = [phase, person_folder.replace('_', ' ').title(), len(os.listdir(person_folder_path))]
    
    return df


if __name__ == "__main__":

    dataset_statistics = pd.DataFrame(columns=["Data preprocessing phase", "person", "images_num"])
    similarity_threshold = 0.01
    outliers_threshold = 0.75
    data_folder_path = "data/images"

    dataset_statistics = collect_dataset_statistics(data_folder_path, dataset_statistics, phase="Web scraping")

    for person_folder in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            print(f"Person: {person_folder.replace('_', ' ').title()}")
            feature_vectors = extract_person_feature_vectors(person_folder_path)
            print(f"Feature vectors shape: {feature_vectors.shape}")
            distances_matrix = cosine_distances(feature_vectors, feature_vectors)
            print(f"Distance matrix shape: {distances_matrix.shape}")
            duplicated_images_indices = search_for_duplicates(distances_matrix, similarity_threshold)
            print(f"Number of duplicated images: {len(duplicated_images_indices)}")
            remove_images(person_folder_path, duplicated_images_indices)

    dataset_statistics = collect_dataset_statistics(data_folder_path, dataset_statistics, phase="Remove duplicates")

    for person_folder in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            print(f"Person: {person_folder.replace('_', ' ').title()}")
            feature_vectors = extract_person_feature_vectors(person_folder_path)
            print(f"Feature vectors shape: {feature_vectors.shape}")
            distances_matrix = cosine_distances(feature_vectors, feature_vectors)
            print(f"Distance matrix shape: {distances_matrix.shape}")
            outliers_indices = search_for_outliers(distances_matrix, outliers_threshold)
            print(f"Number of outliers images: {len(outliers_indices)}")
            remove_images(person_folder_path, outliers_indices)

    dataset_statistics = collect_dataset_statistics(data_folder_path, dataset_statistics, phase="Remove outliers")

    dataset_statistics.to_csv("data/data_statistics.csv", index=False)