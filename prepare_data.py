import os
import json

import numpy as np

from utils.recognition import extract_person_feature_vectors

if __name__ == "__main__":

    data_folder_path = "data/images"

    X = []
    y = []
    id_dict = {}

    current_id = 0
    for person_folder in os.listdir(data_folder_path):
        person_folder_path = os.path.join(data_folder_path, person_folder)
        if os.path.isdir(person_folder_path):
            print(f"Person: {person_folder.replace('_', ' ').title()}", end=" ")
            feature_vectors = extract_person_feature_vectors(person_folder_path)
            print(f"feature vectors shape: {feature_vectors.shape}")
            X.append(feature_vectors)
            y.append(np.full(shape=(feature_vectors.shape[0], 1), fill_value=current_id))
            id_dict[current_id] = person_folder.replace('_', ' ').title()
            current_id += 1

    X = np.concatenate(X)
    y = np.concatenate(y)
    print(f"Data shape: {X.shape}, {y.shape}")
    print(f"People identifiers: ")
    for person_id, person_name in id_dict.items():
        print(f"{person_id}: {person_name}")

    np.save("data/output/X.npy", X)
    np.save("data/output/y.npy", y)

    with open("data/output/id.json", mode="w") as f:
        f.write(json.dumps(id_dict))
