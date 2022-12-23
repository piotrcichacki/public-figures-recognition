import cv2
import numpy as np
import pafy
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from utils.detection import detect_faces_from_image, crop_image
from utils.recognition import extract_feature_vector

if __name__ == "__main__":

    video_id = "BTKL8MKJ0O0"
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = f"data/videos/{video_id}.mp4"

    video = pafy.new(video_url)
    stream = video.getbest(preftype="mp4")
    stream.download(filepath=video_path, quiet=False)

    classifier = tf.keras.models.load_model('saved_model/best_model.h5')

    df = pd.DataFrame(columns=["frame_0", "frame_1", "person_id"])
    video_capture = cv2.VideoCapture(video_path)
    frame_counter = 0
    while video_capture.isOpened():

        frame_counter += 1
        has_frame, frame = video_capture.read()

        if not has_frame:
            video_capture.release()
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detected_faces = detect_faces_from_image(frame)
        for detected_face in detected_faces:
            face_crop = crop_image(frame, detected_face)
            feature_vector = extract_feature_vector(face_crop)
            prediction = classifier.predict(feature_vector)
            person_id = np.argmax(prediction[0])
            df.loc[len(df), :] = [frame_counter - 1, frame_counter, person_id]

    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(30, 10)
    axes.set_title("Footballers appearance on the movie", fontsize=18)
    axes.barh(df.person_id, df.frame_1 - df.frame_0, left=df.frame_0)
    axes.set_xlabel("Frame number")
    axes.set_ylabel("Footballer id")
    axes.set_yticks(list(range(20)))
    axes.grid(False)
    plt.show()

    fig.savefig(f"data/plots/{video_id}.png")


