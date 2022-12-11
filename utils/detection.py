from retinaface import RetinaFace


class BoundingBox:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @property
    def coordinates_xyxy(self):
        return self.left, self.top, self.right, self.bottom

    @property
    def coordinates_xywh(self):
        return self.left, self.top, self.right - self.left, self.bottom - self.top

    @property
    def resolution(self):
        return self.bottom - self.top, self.right - self.left

    @property
    def min_resolution(self):
        return min(self.resolution)

    @property
    def max_resolution(self):
        return max(self.resolution)


def detect_faces_from_image(image, threshold=0.8, min_resolution=80, max_resolution=400):
    detected_faces = []
    detections = RetinaFace.detect_faces(image)
    try:
        for detection in detections.values():
            if detection["score"] > threshold:
                left, top, right, bottom = detection["facial_area"]
                face_box = BoundingBox(left=left, top=top, right=right, bottom=bottom)
                if face_box.min_resolution > min_resolution and face_box.max_resolution < max_resolution:
                    detected_faces.append(face_box)
    except AttributeError:
        pass
    return detected_faces


def crop_image(image, bounding_box):
    left, top, right, bottom = bounding_box.coordinates_xyxy
    crop = image[top:bottom, left:right, :]
    return crop
