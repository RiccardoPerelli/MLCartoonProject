from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def extract_face(filename, required_size=(224, 224)):
    pixels = plt.imread(filename)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array



