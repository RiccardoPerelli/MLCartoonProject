from utils import *
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import glob
import os

device = "cpu"
#device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

from torchvision import transforms

def portrait_segmentation(filename, required_size=(512, 512)):

    img = Image.open(filename)

    width, height = img.size
    left = 918
    top = 0
    right = 3690
    bottom = height - 300

    img = img.crop((left, top, right, bottom))
    img = img.resize(required_size)

    mask = get_mask(img, model)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    not_mask = np.bitwise_not(mask)
    portrait = np.bitwise_or(img, not_mask)
    return portrait

def get_mask(img, model):
    '''trasforma l'immagine in input per renderla compatibile al modello e ritornare la maschera'''
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = tf(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    output = output.argmax(0)

    return infer_op(output)

def infer_op(op):
    '''output convertito in tensore cpu, numpy array, e scalato in un range da [0 a 1] a [0 a 255]'''
    op = op.byte().cpu().numpy()
    op = op * 255
    op.astype(np.uint8)
    return op

# IMAGE PRE PROCESSING
uncropped_real_all_set = glob.glob('../RealImages/all2/*.jpg')
uncropped_real_train_set = glob.glob('../RealImages/train2/*.jpg')
print("Dataset Real Image preprocessing...")
count = 1;
for filename in uncropped_real_all_set:
    print("Converting image n°: ")
    print(count)
    count += 1
    if not os.path.isfile('../RealImages/all_cropped2' + os.path.basename(filename)):
        image = plt.imread(filename)
        image = portrait_segmentation(filename)
        plt.imsave('../RealImages/all_cropped2/' + os.path.basename(filename), image)

for filename in uncropped_real_train_set:
    if not os.path.isfile('../RealImages/train_cropped2' + os.path.basename(filename)):
        image = plt.imread(filename)
        image = portrait_segmentation(filename)
        plt.imsave('../RealImages/train_cropped2/' + os.path.basename(filename), image)

'''
for filename in uncropped_set:
    pixels = plt.imread(filename)
    pixels = fr.extract_face(filename)
    plt.imsave('../RealImages/test_cropped/' + os.path.basename(filename), pixels)
'''
'''
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
    return face_array'''



