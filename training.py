import os
import cv2
import numpy as np
from PIL import Image

recognize = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"


def get_images_with_id(path):
    imaged = [os.path.join(path, f) for f in os.listdir(path)]
    Faces = []
    Id = []

    for impath in imaged:
        faced = Image.open(impath).convert('L')
        faceNp = np.array(faced,'uint8')
        ID = int(os.path.split(impath)[-1].split('.')[1])
        Faces.append(faceNp)
        Id.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return Id, Faces


Ids, faces = get_images_with_id(path)
recognize.train(faces, np.array(Ids))
recognize.save(r'C:\Users\FAMI\PycharmProjects\Facial recognition\trainingData.yml')
cv2.destroyAllWindows()
