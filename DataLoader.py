import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import os
import glob
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


class DataLoader(Sequence):
    def __init__(self, size, mask_size, batch_size):
        self.batch_size = batch_size
        self.size = size
        self.mask_size = mask_size
        self.ROOT = '/media/bonilla/My Book/Gi4e'
        self.images = glob.glob(os.path.join(self.ROOT, 'images', '*'))
        self.annotations = glob.glob(os.path.join(self.ROOT, 'labels', '*'))

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential([
            iaa.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-15, 15), mode='reflect'),
            iaa.Fliplr(0.5),
            iaa.MultiplyAndAddToBrightness(),
            sometimes(iaa.CoarseDropout(p=0.05, per_channel=True)),
            sometimes(iaa.JpegCompression(compression=(0, 25))),
            sometimes(iaa.MedianBlur(k=(1, 3)))
        ])

    def __len__(self):
        return int(np.floor(len(self.annotations) / self.batch_size))

    def __getitem__(self, index):
        X = []
        Y = []
        while len(X) < self.batch_size:
            ann = np.random.choice(self.annotations)
            with open(ann, 'r') as file:
                data = file.read().split('\n')
            d = np.random.choice(data)
            stuff = d.split('\t')
            name = stuff[0]
            coords = stuff[1:]
            try:
                x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = map(lambda x: int(float(x)), coords)
            except ValueError:
                continue
            color = cv2.imread(os.path.join(self.ROOT, 'images', name))
            image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
            if len(faces) == 0:
                continue
            x, y, w, h = faces[0]
            roi_gray = image[y:y + h, x:x + w]
            roi_color = color[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 2:
                continue
            try:
                ex, ey, ew, eh = eyes[np.random.randint(1), :]
            except TypeError:
                continue
            eye = roi_color[ey: ey + eh, ex: ex + ew]

            mask = np.zeros((eye.shape[0], eye.shape[1]), np.uint8)
            cv2.circle(mask, (x2 - x - ex, y2 - y - ey), 2, (255, 255, 255), -1)
            cv2.circle(mask, (x5 - x - ex, y5 - y - ey), 2, (255, 255, 255), -1)
            eye = cv2.resize(eye, self.size)
            mask = cv2.resize(mask, self.mask_size).astype('float32') / 255.

            eye, mask = self.seq(image=eye, heatmaps=mask[np.newaxis, :, :, np.newaxis])
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

            X.append(np.expand_dims(eye, axis=-1).astype('float32') / 255.)
            # X.append(eye)
            Y.append(mask[0])
        return np.array(X, np.float32), np.array(Y, np.float32)


if __name__ == '__main__':
    data = DataLoader((64, 64), (32, 32), 16)
    print(len(data))
    for _ in range(10):
        d = data[0]
        plt.figure(0)
        plt.imshow(d[0][0, :, :, 0])

        plt.figure(1)
        plt.imshow(d[1][0, :, :, 0])

        plt.figure(2)
        i1 = np.uint8(d[0][0, :, :, 0] * 255.)
        i2 = np.uint8(d[1][0, :, :, 0] * 255.)
        i2 = cv2.resize(i2, data.size)
        plt.imshow(cv2.addWeighted(i1, 0.5, i2, 0.5, 1))

        plt.show()
    print()
