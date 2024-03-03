import os
import numpy as np
import pandas as pd
from skimage import transform
from skimage.color import rgb2gray
from skimage.io import imread
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def read_labels(file_name, name_images):
    labels_from_file = pd.read_csv(file_name)
    name_to_class = {image_name: class_index for image_name, class_index in zip(labels_from_file["Image"],
                                                                                labels_from_file["Class"])}
    return [name_to_class[img_name] for img_name in name_images]


def read_color_images(folder_name, images, size=64, grey=False):
    data = []
    for image in images:
        img = imread(f"{folder_name}/{image}")
        resized = transform.resize(img, (size, size))
        if grey:
            resized = rgb2gray(resized)
        data.append(resized.flatten())
    return np.array(data)


def read_train_val_data_labels(size=32, grey=False):
    train_images_names = os.listdir("/kaggle/input/unibuc-dhc-2023/train_images")
    val_images_names = os.listdir("/kaggle/input/unibuc-dhc-2023/val_images")

    x_train = read_color_images("/kaggle/input/unibuc-dhc-2023/train_images", train_images_names, grey=grey, size=size)
    x_val = read_color_images("/kaggle/input/unibuc-dhc-2023/val_images", val_images_names, grey=grey, size=size)

    y_train = read_labels("/kaggle/input/unibuc-dhc-2023/train.csv", train_images_names)
    y_val = read_labels("/kaggle/input/unibuc-dhc-2023/val.csv", val_images_names)

    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    train_images, train_labels, val_images, val_labels = read_train_val_data_labels(size=32)
    test_images = read_color_images("test_images", os.listdir("test_images"), size=32)

    # instantiere clasificator
    svm_one_vs_rest = OneVsRestClassifier(SVC(C=10))
    svm_one_vs_rest.fit(train_images, train_labels)
    # prezicerea claselor pe setul de validare
    predicted_val = svm_one_vs_rest.predict(val_images)
    # prezicerea claselor pe setul de test
    predicted_test = svm_one_vs_rest.predict(test_images)

    #functia de scriere a claselor prezise in fisierul csv e aceeasi ca la modelul de cnn propus
