import numpy as np 
import matplotlib.pyplot as plt 
import os 
import cv2
from tqdm import tqdm
import random
import pickle

"""
DATADIR = "/home/dorter/Downloads/PetImages"

CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 100

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print("general exception", e, os.path.join(path, img))


create_training_data()
print(len(training_data))


random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Let's save this data, so that we don't need to keep calculating it every time we want to play with the neural network model:

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

"""

# We can always load it in to our current script, or a totally new one by doing:

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)


