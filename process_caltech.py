import os
from os.path import join, isfile, exists, isdir
import sys
import _pickle as cPickle
from PIL import Image
import numpy as np

# TODO: Change dataset.py Correctly
# TODO: Save learning loss

train_size = 0.8
excluded_labels = ['Faces', 'BACKGROUND_Google'] # there exists Faces_easy which we only use for now

if __name__ == "__main__":
    folder = sys.argv[1]
    save_path = sys.argv[2]

    # Get labels and path_images
    labels = os.listdir(folder)
    for el in excluded_labels:
        labels.remove(el)

    label_to_image = {}
    for l in labels:
        label_path = join(folder, l)
        if isdir(label_path):
            files = os.listdir(label_path)
            flist = []
            for f in files:
                flist.append(join(label_path, f))
            label_to_image[l] = flist
    # Map label to int
    label_to_int = {}
    index = 0
    for k in label_to_image.keys():
        label_to_int[k] = index
        index += 1

    # Save all data in the following dict
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []

    # Go through images
    for l, image_paths in label_to_image.items():
        nbr_training_samples = int(len(image_paths)*train_size)

        # Training data
        for i in range(nbr_training_samples):
            p = image_paths[i]
            # Get image
            image = Image.open(p)
            arr = np.asarray(image)
            # Append to lists
            images_train.append(arr)
            labels_train.append(label_to_int.get(l))

        # Test data
        for i in range(nbr_training_samples, len(image_paths)):
            p = image_paths[i]
            # Get image
            image = Image.open(p)
            arr = np.asarray(image)
            # Append to lists
            images_test.append(arr)
            labels_test.append(label_to_int.get(l))

    dataset_dict_train = {}
    dataset_dict_train['data'] = images_train
    dataset_dict_train['labels'] = labels_train

    dataset_dict_test = {}
    dataset_dict_test['data'] = images_test
    dataset_dict_test['labels'] = labels_test

    # Save in Pickle format
    with open(join(save_path, "train"),'wb') as fp:
        cPickle.dump(dataset_dict_train,fp)

    with open(join(save_path, "test"),'wb') as fp:
        cPickle.dump(dataset_dict_test,fp)

