import os
from os.path import join, isfile, exists, isdir
import sys
import _pickle as cPickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

train_size = 0.8
excluded_labels = ['BACKGROUND_Google'] # there exists Faces_easy which we only use for now
oversampled_images = ['Motorbikes', 'Faces_easy', 'Faces', 'airplanes','Leopards', 'watch']
basewidth = 224
hsize = 224

if __name__ == "__main__":
    folder = sys.argv[1]
    save_path = sys.argv[2]

    # Get labels and path_images
    labels = os.listdir(folder)
    for el in excluded_labels:
        labels.remove(el)

    print("Indexing images...")
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
        #print(k, index, len(label_to_image.get(k)))
        index += 1
       

    # Save all data in the following dict
    images_train = []
    labels_train = []
    images_test = []
    labels_test = []

    # Go through images
    print("Sorting all images...")
    for l, image_paths in label_to_image.items():

        #if l in oversampled_images:
        #    total_training_samples = 120
        #else:
        #    total_training_samples = len(image_paths)
        
        total_training_samples = len(image_paths)
        nbr_training_samples = int(0.8*total_training_samples)
        
        # Training data
        for i in range(nbr_training_samples):
            p = image_paths[i]
            # Get image
            image = Image.open(p).convert('RGB') # get RGB image
            image = image.resize((basewidth,hsize)) # rescale
            arr = np.asarray(image)
            # Append to lists
            images_train.append(arr)
            labels_train.append(label_to_int.get(l))

        # Test data
        for i in range(nbr_training_samples, total_training_samples):
            p = image_paths[i]
            # Get image
            image = Image.open(p).convert('RGB') # get RGB image
            image = image.resize((basewidth,hsize)) # rescale
            arr = np.asarray(image)
            # Append to lists
            images_test.append(arr)
            labels_test.append(label_to_int.get(l))

    print("Concatenating data...")
    # Concatenate list of arrays into single array
    images_train = np.array(images_train).reshape((-1, 3,basewidth, hsize))
    images_test = np.array(images_test).reshape((-1, 3, basewidth, hsize))

    print("Labels in test:")
    print(np.unique(labels_test, return_counts=True))

    #print("Channel-wise mean RGB of images:")
    #print(train_size*np.mean(images_train, axis=(0, 2, 3)) + (1-train_size)*np.mean(images_test, axis=(0, 2, 3)))
    #print("Channel-wise std RGB of images:")
    #print(np.std(images_test, axis=(0, 2, 3)))

    #print("Per-image mean RGB of images:")
    #print(train_size*np.mean(images_train, axis=(0,1,2,3)) + (1-train_size)*np.mean(images_test, axis=(0,1,2,3)))
    #print("Per-image std RGB of images:")
    #print(np.std(images_test, axis=(0,1,2,3)))

    for i in range(0,110):
        plt.imshow(images_train[i].reshape(224,224,3))
        plt.show()
    print("Saving data...")
    # Save in dict
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

