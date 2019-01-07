import os
import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

def process_image(img, img_size, grayscale=False):
    """
    Loads an image from file, resize to img_size and return as numpy array.
    Values are normalized to range [0,1].
    :param img: image path
    :param img_size: image size as tuple (width, height)
    :return: ndarray (width, height, 3) or (width, height)
    """
    if grayscale == True:
        img = load_img(img, target_size=img_size, color_mode='grayscale')
    else:
        img = load_img(img, target_size=img_size, color_mode='rgb')
    img = img_to_array(img)
    img = img/255
    img = img.clip(0, 1)
    return img


def get_dataset(train_path, test_path, test_labels_path, img_size, grayscale):
    """
    Builds the dataset from file.
    :param grayscale: set True to use grayscale images
    :param train_path: directory for training images
    :param test_path: directory for test images
    :param test_labels_path: directory for class labels of the test set
    :param img_size image size (with, height)
    :return: subsets splitted in train test and images and labels.
    """
    # read training set and labels
    file_name_list = []
    class_labels = []
    for (dirpath, dirnames, filenames) in os.walk(train_path):
        for directory in dirnames:
            # print(os.path.join(dirpath, directory))
            splitted = os.path.join(dirpath, directory).split("/")
            # class labels from 0 to 42 as int instead of "00000" to "00042"
            label = int(splitted[len(splitted) - 1])
            class_labels.append(label)
            for root, directories, filenames in os.walk(os.path.join(dirpath, directory)):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if filename.endswith('.ppm'):
                        file_name_list.append(filepath)
    # build training set
    xtrain = []
    ytrain = []
    for file_name in file_name_list:
        image = process_image(file_name, img_size, grayscale)
        xtrain.append(image)
        splitted = file_name.split("/")
        label = splitted[len(splitted) - 2]
        ytrain.append(label)
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain, dtype=int)
    print("number of classes: {}".format(len(class_labels)))
    print("shape of train images: {}".format(xtrain.shape))
    print("shape of labels: {}".format(ytrain.shape))
    # build test set
    x = os.path.join(os.getcwd(),test_path)
    file_name_list_test = list(filter(lambda f : f.endswith(".ppm"), os.listdir(x)))
    print("number of test images: {}".format(len(file_name_list_test)))
    xtest = np.zeros((len(file_name_list_test), xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))
    ytest = np.zeros((len(file_name_list_test)), dtype=int)
    import csv
    label_dict = {} # (file_name, class label)
    with open(test_labels_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        i = 0
        for line in reader:
            if i == 0:
                i += 1
            else:
                label_dict[line[0]] = line[7]
                i += 1
    for ix, file in enumerate(file_name_list_test):
        xtest[ix] = process_image(os.path.join(test_path ,file), (64,64), grayscale)
        ytest[ix] = int(label_dict[file])
    print("shape of test images: {}".format(xtest.shape))
    print("shape of test labels: {}".format(ytest.shape))
    # shuffle data with fixed random state for retraining and reproducibility
    from sklearn.utils import shuffle
    xtrain, ytrain = shuffle(xtrain, ytrain, random_state=25)
    xtest, ytest = shuffle(xtest, ytest, random_state=25)
    return xtrain, ytrain, xtest, ytest