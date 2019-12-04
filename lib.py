import os
import cv2
import numpy as np
import h5py
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.io import loadmat
from itertools import chain


labels_keys = (['__header__', '__version__', '__globals__',  # Attributes
                'blur_label_list',                              # clear->0, normal blur->1, heavy blur->2
                'event_list',                                   # events
                'expression_label_list',                        # typical expression->0, exaggerate expression->1
                'face_bbx_list',                                # x, y, w, h
                'file_list',                                    # names
                'illumination_label_list',                      # normal illumination->0, extreme illumination->1
                'invalid_label_list',                           # false->0(valid image), true->1(invalid image)
                'occlusion_label_list',                         # no occlusion->0, partial->1, heavy->2
                'pose_label_list'                               # typical pose->0, atypical pose->1
                ])

# ======================================================================================================================
#                                               DATASET MANAGEMENT
# ======================================================================================================================

def hdf5_store(image, path):
    # Create a new HDF5 file
    f = h5py.File(path, "w")
    # Create a dataset in the file
    data_set = f.create_dataset("image", np.shape(image), h5py.h5t.STD_U8BE, data=image)
    f.close()


def hdf5_load(path):
    with h5py.File(path, 'r') as f:
        # print("Keys in h5 dataset: %s" % f.keys())
        image = f['image']
        image = np.array(image[:,:,:])
    return image


def pickle_store(obj, path):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()


def pickle_load(path):
    f = open(path, 'rb')
    return pickle.load(f)


def initialize(dataset):
    print("Generating " + str(dataset) + " dictionary...")
    face_split = loadmat('input/wider_face_split/wider_face_' + str(dataset) + '.mat')
    event_list = list(chain.from_iterable(face_split['event_list']))
    file_list = list(chain.from_iterable(face_split['file_list']))
    face_bbx_list = list(chain.from_iterable(face_split['face_bbx_list']))
    data_dict = []
    count = 0

    for i in range(len(file_list)):  # Iterate over events
        event = ''.join(str(el) for el in event_list[i])
        for j in range(len(file_list[i])):  # Iterate over elements of events (names or labels)
            name = ''.join(str(el) for el in file_list[i][j])
            name = name[2:len(name) - 2]
            jpg_dir = 'input/WIDER_' + str(dataset) + '/' + '/images/'
            hdf5_dir = 'input/hdf5_' + str(dataset) + '/'
            event_dir = str(event) + '/'
            jpg_path = jpg_dir  + event_dir + str(name) + '.jpg'
            h5_path = hdf5_dir + event_dir + str(name) + '.h5'
            os.mkdir(hdf5_dir) if not os.path.isdir(hdf5_dir) else None
            os.mkdir(hdf5_dir+event_dir) if not os.path.isdir(hdf5_dir+event_dir) else None
            image = cv2.imread(jpg_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape = image.shape
            hdf5_store(image, h5_path)
            box = [np.array(bbx) for bbx in face_bbx_list[i][j]]
            box = list(box[0])
            label_dict = [{'box': bbx, 'confidence': 1.0} for bbx in box]
            n_faces = len(label_dict)
            data_dict.append({'id': count, 'event': event, 'h5_path': h5_path, 'shape': shape, 'n_faces': n_faces, 'labels': label_dict })
            count += 1

    pickle_store(data_dict, 'input/' + dataset + '_dict.pickle')

    return data_dict

# ======================================================================================================================
#                                                   PLOTS
# ======================================================================================================================

def plot_box(image, labels, color='r', debug=False, show=True):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for n in range(len(labels)):
        print(labels) if debug else None
        x, y, w, h = labels[n]['box']
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
    plt.show() if show else None


def plot_box_from_dict(data_dict, event='all', max_iter=None):
    for i, d in enumerate(data_dict):
        if (d['event'] == str(event)) or (str(event) == 'all'):
            image = hdf5_load(d['h5_path'])
            print("Path:", d['h5_path'], "\t\tDimensions:", image.shape)
            plot_box(image, d['labels'])
            if i == max_iter - 1:
                break

# ======================================================================================================================
#                                               PREPROCESSING
# ======================================================================================================================

def filter_box(data_dict, box_size, debug=False):
    print("Removing boxes with size < {}".format(box_size))
    clean = 0
    for d in data_dict:
        print("Path: ", d['h5_path']) if debug else None
        print("Before: ", d['n_faces']) if debug else None
        d['labels'] = [label for label in d['labels'] if (label['box'][2]*label['box'][3]) > box_size]
        d['n_faces'] = len(d['labels'])
        print("After: ", d['n_faces']) if debug else None
        if d['n_faces'] == 0:
            clean = 1
    if clean:
        print("Found images with no labels. Removing them...")
        init_len = len(data_dict)
        data_dict = [d for d in data_dict if d['n_faces'] != 0]
        print("Removed images:", init_len-len(data_dict))
    return data_dict


def heights_description(data_dict, ):
    heights = []
    for img in data_dict:
        # h changes, while w and c are the same (w=1024, c=3)
        heights.append(img['shape'][0])
    print("\nHeight description:")
    print(pd.Series(heights).describe(percentiles=[.05, .25, .5, .75, .80, .85, .90, .95]))  # Series is the type!
    return heights










