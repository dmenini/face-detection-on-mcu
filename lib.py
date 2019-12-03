import pickle
from scipy.io import loadmat
from itertools import chain

from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def load2dict(dataset, save=False):
    """
    Load input dataset to dictionary
    :param dataset:
    :param save: whether to save the dictionary to pickle file
    :return: list of dictionaries --> keys: int = 'id', str = 'path', int = 'n_faces', list of dictionaries = 'labels'{'box, 'confidence'}
    """
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
            path = './input/WIDER_' + str(dataset) + '/images/' + str(event) + '/' + str(name) + '.jpg'
            # image = cv2.imread(path)
            box = [np.array(bbx) for bbx in face_bbx_list[i][j]]
            box = list(box[0])
            label_dict = [{'box': bbx, 'confidence': 1.0} for bbx in box]

            n_faces = len(label_dict)
            data_dict.append({'id': count, 'event': event, 'path': path, 'n_faces': n_faces, 'labels': label_dict})
            count += 1

    if save:
        file = open('input/' + str(dataset) + '_dict.pickle', 'wb')
        pickle.dump(data_dict, file)

    return data_dict


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
            image = Image.open(d['path'])
            print("Path:", d['path'], "\t\tSize:", image.size)
            plot_box(image, d['labels'])
            if i == max_iter - 1:
                break


def filter_small_bbx(data_dict, box_size, debug=False):
    print("Removing boxes with size < {}".format(box_size))
    clean = 0
    for d in data_dict:
        print("Path: ", d['path']) if debug else None
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





