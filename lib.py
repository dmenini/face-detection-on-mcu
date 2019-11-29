import pickle
from scipy.io import loadmat
from itertools import chain

import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
    print("Generating " + str(dataset) + " dictionary")
    face_split = loadmat('input/wider_face_split/wider_face_' + str(dataset) + '.mat')
    event_list = list(chain.from_iterable(face_split['event_list']))
    file_list = list(chain.from_iterable(face_split['file_list']))
    face_bbx_list = list(chain.from_iterable(face_split['face_bbx_list']))
    set_dict = []
    count = 0

    for i in range(len(file_list)):  # Iterate over events
        event = ''.join(str(el) for el in event_list[i])
        for j in range(len(file_list[i])):  # Iterate over elements of events (names or labels)
            name = ''.join(str(el) for el in file_list[i][j])
            name = name[2:len(name) - 2]
            path = './input/WIDER_' + str(dataset) + '/images/' + str(event) + '/' + str(name) + '.jpg'
            # image = mpimg.imread(path)
            labels = [label for label in face_bbx_list[i][j]]
            labels = labels[0]
            n_faces = len(labels)
            set_dict.append(
                {'id': count, 'event': event, 'name': name, 'path': path, 'n_faces': n_faces, 'labels': labels})
            count += 1

    if save:
        file = open('input/' + str(dataset) + '_dict.pickle', 'wb')
        pickle.dump(set_dict, file)

    return set_dict


def plot_patches(set_dict, event='all', max_iter=None):
    for i, d in enumerate(set_dict):
        if (d['event'] == str(event)) or (str(event) == 'all'):
            img = Image.open(d['path'])
            fig, ax = plt.subplots()
            ax.imshow(img)
            for n in range(d['n_faces']):
                rect_point = (d['labels'][n][0], d['labels'][n][1])
                w = d['labels'][n][2]
                h = d['labels'][n][3]
                rect = patches.Rectangle(rect_point, w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.show()
            if i == max_iter - 1:
                break


