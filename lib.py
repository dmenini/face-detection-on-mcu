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


def initialize(dataset, keep_all=True):
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
            box = [np.array(bbx) for bbx in face_bbx_list[i][j]]
            box = list(box[0])
            if not keep_all:
                b_shape = []
                b_shape += [bbx[2]*bbx[3] for bbx in box]
                if (np.array(b_shape) < 1000).all():
                    continue
            label_dict = [{'box': bbx, 'confidence': 1.0} for bbx in box]
            n_faces = len(label_dict)
            data_dict.append({'id': count, 'event': event, 'h5_path': h5_path, 'shape': shape, 'n_faces': n_faces, 'labels': label_dict })
            hdf5_store(image, h5_path)
            count += 1
    print(12880-count)
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

def too_many_faces(data_dict, n_faces):
    data_dict = [d for d in data_dict if d['n_faces']<n_faces+1]
    print('remained images', len(data_dict))
    return data_dict

def filter_box(data_dict, box_size, debug=False):
    print("Removing boxes with size < {}".format(box_size))
    clean = 0
    for d in data_dict:
        m_dim = np.amax(d['shape'])
        print("Path: ", d['h5_path']) if debug else None
        print("Before: ", d['n_faces']) if debug else None
        d['labels'] = [label for label in d['labels'] if (label['box'][2]*label['box'][3]) > box_size * m_dim / 250]
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


def heights_description(data_dict):
    heights = []
    for img in data_dict:
        # h changes, while w and c are the same (w=1024, c=3)
        heights.append(img['shape'][0])
    print("\nHeight description:")
    print(pd.Series(heights).describe(percentiles=[.05, .25, .5, .75, .80, .85, .90, .95]))  # Series is the type!
    return heights

def downscale(image):
    h, w, _ = image.shape
    max_size = 250
    rescaling_factor = np.amax([h, w]) / max_size
    factor = np.amin([h, w]) / np.amax([h, w])
    image = cv2.GaussianBlur(image, (3, 3), 0)
    if w == np.amax([h, w]):
        image = cv2.resize(image, (max_size, int(max_size*factor)))
    else:
        image = cv2.resize(image, (int(max_size*factor), max_size))
    return image, rescaling_factor

def bb_intersection_over_union1(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def face_metric1(result, expected, factor, height, width):
    pred = - np.ones((height, width))
    act = np.zeros((height, width))
    f_pred = - np.ones((height, width))
    f_act = np.zeros((height, width))
    r_boxes = np.array([np.array(r['box']) for r in result])
    e_boxes = np.array([np.array(e['box']) for e in expected])
    summing = 0
    count = 0
    value = 0
    for i, box in enumerate(r_boxes):
        tick = 0
        for j, real in enumerate(e_boxes):
            sig_val = bb_intersection_over_union1(box, real/factor)
            if sig_val > 0:
                if sig_val > value:
                    value = sig_val
                pos = j
                tick = 1
        if tick == 1:
            f_pred[r_boxes[i][1]:r_boxes[i][1]+r_boxes[i][3], r_boxes[i][0]:r_boxes[i][0]+r_boxes[i][2]] = 1
            f_act[int(e_boxes[pos][1]/factor):int((e_boxes[pos][1]+e_boxes[pos][3])/factor), int(e_boxes[pos][0]/factor):int((e_boxes[pos][0]+e_boxes[pos][2])/factor)] = 1
            summing += 1
            r_boxes[i] = 0
            e_boxes[pos] = 0
    for box in r_boxes:
        if box[0] != 0:
            count += 1
        pred[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1
    for box in e_boxes:
        if box[0] != 0:
            count += 1
        act[int(box[1]/factor):int((box[1]+box[3])/factor), int(box[0]/factor):int((box[0]+box[2])/factor)] = 1
    f_overlap = f_pred-f_act
    overlap = pred-act
    overlap = np.count_nonzero(overlap==0)
    f_overlap = np.count_nonzero(f_overlap==0)
    tot = np.count_nonzero(act)
    temp = pred-act
    tot += np.count_nonzero(temp==1)
    try:
        acc = (f_overlap + overlap) / (f_overlap + tot)
        #acc = (summing + count * acc) / (summing + count)
    except:
        if overlap==0:
            acc = 1
        else:
            acc = 0
    #print(acc)
    return acc

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    if interArea / boxAArea > 0.5 or interArea / boxBArea > 0.5:
        iou = 0.51

    # return the intersection over union value
    return iou

def face_metric2(result, expected, factor, height, width):
    r_boxes = np.array([np.array(r['box']) for r in result])
    e_boxes = np.array([np.array(e['box']) for e in expected])
    overlap = 0
    for pre in r_boxes:
        temp = 0
        for exp in e_boxes:
            if bb_intersection_over_union(pre, exp/factor) > 0:
                temp=1
        overlap += temp
    tot = e_boxes.shape[0]
    acc = overlap / tot
    #print(acc)
    return acc

def face_metric3(result, expected, result_matrix):
    if len(result) > 10:
        result_matrix[10, int(len(expected))] += 1
        return result_matrix
    result_matrix[int(len(result)), int(len(expected))] += 1
    return result_matrix

def face_metric(result, expected, factor, h, w):
    #print(result)
    r_boxes = np.array([np.array(r['box']) for r in result])
    e_boxes = np.array([np.array(e['box']) for e in expected])
    #print(r_boxes.shape)
    tempr = np.zeros((2, 4))
    tempe = np.zeros((2, 4))
    try:
        tempr[0, :] = r_boxes[:] if len(r_boxes.shape) == 1 else r_boxes
    except:
        r_boxes = tempr[0, :] if len(r_boxes.shape) == 1 else r_boxes
    try:
        tempe[0, :] = e_boxes[:] if len(e_boxes.shape) == 1 else e_boxes
    except:
        e_boxes = tempe[0, :] if len(e_boxes.shape) == 1 else e_boxes
    r_boxes = tempr if len(r_boxes.shape) == 1 else r_boxes
    e_boxes = tempe if len(e_boxes.shape) == 1 else e_boxes
    acc = (np.dot(np.transpose(r_boxes[:, 2]*factor), r_boxes[:, 3]*factor))/((np.dot(np.transpose(e_boxes[:, 2]), e_boxes[:, 3])))
    if np.isnan(acc) or acc==float("inf"):
        acc=1
    print("Num faces accuracy", acc)
    return acc

def clean_result(result):
    #r_boxes = np.array([np.array(r['box']) for r in result])
    if len(result) == 0:
        return []
    cleaned_result = []
    #cleaned_result += [result[0]]
    count = 0
    #if len(result) == 1:
    #    return cleaned_result
    for single_box in result[0:]:
        overlap = 0.0
        for i in range(count):
            temp = bb_intersection_over_union(single_box['box'], cleaned_result[i]['box'])
            if temp > overlap:
                overlap = temp
                pos = i
        if overlap > 0.4 and single_box['confidence'] > cleaned_result[pos]['confidence']:
            cleaned_result[pos] = [single_box]
        elif overlap < 0.4 and single_box['confidence'] > 0.4:
            cleaned_result += [single_box]
            count += 1
    return cleaned_result






