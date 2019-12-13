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
                'blur_label_list',  # clear->0, normal blur->1, heavy blur->2
                'event_list',  # events
                'expression_label_list',  # typical expression->0, exaggerate expression->1
                'face_bbx_list',  # x, y, w, h
                'file_list',  # names
                'illumination_label_list',  # normal illumination->0, extreme illumination->1
                'invalid_label_list',  # false->0(valid image), true->1(invalid image)
                'occlusion_label_list',  # no occlusion->0, partial->1, heavy->2
                'pose_label_list'  # typical pose->0, atypical pose->1
                ])


# ======================================================================================================================
#                                               DATASET MANAGEMENT
# ======================================================================================================================


def hdf5_store(image, path):
    # Create a new HDF5 file
    f = h5py.File(path, "w")
    # Create a dataset in the file
    f.create_dataset("image", np.shape(image), h5py.h5t.STD_U8BE, data=image)
    f.close()


def hdf5_load(path):
    with h5py.File(path, 'r') as f:
        # print("Keys in h5 dataset: %s" % f.keys())
        image = f['image']
        image = np.array(image[:, :, :])
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
            jpg_path = jpg_dir + event_dir + str(name) + '.jpg'
            h5_path = hdf5_dir + event_dir + str(name) + '.h5'
            os.mkdir(hdf5_dir) if not os.path.isdir(hdf5_dir) else None
            os.mkdir(hdf5_dir + event_dir) if not os.path.isdir(hdf5_dir + event_dir) else None
            image = cv2.imread(jpg_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            shape = image.shape
            box = [np.array(bbx) for bbx in face_bbx_list[i][j]]
            box = list(box[0])
            if not keep_all:
                b_shape = []
                b_shape += [bbx[2] * bbx[3] for bbx in box]
                if (np.array(b_shape) < 1000).all():
                    continue
            label_dict = [{'box': bbx, 'confidence': 1.0} for bbx in box]
            n_faces = len(label_dict)
            data_dict.append({'id': count, 'event': event, 'h5_path': h5_path, 'shape': shape, 'n_faces': n_faces,
                              'labels': label_dict})
            hdf5_store(image, h5_path)
            count += 1
    print(12880 - count)
    pickle_store(data_dict, 'input/' + dataset + '_dict.pickle')

    return data_dict


# ======================================================================================================================
#                                                   PLOTS
# ======================================================================================================================


def plot_comparison(image_e, image_r, label_e, label_r, color_e='r', color_r='cyan', line=1, pause=1):
    fig = plt.figure(figsize=(1, 2))
    ax = []
    ax.append(fig.add_subplot(1, 2, 1))
    ax[-1].set_title("expected:")  # set title
    plt.imshow(image_e)
    ax.append(fig.add_subplot(1, 2, 2))
    ax[-1].set_title("result:")  # set title
    plt.imshow(image_r)

    for n in range(len(label_e)):
        x, y, w, h = label_e[n]['box']
        rect = patches.Rectangle((x, y), w, h, linewidth=line, edgecolor=color_e, facecolor='none')
        ax[0].add_patch(rect)
    for n in range(len(label_r)):
        x, y, w, h = label_r[n]['box']
        rect = patches.Rectangle((x, y), w, h, linewidth=line, edgecolor=color_r, facecolor='none')
        ax[1].add_patch(rect)

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.state('zoomed')
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


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
    data_dict = [d for d in data_dict if d['n_faces'] < n_faces + 1]
    print('Filtering out images with more than {} faces...\nRemaining images: {}'.format(n_faces, len(data_dict)))
    return data_dict


def filter_box(data_dict, box_size=400, max_size=90, debug=False):
    print("Removing boxes with size < {}".format(box_size))
    clean = 0
    for d in data_dict:
        m_dim = np.amax(d['shape'])
        print("Path: ", d['h5_path']) if debug else None
        print("Before: ", d['n_faces']) if debug else None
        d['labels'] = [label for label in d['labels'] if
                       (label['box'][2] * label['box'][3]) > box_size * m_dim / max_size]
        d['n_faces'] = len(d['labels'])
        print("After: ", d['n_faces']) if debug else None
        if d['n_faces'] == 0:
            clean = 1
    if clean:
        print("Found images with no labels. Removing them...")
        init_len = len(data_dict)
        data_dict = [d for d in data_dict if d['n_faces'] != 0]
        print("Removed images:", init_len - len(data_dict))
    return data_dict


def heights_description(data_dict):
    heights = []
    for img in data_dict:
        # h changes, while w and c are the same (w=1024, c=3)
        heights.append(img['shape'][0])
    print("\nHeight description:")
    print(pd.Series(heights).describe(percentiles=[.05, .25, .5, .75, .80, .85, .90, .95]))  # Series is the type!
    return heights


def downscale(image, max_size=90):
    h, w, _ = image.shape
    rescaling_factor = np.amax([h, w]) / max_size
    factor = np.amin([h, w]) / np.amax([h, w])
    image = cv2.GaussianBlur(image, (5, 5), 0)
    if w == np.amax([h, w]):
        image = cv2.resize(image, (max_size, int(max_size * factor)))
    else:
        image = cv2.resize(image, (int(max_size * factor), max_size))
    return image, rescaling_factor


def image_padding_max_ver(image, max_size=90):
    second_size = max_size - 20
    top = (max_size - image.shape[0]) // 2
    bottom = (max_size + 1 - image.shape[0]) // 2
    left = (second_size - image.shape[1]) // 2
    right = (second_size + 1 - image.shape[1]) // 2
    image = cv2.copyMakeBorder(image, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_REPLICATE)
    return image


def image_padding_max_hor(image, max_size=90):
    second_size = max_size - 20
    top = (second_size - image.shape[0]) // 2
    bottom = (second_size + 1 - image.shape[0]) // 2
    left = (max_size - image.shape[1]) // 2
    right = (max_size + 1 - image.shape[1]) // 2
    image = cv2.copyMakeBorder(image, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_REPLICATE)
    return image


def bb_intersection_over_union(boxA, boxB, mode='metric'):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)

    # compute the intersection over union by taking the intersection area and
    # dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    if (mode is 'clean') and (interArea / boxAArea > 0.3 or interArea / boxBArea > 0.3):  # 0.5 before
        iou = 0.51

    return iou


def face_pixel_metric(result, expected, factor, height, width):
    pred = -np.ones((height, width))
    act = np.zeros((height, width))
    f_pred = -np.ones((height, width))
    f_act = np.zeros((height, width))
    r_boxes = np.array([np.array(r['box']) for r in result])
    e_boxes = np.array([np.array(e['box']) for e in expected])
    summing, count, value = 0, 0, 0

    for i, box in enumerate(r_boxes):
        tick = 0
        for j, real in enumerate(e_boxes):
            sig_val = bb_intersection_over_union(box, real / factor)
            if sig_val > 0:
                if sig_val > value:
                    value = sig_val
                pos = j
                tick = 1
        if tick == 1:
            f_pred[r_boxes[i][1]:r_boxes[i][1] + r_boxes[i][3], r_boxes[i][0]:r_boxes[i][0] + r_boxes[i][2]] = 1
            f_act[int(e_boxes[pos][1] / factor):int((e_boxes[pos][1] + e_boxes[pos][3]) / factor),
                  int(e_boxes[pos][0] / factor):int((e_boxes[pos][0] + e_boxes[pos][2]) / factor)] = 1
            summing += 1
            r_boxes[i] = 0
            e_boxes[pos] = 0
    for box in r_boxes:
        if box[0] != 0:
            count += 1
        pred[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = 1
    for box in e_boxes:
        if box[0] != 0:
            count += 1
        act[int(box[1] / factor):int((box[1] + box[3]) / factor),
            int(box[0] / factor):int((box[0] + box[2]) / factor)] = 1
    f_overlap = f_pred - f_act
    overlap = pred - act
    overlap = np.count_nonzero(overlap == 0)
    f_overlap = np.count_nonzero(f_overlap == 0)
    tot = np.count_nonzero(act)
    temp = pred - act
    tot += np.count_nonzero(temp == 1)
    try:
        acc = (f_overlap + overlap) / (f_overlap + tot)
    except:
        if overlap == 0:
            acc = 1
        else:
            acc = 0

    return acc


def n_faces_matrix(result, expected, result_matrix):
    if len(result) > 10:
        result_matrix[10, int(len(expected))] += 1
        return result_matrix
    result_matrix[int(len(result)), int(len(expected))] += 1
    return result_matrix


def clean_result(result, conf_t=0.80):
    if len(result) == 0:
        return []
    cleaned_result = []
    count = 0
    for single_box in result:
        overlap = 0.0
        for i in range(count):
            temp = bb_intersection_over_union(single_box['box'], cleaned_result[i]['box'], mode='clean')
            if temp > overlap:
                overlap = temp
                pos = i
        if overlap > 0.5 and single_box['confidence'] > cleaned_result[pos]['confidence']:
            cleaned_result[pos] = [single_box]
        elif (overlap < 0.5) and (single_box['confidence'] > conf_t):
            cleaned_result += [single_box]
            count += 1
    return cleaned_result


def generate_bb(imap, reg, scale, t=0.96):

    # use heatmap to generate bounding boxes
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])

    y, x = np.where(imap >= t)

    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)

    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

    if reg.size == 0:
        reg = np.empty(shape=(0, 3))

    bb = np.transpose(np.vstack([y, x]))

    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])

    return boundingbox


def nms(boxes, threshold, method):
    """
    Non Maximum Suppression.
    :param boxes: np array with bounding boxes.
    :param threshold:
    :param method: NMS method to apply. Available values ('Min', 'Union')
    :return:
    """
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_s = np.argsort(s)

    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while sorted_s.size > 0:
        i = sorted_s[-1]
        pick[counter] = i
        counter += 1
        idx = sorted_s[0:-1]

        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        sorted_s = sorted_s[np.where(o <= threshold)]

    pick = pick[0:counter]

    return pick


def create_boxes(partial, scale, total_boxes):
    out0 = np.transpose(partial[0], (0, 1, 2, 3))
    out1 = np.transpose(partial[1], (0, 1, 2, 3))
    bounding_boxes = generate_bb(out1[0, :, :, 1].copy(),
            out0[0, :, :, :].copy(), scale)
    print(bounding_boxes.shape)
    pick = nms(bounding_boxes.copy(), 0.5, 'Union')
    if bounding_boxes.size > 0 and pick.size > 0:
        boxes = bounding_boxes[pick, :]
        total_boxes = np.append(total_boxes, boxes, axis=0)
    return total_boxes


def rerec(bbox):
    # convert bbox to square
    height = bbox[:, 3] - bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]
    max_side_length = np.maximum(width, height)
    bbox[:, 0] = bbox[:, 0] + width * 0.5 - max_side_length * 0.5
    bbox[:, 1] = bbox[:, 1] + height * 0.5 - max_side_length * 0.5
    bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(max_side_length, (2, 1)))
    return bbox


def manage_result(complete_list):
    total_boxes = np.empty((0, 9))
    total_boxes = create_boxes(complete_list[0:2], scale=1.0, total_boxes=total_boxes)
    total_boxes = create_boxes(complete_list[2:4], scale=0.5, total_boxes=total_boxes)
    total_boxes = create_boxes(complete_list[4:6], scale=0.25, total_boxes=total_boxes)
    numboxes = total_boxes.shape[0]

    if numboxes > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]

        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]

        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())

        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)

    bounding_boxes = []

    for bounding_box in total_boxes:
        bounding_boxes.append({
            'box': [int(bounding_box[0]), int(bounding_box[1]),
                    int(bounding_box[2] - bounding_box[0]), int(bounding_box[3] - bounding_box[1])],
            'confidence': bounding_box[-1]})

    return bounding_boxes

