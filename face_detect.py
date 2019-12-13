from lib import *
from mtcnn_class import *
import cv2
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


def main():
    # =====================================================================
    #                      LOAD AND STORE DATA
    # =====================================================================

    # DO ONLY FIRST TIME!!
    # train_dict = initialize(dataset='train', keep_all=False)
    # val_dict = initialize(dataset='val')

    train_dict = pickle_load('input/train_dict.pickle')
    val_dict = pickle_load('input/val_dict.pickle')
    print("Total train images: ", len(train_dict))
    print("Total val images: ", len(val_dict))

    # EXAMPLE: HOW TO ACCESS AN IMAGE IN h5 AND PLOT IT WITH BOXES
    # image = lib.hdf5_load(train_dict[1]['h5_path'])
    # lib.plot_box(image, train_dict[1]['labels'])

    # To plot the boxes on the image from the dicts
    # lib.plot_box_from_dict(train_dict, max_iter=10, event='all')

    train_heights = heights_description(train_dict)
    val_heights = heights_description(val_dict)

    # =====================================================================
    #                           PREPROCESSING
    # =====================================================================

    h_cut = int(pd.Series(train_heights).quantile(q=0.95))
    train_dict = [img for img in train_dict if img['shape'][0] <= h_cut]
    val_dict = [img for img in val_dict if img['shape'][0] <= h_cut]

    train_dict = filter_box(train_dict, 1000, debug=False)
    # val_dict = filter_box(val_dict, 140, debug=False)
    val_dict = too_many_faces(val_dict, 7)
    val_dict = filter_box(val_dict, 400, debug=False)

    # lib.plot_box_from_dict(train_dict, max_iter=5, event='all')

    # =====================================================================
    #                             DETECTION
    # =====================================================================

    print("Final train images: ", len(train_dict))
    print("Final val images: ", len(val_dict))
    p_net = load_model('./model/p_net.h5')
    acc = []
    result_matrix = np.zeros((11, 11))
    detector = MTCNN(min_face_size=12, scale_factor=0.5)
    for i in range(0, len(val_dict[:])):
        expected = val_dict[i]['labels']
        if len(expected) > 7 or len(expected) == 0:
            continue
        image = hdf5_load(val_dict[i]['h5_path'])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_ds, factor = downscale(image)
        if image_ds.shape[1]>70:
            continue
        image_ds = cv2.copyMakeBorder(image_ds, top=(90-image_ds.shape[0])//2, bottom=(91-image_ds.shape[0])//2, left=(70-image_ds.shape[1])//2, right=(71-image_ds.shape[1])//2, borderType=cv2.BORDER_REPLICATE)
        image_ds1 = np.expand_dims(image_ds, axis=0)
        image_ds1 = (image_ds1 - 127.5) / 128.0
        complete_list = p_net.predict(image_ds1)
        result = manage_result(complete_list)
        #result = detector.detect_faces(image_ds)
        result = clean_result(result)
        result_matrix = face_metric3(result, expected, result_matrix)
        acc.append(face_metric1(result, expected, factor, image_ds.shape[0], image_ds.shape[1]))
        print(image_ds.shape, image.shape)
        plot_box(image_ds, result, color='cyan', debug=False, show=True)
        #plot_box(image, expected, color='r', debug=False)
    print("Pixel metric accuracy", np.mean(np.array(acc)))

    # Plot result matrix
    result_matrix = result_matrix / np.tile((np.sum(result_matrix, axis=0).astype(float)+0.001), (11, 1))
    for i in range(result_matrix.shape[1]-1):
        result_matrix[i, i] = result_matrix[i-1, i] + result_matrix[i, i] + result_matrix[i+1, i]
    result_m = result_matrix[1:6, 1:6]
    sn.heatmap(result_m, annot=True)
    plt.show()


if __name__ == "__main__":
    main()
