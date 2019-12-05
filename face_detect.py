from lib import *
from mtcnn_class import *
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


def main():
    # =====================================================================
    #                      LOAD AND STORE DATA
    # =====================================================================

    # DO ONLY FIRST TIME!!
    #train_dict = initialize(dataset='train', keep_all=False)
    #val_dict = initialize(dataset='val')

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
    #val_dict = filter_box(val_dict, 400, debug=False)

    #lib.plot_box_from_dict(train_dict, max_iter=5, event='all')

    # =====================================================================
    #                             DETECTION
    # =====================================================================

    print("Final train images: ", len(train_dict))
    print("Final val images: ", len(val_dict))

    acc = []
    detector = MTCNN(min_face_size=12, scale_factor=0.36)
    for i in range(0,len(val_dict)):
        image = hdf5_load(val_dict[i]['h5_path'])
        image1, factor = downscale(image)
        result = detector.detect_faces(image1)
        result = clean_result(result)
        expected = val_dict[i]['labels']
        acc.append(face_metric1(result, expected, factor, image1.shape[0], image1.shape[1]))
        #plot_box(image1, result, color='cyan', debug=False, show=False)
        #plot_box(image, expected, color='r', debug=False)
    print(np.mean(np.array(acc)))


if __name__ == "__main__":
    main()
