import lib
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # =====================================================================
    #                      LOAD AND STORE DATA
    # =====================================================================

    # DO ONLY FIRST TIME!!
    # train_dict = lib.initialize(dataset='train')
    # val_dict = lib.initialize(dataset='val')

    train_dict = lib.pickle_load('input/train_dict.pickle')
    val_dict = lib.pickle_load('input/val_dict.pickle')
    print("Total train images: ", len(train_dict))
    print("Total val images: ", len(val_dict))

    # EXAMPLE: HOW TO ACCESS AN IMAGE IN h5 AND PLOT IT WITH BOXES
    # image = lib.hdf5_load(train_dict[1]['h5_path'])
    # lib.plot_box(image, train_dict[1]['labels'])

    # To plot the boxes on the image from the dicts
    # lib.plot_box_from_dict(train_dict, max_iter=10, event='all')

    train_heights = lib.heights_description(train_dict)
    val_heights = lib.heights_description(val_dict)

    # =====================================================================
    #                           PREPROCESSING
    # =====================================================================

    h_cut = int(pd.Series(train_heights).quantile(q=0.95))
    train_dict = [img for img in train_dict if img['shape'][0] <= h_cut]
    val_dict = [img for img in val_dict if img['shape'][0] <= h_cut]

    train_dict = lib.filter_box(train_dict, 200, debug=False)
    lib.plot_box_from_dict(train_dict, max_iter=5, event='all')

    # =====================================================================
    #                             DETECTION
    # =====================================================================

    print("Final train images: ", len(train_dict))
    print("Final val images: ", len(val_dict))

    detector = MTCNN()
    for i in range(3):
        image = lib.hdf5_load(val_dict[i]['h5_path'])
        result = detector.detect_faces(image)
        expected = val_dict[i]['labels']
        lib.plot_box(image, result, color='g', debug=False, show=False)
        lib.plot_box(image, expected, color='r', debug=False)


if __name__ == "__main__":
    main()
