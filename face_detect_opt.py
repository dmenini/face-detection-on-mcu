from lib import *
from mtcnn_class import *
import cv2
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

n_faces = 7
max_size = 90
scale_factor = 0.5
confidence = 0.80


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

    train_dict = filter_box(train_dict, box_size=1000, max_size=max_size, debug=False)
    val_dict = too_many_faces(val_dict, n_faces=n_faces)
    val_dict = filter_box(val_dict, box_size=400, max_size=max_size, debug=False)

    # lib.plot_box_from_dict(train_dict, max_iter=5, event='all')

    # =====================================================================
    #                             DETECTION
    # =====================================================================

    print("Final train images: ", len(train_dict))
    print("Final val images: ", len(val_dict))

    acc = []
    result_matrix = np.zeros((11, 11))      # safe value
    detector = MTCNN(min_face_size=12, scale_factor=scale_factor)

    for i in range(0, len(val_dict[:])):
        expected = val_dict[i]['labels']
        if len(expected) > n_faces or len(expected) == 0:
            continue
        image = hdf5_load(val_dict[i]['h5_path'])
        image_ds, factor = downscale(image)
        result = detector.detect_faces(image_ds)
        result = clean_result(result, conf_t=confidence)
        result_matrix = n_faces_matrix(result, expected, result_matrix)
        acc.append(face_pixel_metric(result, expected, factor, image_ds.shape[0], image_ds.shape[1]))
        plot_comparison(image, image_ds, expected, result, color_e='r', color_r='cyan', line=1, pause=1.5)

    # =====================================================================
    #                          PLOT RESULTS
    # =====================================================================

    print("Pixel metric accuracy", np.mean(np.array(acc)))

    # Plot result matrix
    result_matrix = result_matrix / np.tile((np.sum(result_matrix, axis=0).astype(float)+0.001), (11, 1))
    diagonal = np.zeros((result_matrix.shape[1]-1, result_matrix.shape[1]-1))
    for i in range(result_matrix.shape[1]-1):
        diagonal[i, i] = result_matrix[i, i] + result_matrix[i - 1, i] + result_matrix[i + 1, i]
    result_matrix_cut = result_matrix[0:n_faces+1, 0:n_faces+1]
    diagonal_cut = diagonal[1:n_faces+1, 1:n_faces+1]
    sn.heatmap(result_matrix_cut, annot=True)
    plt.show()
    sn.heatmap(diagonal_cut, annot=True, xticklabels=False, yticklabels=False)
    plt.show()


if __name__ == "__main__":
    main()
