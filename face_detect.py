import lib
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt



def main():
    # =====================================================================
    #                      LOAD AND ANALYZE DATA
    # =====================================================================

    train_dict = lib.load2dict(dataset='train', save=False)
    val_dict = lib.load2dict(dataset='val', save=False)
    # lib.plot_box_from_dict(train_dict, max_iter=10, event='all')

    # =====================================================================
    #                           PREPROCESSING
    # =====================================================================
    train_dict = lib.filter_small_bbx(train_dict, 200, debug=False)
    # lib.plot_box_from_dict(train_dict, max_iter=10, event='all')



    # =====================================================================
    #                             DETECTION
    # =====================================================================

    detector = MTCNN()
    for i in range(10):
        image = cv2.imread(val_dict[i]['path'])
        result = detector.detect_faces(image)
        expected = val_dict[i]['labels']
        lib.plot_box(image, result, color='r', debug=True, show=False)
        lib.plot_box(image, expected, color='g', debug=True)


if __name__ == "__main__":
    main()
