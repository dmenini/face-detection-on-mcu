import cv2
from lib import *
from mtcnn_class import *


def main():
    video_capture = cv2.VideoCapture(0)
    detector = MTCNN(min_face_size=12, scale_factor=0.5, steps_threshold=[0.95, 0.7, 0.7])

    while True:
        ret, frame = video_capture.read()
        frame, factor = downscale(frame, max_size=90)
        result = detector.detect_faces(frame)
        result = clean_result(result, conf_t=0.80)
        for box in result:
            (x, y, w, h) = box['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.namedWindow('Window', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Window', frame)
        key = cv2.waitKey(2)
        if key == 27:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()