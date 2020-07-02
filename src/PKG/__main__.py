import cv2
import time
import tensorflow as tf
from PKG.models import ResnetV2
from PKG.models import CascadeDetector
from PKG.models import Regressor
from PKG.models import Classifier



cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector = CascadeDetector(1.3, 5)
# recording = False
if __name__ == "__main__":
    X = tf.random.uniform((64, 96, 96, 1))
    # net = ResnetV2()
    # net = Regressor(68)
    net = Classifier(7)
    print(net(X).shape)
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         t1 = time.perf_counter()
    #         #############################
    #         detectedImg = detector(frame, visualize=True)
    #         #############################
    #         t2 = time.perf_counter()
    #         cv2.putText(frame, f"FPS: {int(1.0/(t2-t1))}", (25, 25),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 15), 1, cv2.LINE_AA)
    #         cv2.imshow("detection", frame)
    #         k = cv2.waitKey(1)
    #         if k & 0xFF == ord('q'):
    #             break
    #         elif k & 0xFF == ord('p'):
    #             k = cv2.waitKey(0)
    #     else:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()