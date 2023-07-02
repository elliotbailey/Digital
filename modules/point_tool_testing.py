import cv2

from point_tools import PointCloud
from camera import CameraManager

def main():

    cm = CameraManager()

    while True:
        output = cm.process_hands()
        cv2.imshow("Output", output.frame)
        if output.hand_landmarks is not None:
            point_cloud = PointCloud.from_mphands(output.hand_landmarks[0])

        keypress = cv2.waitKey(1)
        keypress = chr(keypress) if keypress != -1 else None
        match keypress:
            case "q":
                return False
            case "p":
                del cm

if __name__ == "__main__":
    main()