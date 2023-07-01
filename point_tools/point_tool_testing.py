import mediapipe as mp
import cv2

from point_tools import PointCloud

def main_loop(cap, mpHands, hands, mpDraw):
    _, frame = cap.read()

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)

    point_cloud = None
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            point_cloud = PointCloud.from_mphands(handLms.landmark)
            #print(point_cloud.points)
    
    cv2.imshow("Output", frame)

    # Keyboard input
    keypress = cv2.waitKey(1)
    if keypress != -1:
        keypress = chr(keypress)
        match keypress:
            case "q":
                return False
            case "p":
                if point_cloud is not None:
                    point_cloud.plot()


    return True

def main():

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.8
    )
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    while True:
        if not main_loop(cap, mpHands, hands, mpDraw):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()