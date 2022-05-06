import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, 
                 static_image_mode=False, 
                 enable_segmentation=False, 
                 smooth_landmarks=True, 
                 model_complexity=1, 
                 smooth_segmentation=True, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.enable_segmentation = enable_segmentation 
        self.smooth_landmarks = smooth_landmarks and not static_image_mode
        self.model_complexity = model_complexity  
        self.smooth_segmentation = smooth_segmentation and not static_image_mode
        self.min_detection_confidence = min_detection_confidence 
        self.min_tracking_confidence = min_tracking_confidence 

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, 
                                     self.enable_segmentation, 
                                     self.smooth_landmarks, 
                                     self.model_complexity,
                                     self.smooth_segmentation, 
                                     self.min_detection_confidence, 
                                     self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, 
                 img, 
                 draw=True):
        imgRGB = cv2.cvtColor(img, 
                              cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, 
                                           self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

        # for id, lm in enumerate(results.pose_landmarks.landmark):
        #     h, w, c = img.shape
        #     print(id, lm)
        #     cx, cy = int(lm.x * w), int(lm.y * h)
        #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


def main():
    cap = cv2.VideoCapture('Video/4vid.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
