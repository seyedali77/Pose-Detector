import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, modelcomplexity=1, smoothlandmark=True, enableseg=False,
                 smoothseg=True, detectioncon=0.5, trackingcon=0.5):

        # Initialize pose estimation parameters
        self.mode = mode
        self.modelcomplexity = modelcomplexity
        self.smoothlandmark = smoothlandmark
        self.enableseg = enableseg
        self.smoothseg = smoothseg
        self.detectioncon = detectioncon
        self.trackingcon = trackingcon

        # Load MediaPipe pose detection model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelcomplexity, self.smoothlandmark,
                                     self.enableseg, self.smoothseg, self.detectioncon,
                                     self.trackingcon)

    def findPose(self, img, draw=True):
        # Convert image to RGB format (required by MediaPipe)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                # Draw pose landmarks on the image
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosotion(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # Convert normalized landmark coordinates to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    # Draw a circle at the landmark position
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture("pose videos/6.mp4")
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break  # Stop loop if video ends or there's an issue

        img = detector.findPose(img)
        lmList = detector.findPosotion(img, draw=False)

        if len(lmList) != 0:
            # Highlight a specific landmark (e.g., elbow at index 14)
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Resize and display the image
        img_resized = cv2.resize(img, (800, 600))
        cv2.imshow("Image", img_resized)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()












