import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    success, img = cap.read()

    # Detect hands in the frame
    hands, img = detector.findHands(img)  # This method returns the image and a list of hands found

    # Process the detected hands
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1['lmList']  # List of 21 Landmark points
        bbox1 = hand1['bbox']  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1['type']  # Handtype Left or Right

        # Draw landmarks
        for lm in lmList1:
            cv2.circle(img, lm, 5, (255, 0, 0), cv2.FILLED)

    # Display the image
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
