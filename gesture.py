import cv2
import mediapipe as mp
import pyautogui as p
import time


mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
 

tipIds = [4, 8, 12, 16, 20]

video = cv2.VideoCapture(0)

hands = mp_hand.Hands(max_num_hands=1)

flag=True
flag1=False 

while True:
     
    ret, frame = video.read()
    frame = cv2.flip(frame,2)  
    cv2.rectangle(frame, (325,0), (639,478), (0, 0, 255), 0)
    image = frame[0:478, 325:639]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

   
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cv2.rectangle(frame, (0,0), (322,478), ( 255, 0, 0), 0)
    facedet= frame[0:478,0:322]
    facedet=cv2.cvtColor(facedet, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(facedet,1.1, 4)
  
    face_check=0

 
    for(x,y,w,h) in faces:
        cv2.rectangle(facedet,(x,y), (x+w, y+h), (0,255,0),2)
        face_check=1
        flag=True        
    

    if face_check==0: 
        if flag==True:
             p.press("space")  
             flag=False 
             flag1=True
        
        continue

    elif face_check==1 and flag==True and flag1==True: 
        p.press("space")
        flag1=False
 
  




    lmList = []
    if results.multi_hand_landmarks: 
        for hand_landmark in results.multi_hand_landmarks:
            myHands = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])

            mp_draw.draw_landmarks(
                image, hand_landmark,
                mp_hand.HAND_CONNECTIONS)

    fingers = []
    if len(lmList) != 0:
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)  # 1 means open
        else:
            fingers.append(0)  # 0 means close

        # Other four fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)  # 1 means open
            else:
                fingers.append(0)  # 0 means close

        finger_count = fingers.count(1)

        if finger_count == 0:
            p.press("M")
            cv2.putText(image, "Mute", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            time.sleep(2)

        elif finger_count == 1:

            p.press("space")
            cv2.putText(image, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            time.sleep(2)

        elif finger_count == 2:
            p.press("up")

            cv2.putText(image, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        elif finger_count == 3:
            p.press("down")

            cv2.putText(image, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        elif finger_count == 4:
            p.press("right")

            cv2.putText(image, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        elif finger_count == 5:
           p.press("left")

           cv2.putText(image, "Backward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        else:
                pass

    
    cv2.namedWindow("Finger Counter", cv2.WINDOW_NORMAL)
  
    imS = cv2.resize(image, (960, 720))                    # Resize 
    cv2.imshow("Result", frame)   
    cv2.imshow("Temp", imS)
    cv2.imshow("chehra", facedet)

  

    k = cv2.waitKey(1)
    # press q for close
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()