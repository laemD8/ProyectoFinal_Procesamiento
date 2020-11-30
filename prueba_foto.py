import cv2
import numpy as np
import math

def detect_eyes(img, classifier):
    global bounding_lefteye, bounding_righteye
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame, 1.3, 5) # detect eyes
    width = np.size(img, 1) # get face frame width
    height = np.size(img, 0) # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y < height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            print('left_eye',x, y, w, h)
            left_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
        else:
            print('right_eye',x, y, w, h)
            right_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
    return left_eye, right_eye

def detect_faces(img, classifier):
    global biggest
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        print('face', x, y, w, h)
        frame = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w,y+h), 255, 2)
    return frame

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(img, 5) #3
    keypoints = detector.detect(img)
    return keypoints



def main():
    photo = cv2.imread('C:/Users/karen/Documents/Ingenieriaelectronica/NovenoSemestre/Proyectodegrado 1/Codigos/Puntoscara/Distancias/foto1.jpg')
    face_frame = detect_faces(photo, face_cascade)
    if face_frame is not None:
        eyes = detect_eyes(face_frame, eye_cascade)
        for eye in eyes:
            if eye is not None:
                #eye = cut_eyebrows(eye)
                gray_frame = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                threshold = 40
                ret, Ibw_manual = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
                opening = cv2.morphologyEx(Ibw_manual, cv2.MORPH_OPEN, (5, 5))
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (5, 5))
                cv2.imshow('prueb', closing)
                cv2.imshow('prueba', Ibw_manual)
                contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                for idx, cont in enumerate(contours):
                    #cv2.drawContours(eye, contours, idx, (0, 255, 255), 5)
                    M = cv2.moments(contours[idx])
                    area = int(M['m00'])
                    print(area)
                    if area > 50 and area <1000:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        print('center_eyes', cx, cy)
                        radio = int(math.sqrt(area/math.pi))
                        cv2.circle(eye, (cx, cy), radio, (0, 255, 0), 2)
                        cv2.circle(eye, (cx, cy), 1, (0, 0, 255), 3)
                #_, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
                # Blur using 3 * 3 kernel.
                # gray_blurred = cv2.blur(gray_frame, (3, 3))
                #
                # # Apply Hough transform on the blurred image.
                # detected_circles = cv2.HoughCircles(gray_blurred,
                #                                     cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                #                                    param2=30, minRadius=1, maxRadius=40)
                # # Draw circles that are detected.
                # if detected_circles is not None:
                #
                #     # Convert the circle parameters a, b and r to integers.
                #     detected_circles = np.uint16(np.around(detected_circles))
                #
                #     for pt in detected_circles[0, :]:
                #         a, b, r = pt[0], pt[1], pt[2]
                #         print('center_eyes', a, b)
                #         # Draw the circumference of the circle.
                #         cv2.circle(eye, (a, b), r, (0, 255, 0), 2)
                #
                #         # Draw a small circle (of radius 1) to show the center.
                #         cv2.circle(eye, (a, b), 1, (0, 0, 255), 3)
                #         cv2.imshow("Detected Circle", eye)
                        # keypoints = blob_process(eye, threshold, detector)
                # eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image', photo)
    cv2.imshow('image1', face_frame)
    cv2.imshow('image2', eyes[0])
    cv2.imshow('image3', eyes[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def nothing(x):
    pass

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    main()