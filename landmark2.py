import dlib
import cv2
import numpy as np
import math
import json
from camera_model import *

def nothing(x):
    pass

def puntos(event, x, y, flags, param):
    global teo, a, b
    if event == cv2.EVENT_LBUTTONDOWN:
        teo = (48 * dist) / fx
        print(teo)
        a = 1
        b =1

def dato(dist, fx):
    if a == 1:
        distancia = (teo * fx) / dist
        return distancia
    else:
        distancia = 0
        return distancia
if __name__ == '__main__':
    global a, b
    a = 0
    file_name = 'C:/Users/karen/Documents/Ingenieriaelectronica/NovenoSemestre/Procesamientodeimagenes/CamaraComputador/calibration.json'  # Se lee el archivo.js
    with open(file_name) as fp:
        json_data = json.load(fp)

    # intrinsics parameters
    K = np.array(json_data['K'])
    fx = K[0][0]
    fy = K[1][1]
    width = int(K[0][2]) * 2
    height = int(K[1][2]) * 2
    cx = 320
    cy = 240

    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    cx2=0
    cx1=0
    cy2=0
    b=0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    while True:
        _, frame = cap.read()
        if b==0:
            cv2.rectangle(frame, (170, 90), (170 + 300, 90 + 300), 255, 2)

        cv2.setMouseCallback('image', puntos)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = predictor(img, d)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            for (x, y) in coords:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            ojos_i = coords[36:41, :]
            ojos_d = coords[42:47, :]
            frame1 = frame.copy()

            xi1 = ojos_i[0, 0] - 10
            yi1 = ojos_i[1, 1] - 20
            xi2 = ojos_i[3, 0] + 10
            yi2 = ojos_i[4, 1] + 20
            im_gray_eye = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            ret, im_bw = cv2.threshold(im_gray_eye, 200, 255, cv2.THRESH_BINARY_INV)
            cv2.fillPoly(im_bw, [ojos_i], 0)
            cv2.fillPoly(im_bw, [ojos_d], 0)
            eye = frame[yi1:yi2, xi1:xi2]
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye_con = im_bw[yi1:yi2, xi1:xi2]
            mask_eye = cv2.bitwise_not(eye_con)
            mask = cv2.bitwise_and(eye, mask_eye)
            mask_f = cv2.bitwise_or(mask, eye_con)

            threshold = cv2.getTrackbarPos('threshold', 'image')
            ret, Ibw_manual = cv2.threshold(mask_f, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('image2', Ibw_manual)
            closing = cv2.morphologyEx(Ibw_manual, cv2.MORPH_CLOSE, (5, 5))
            contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for idx, cont in enumerate(contours):
                # cv2.drawContours(eye, contours, idx, (0, 255, 255), 5)
                M = cv2.moments(contours[idx])
                area = int(M['m00'])
                if area > 100 and area < 1000:
                    cx1 = int(M['m10'] / M['m00'])
                    cy1 = int(M['m01'] / M['m00'])
                    radio = int(math.sqrt(area / math.pi))
                    #cv2.circle(frame, (cx + xi1, cy + yi1), radio, (0, 255, 0), 2)
                    cv2.circle(frame, (cx1 + xi1, cy1 + yi1), 3, (0, 0, 255), 3)

            xj1 = ojos_d[0, 0] - 10
            yj1 = ojos_d[1, 1] - 20
            xj2 = ojos_d[3, 0] + 10
            yj2 = ojos_d[4, 1] + 20
            im_gray_eye = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            ret, im_bw = cv2.threshold(im_gray_eye, 200, 255, cv2.THRESH_BINARY_INV)
            cv2.fillPoly(im_bw, [ojos_i], 0)
            cv2.fillPoly(im_bw, [ojos_d], 0)
            eye = frame[yj1:yj2, xj1:xj2]
            eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            eye_con = im_bw[yj1:yj2, xj1:xj2]
            mask_eye = cv2.bitwise_not(eye_con)
            mask = cv2.bitwise_and(eye, mask_eye)
            mask_f = cv2.bitwise_or(mask, eye_con)
            threshold = cv2.getTrackbarPos('threshold', 'image')
            ret, Ibw_manual = cv2.threshold(mask_f, threshold, 255, cv2.THRESH_BINARY)
            cv2.imshow('image2', Ibw_manual)
            closing = cv2.morphologyEx(Ibw_manual, cv2.MORPH_CLOSE, (5, 5))
            contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for idx, cont in enumerate(contours):
                # cv2.drawContours(eye, contours, idx, (0, 255, 255), 5)
                M = cv2.moments(contours[idx])
                area = int(M['m00'])
                if area > 100 and area < 1000:
                    cx2 = int(M['m10'] / M['m00'])
                    cy2 = int(M['m01'] / M['m00'])
                    radio = int(math.sqrt(area / math.pi))
                    #cv2.circle(frame, (cx + xi1, cy + yi1), radio, (0, 255, 0), 2)
                    cv2.circle(frame, (cx2 + xj1, cy2 + yj1), 3, (0, 0, 255), 3)

            dist = (cx2 + xj1) - (cx1 + xi1)
            dist2 = cy2 + yj1
            centro = (cx1 + xi1) + (dist/2)
            distancia = dato(dist,fx)
            if centro > cx:
                rica_x = ((centro-cx) * distancia) / fx
            else:
                rica_x = ((cx-centro) * distancia) / fx
                rica_x = -rica_x
            if dist2 > cy:
                print(dist2-cy)
                h = ((dist2-cy) * distancia) / fy
            else:
                h = ((cy-dist2) * distancia) / fy
                h = -h
                print(cy-dist2)
            # extrinsics parameters
            d = distancia
            R = set_rotation(0, 0, 0)
            t = np.array([rica_x, d, h])
            camera = projective_camera(K, width, height, R, t)

            square_3D = np.array(
                [[10, 10, 0], [10, -10, 0], [-10, -10, 0], [-10, 10, 0], [10, 10, 10], [10, -10, 10],
                 [-10, -10, 10], [-10, 10, 10]])
            square_2D = projective_camera_project(square_3D, camera)

            image_projective = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)
            cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[1][0], square_2D[1][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[2][0], square_2D[2][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[3][0], square_2D[3][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[0][0], square_2D[0][1]),
                     (200, 1, 255), 3)

            cv2.line(image_projective, (square_2D[4][0], square_2D[4][1]), (square_2D[5][0], square_2D[5][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[5][0], square_2D[5][1]), (square_2D[6][0], square_2D[6][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[6][0], square_2D[6][1]), (square_2D[7][0], square_2D[7][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[7][0], square_2D[7][1]), (square_2D[4][0], square_2D[4][1]),
                     (200, 1, 255), 3)

            cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[4][0], square_2D[4][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[5][0], square_2D[5][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[6][0], square_2D[6][1]),
                     (200, 1, 255), 3)
            cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[7][0], square_2D[7][1]),
                     (200, 1, 255), 3)

            cv2.imshow("cubo", image_projective)
            font = cv2.FONT_HERSHEY_SIMPLEX
            mensaje = "Distancia =" + str(round(distancia,2)) + "cm"
            cv2.putText(frame, mensaje, (10,70), font, 1, (255,0,0),2,cv2.LINE_AA)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()