import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from glob import glob
from detector import Detector
# import detector_ing

def main():
    print("start")
    # Open the video file
    # cap = cv2.VideoCapture(r"D:\sample\hyundai\용접부test용\전면_약1000.avi")
    # cap = cv2.VideoCapture(r"D:\sample\hyundai\2023-02-07\top.avi")
    files = []
    PATH = r"D:\sample\hyundai\20230208\video"
    # PATH = r"D:\sample\hyundai\2023-02-07"
    file_list = os.listdir(PATH)
    for file in file_list:  # 코드간결화 작업전
        if file.endswith(".avi"):
            files.append(file)

    files = files[1:]
    for file in files:
        file = os.path.splitext(file)[0]
        file = "SSM Server_XNO-6120R(192.168.1.158)_20230126_073150_092448_ID_0001"
        # file = "bot_test2"
        cap = cv2.VideoCapture(PATH + r"\{}.avi".format(file))
        if not os.path.exists(PATH + r"\{}".format(file)):
            os.makedirs(PATH + r"\{}\380_left\gray".format(file))
            os.makedirs(PATH + r"\{}\380_left\rgb".format(file))
            os.makedirs(PATH + r"\{}\380_right\gray".format(file))
            os.makedirs(PATH + r"\{}\380_right\rgb".format(file))
            os.makedirs(PATH + r"\{}\380_total\gray".format(file))
            os.makedirs(PATH + r"\{}\380_total\rgb".format(file))
            os.makedirs(PATH + r"\{}\error")

        start(cap, PATH, file)
        # detector_ing.run(cap, PATH, file)



def start(cap, PATH, file):
    # # Set the frame index
    cur_frame = 0
    start_frame = 1000

    initial_left_point = (70, 500)
    initial_right_point = (1850, 500)

    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output1.avi', fourcc, 30.0, (1920, 1080))

    fig = plt.figure(None, (15, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax3_1 = ax3.twinx()
    ax4_1 = ax4.twinx()

    color = (0, 0, 200)
    thickness = 3

    detector_left = Detector(start_frame, cur_frame)
    detector_right = Detector(start_frame, cur_frame)

    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(detector_left.cur_frame)

        if ret:
            # left
            left_point, left_numOfDefectPixel = detector_left.detect(frame, gray, initial_left_point, isLeft=True)
            if left_numOfDefectPixel > detector_left.ignore_edge or detector_left.isHole:
                frame = cv2.rectangle(frame,
                                      (initial_left_point[0] + left_point[0] - 5, left_point[1] - int(detector_left.numOfLine / 2) - 5),
                                      (initial_left_point[0] + left_point[0] + detector_left.check_area_defectScore[0] + left_numOfDefectPixel + 5,
                                       left_point[1] + int(detector_left.numOfLine / 2) + 5), color, thickness)
                gray = cv2.rectangle(gray,
                                     (initial_left_point[0] + left_point[0] - 5, left_point[1] - int(detector_left.numOfLine / 2) - 5),
                                     (initial_left_point[0] + left_point[0] + detector_left.check_area_defectScore[0] + left_numOfDefectPixel + 5,
                                      left_point[1] + int(detector_left.numOfLine / 2) + 5), color, thickness)
                # detector_left.save(PATH, file, frame, gray, left_point, left_numOfDefectPixel, color, thickness, isLeft=True)

            # right
            right_point, right_numOfDefectPixel = detector_right.detect(frame, gray, initial_right_point, isLeft=False)
            if right_numOfDefectPixel > detector_right.ignore_edge or detector_right.isHole:
                frame = cv2.rectangle(frame,
                                      (initial_right_point[0] - right_point[0] - detector_right.check_area_defectScore[0] - right_numOfDefectPixel - 5,
                                       right_point[1] - int(detector_right.numOfLine / 2) - 5),
                                      (initial_right_point[0] - right_point[0] + 5, right_point[1] + int(detector_right.numOfLine / 2) + 5), color, thickness)
                gray = cv2.rectangle(gray,
                                     (initial_right_point[0] - right_point[0] - detector_right.check_area_defectScore[0] - right_numOfDefectPixel - 5,
                                      right_point[1] - int(detector_right.numOfLine / 2) - 5),
                                     (initial_right_point[0] - right_point[0] + 5, right_point[1] + int(detector_right.numOfLine / 2) + 5), color, thickness)
                # detector_right.save(PATH, file, frame, gray, right_point, right_numOfDefectPixel, color, thickness, isLeft=False)
            cv2.imshow('frame', cv2.resize(frame, (int(frame.shape[1] / 4 * 3), int(frame.shape[0] / 4 * 3))))
            cv2.waitKey(1)

            ax1, ax3, ax3_1 = detector_left.visualize(left_point, left_numOfDefectPixel, ax1, ax3, ax3_1)
            ax2, ax4, ax4_1 = detector_right.visualize(right_point, right_numOfDefectPixel, ax2, ax4, ax4_1, isLeft=False)
            plt.draw()
            plt.tight_layout()
            plt.pause(0.001)
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            ax3_1.cla()
            ax4_1.cla()

            # out.write(frame)
            toc = time.time()
            print("total time : ", (toc - tic))
        else:
            break

    # Release the video file and destroy all windows
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()


# import cv2
# import numpy as np
#
# # Create a VideoCapture object
# cap = cv2.VideoCapture(r"D:\sample\현대제철\용접부test용\전면_약1000.avi")
#
# # Get the first frame of the video
# _, first_frame = cap.read()
#
# # Convert the first frame to grayscale
# prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
#
# # Create a mask to store the foreground
# mask = np.zeros_like(prev_gray)
#
# # Set a background subtractor
# back_sub = cv2.createBackgroundSubtractorMOG2()
#
# while True:
#     # Read the next frame
#     ret, frame = cap.read()
#
#     # Break the loop if the video ends
#     if not ret:
#         break
#
#     # Convert the frame to grayscale
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Use the background subtractor to get the foreground
#     fg_mask = back_sub.apply(frame)
#
#     # Apply morphological transformations to refine the mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
#
#     # Update the mask with the foreground
#     print(fg_mask.shape)
#     print(frame.shape)
#     print(mask.shape)
#     mask = np.where(fg_mask==255, frame, mask)
#
#     # Display the frame--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#     cv2.imshow("frame", frame)
#     cv2.imshow("mask", mask)
#
#     # Break the loop if the user presses the "q" key
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break
#
# # Release the VideoCapture object
# cap.release()
#
# # Destroy all windows
# cv2.destroyAllWindows()


# import cv2
#
# cap = cv2.VideoCapture(r"D:\sample\현대제철\용접부test용\전면_약1000.avi")
# fgbg = cv2.createBackgroundSubtractorMOG2()
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#
#     fgmask = fgbg.apply(frame)
#     fgmask = cv2.erode(fgmask, None, iterations=2)
#     fgmask = cv2.dilate(fgmask, None, iterations=2)
#
#     output = cv2.bitwise_and(frame,frame,mask = fgmask)
#
#     out.write(output)
#     cv2.imshow("output",output)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()