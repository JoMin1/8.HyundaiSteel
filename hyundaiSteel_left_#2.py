import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from glob import glob
from detector import run

clear = lambda : os.system('cls')


def main():
    print("start")
    # Open the video file
    # cap = cv2.VideoCapture(r"D:\sample\hyundai\용접부test용\전면_약1000.avi")
    # cap = cv2.VideoCapture(r"D:\sample\hyundai\2023-02-07\top.avi")
    files = []
    PATH = r"D:\sample\hyundai\20230208\video"
    PATH = r"D:\sample\hyundai\test"
    PATH = r"D:\sample\hyundai\2023-02-07"
    file_list = os.listdir(PATH)
    for file in file_list:  # 코드간결화 작업전
        if file.endswith(".avi"):
            files.append(file)
    # file = "SSM Server_XNO-6120R(192.168.1.158)_20230124_150339_165637_ID_0001"
    # file = "SSM Server_XNO-6120R(192.168.1.158)_20230127_111748_124921_ID_0001"

    count = 0
    files = files[1:]
    for file in files:
        if count == 0:
            count += 1
            continue
        file = os.path.splitext(file)[0]
        # file = "SSM Server_XNO-6120R(192.168.1.158)_20230126_111743_131042_ID_0001"
        cap = cv2.VideoCapture(PATH + r"\{}.avi".format(file))
        if not os.path.exists(PATH + r"\{}".format(file)):
            # os.makedirs(PATH + r"\{}\760\gray".format(file))
            # os.makedirs(PATH + r"\{}\760\rgb".format(file))
            # os.makedirs(PATH + r"\{}\380\gray".format(file))
            # os.makedirs(PATH + r"\{}\380\rgb".format(file))
            os.makedirs(PATH + r"\{}\380\rgbTest_defect".format(file))
            os.makedirs(PATH + r"\{}\total\rgbTest_defect".format(file))
            os.makedirs(PATH + r"\{}\380\rgbTest_good".format(file))
            os.makedirs(PATH + r"\{}\total\rgbTest_good".format(file))

        run(cap, PATH, file)
        count += 1

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