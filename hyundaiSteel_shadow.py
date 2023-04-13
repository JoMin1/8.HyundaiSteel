import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

clear = lambda : os.system('cls')


def main():
    print("start")
    # Open the video file
    # cap = cv2.VideoCapture(r"D:\sample\hyundai\용접부test용\전면_약1000.avi")
    cap = cv2.VideoCapture(r"D:\sample\hyundai\2023-02-07\top.avi")
    # PATH = r"D:\sample\hyundai\20230208\video"
    # file = "SSM Server_XNO-6120R(192.168.1.158)_20230125_092526_111824_ID_0001"
    # cap = cv2.VideoCapture(PATH + r"\{}.avi".format(file))
    # if not os.path.exists(PATH + r"\{}".format(file)):
    #     os.makedirs(PATH + r"\{}\gray".format(file))
    #     os.makedirs(PATH + r"\{}\rgb".format(file))

    # Set the frame index
    frame_index = 0


    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()

        if ret:
            blue = frame[:, :, 0].copy()
            blue = blue.astype(np.int16)
            red = frame[:, :, 2].copy()
            red = red.astype(np.int16)
            frame = (2 * blue - red).copy()
            frame = frame.astype(np.uint8)
            rgb_planes = cv2.split(frame)

            result_planes = []
            result_norm_planes = []
            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 255 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_8UC1)
                result_planes.append(diff_img)
                result_norm_planes.append(norm_img)

            result = cv2.merge(result_planes)
            result_norm = cv2.merge(result_norm_planes)
            cv2.imshow("result", cv2.resize(result, (int(result.shape[1]), int(result.shape[0]))))
            cv2.imshow("bg_img", cv2.resize(bg_img, (int(bg_img.shape[1]), int(bg_img.shape[0]))))
            cv2.imshow("frame", cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))))
            cv2.waitKey(0)

            toc = time.time()
            print("inference time : ", (toc - tic))

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