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
    # cap = cv2.VideoCapture(r"D:\sample\hyundai\2023-02-07\top.avi")
    PATH = r"D:\sample\hyundai\20230208\video"
    file = "SSM Server_XNO-6120R(192.168.1.158)_20230124_204233_223531_ID_0001"
    cap = cv2.VideoCapture(PATH + r"\{}.avi".format(file))
    if not os.path.exists(PATH + r"\{}".format(file)):
        os.makedirs(PATH + r"\{}\gray".format(file))
        os.makedirs(PATH + r"\{}\rgb".format(file))

    # Set the frame index
    frame_index = 0
    capture_frame = 1
    start_frame = 16375
    start_frame = 0
    """
    "D:\sample\hyundai\2023-02-07\top.avi"
    line_sample = 80
    start_frame = 0
    thr = 415000
    thr_width = 5100
    
    start_frame = 11000
    """

    # draw line
    color = (200, 200, 200)
    thickness = 2
    line = 600
    line_sample = 200
    line_backgroud = 0
    line_total = line_sample + line_backgroud

    # threshold
    # 1650
    thr = 118000
    thr_width = 3750
    # 1000
    # thr = 385000
    # thr_width = 5100
    # 1800
    # thr = 280000
    # thr_width = 3700
    thr_background = 60

    # others
    numOfMean = 5
    numOfLine = 40
    count_save = 0
    save_img = []

    # fig = plt.figure(None ,(15, 8))
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax3 = fig.add_subplot(2, 2, (3, 4))
    # plt.tight_layout()
    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()
        start_point = (20, 0)
        print(frame_index)

        if ret:
            if frame_index % capture_frame == 0 and frame_index >= start_frame:
                # Save the frame as an image file
                # cv2.imwrite("frame_{}.jpg".format(frame_index), frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # histogram for line
                line_candidata = [gray[start_point[1], start_point[0] + i] for i in range(line)]
                for i in range(line):
                    if line_candidata[i] > thr_background:
                        start_point = (start_point[0] + i - line_backgroud, start_point[1])
                        break

                extract_line = [0 for i in range(line_total)]
                for i in range(line_total):
                    for row in range(numOfLine):
                        extract_line[i] += int(gray[start_point[1] + row, start_point[0] + i])

                # thr_width
                thr_width_accumulate = 0
                for idx in range(100, len(extract_line), 1):
                    thr_width_accumulate += extract_line[idx]
                thr_width = thr_width_accumulate / (line_sample - 100)
                # for idx, ext in enumerate(extract_line):
                #     if ext < thr_width:
                #         extract_line[idx] = thr_width

                # 초기 선언
                if frame_index == start_frame:
                    pre_start_point = start_point
                    x = np.linspace(start_frame - 200, start_frame, 201)
                    y = np.ones(shape=(x.shape[0],)) * 0
                    y_lpf = np.ones(shape=(x.shape[0],)) * 0

                # # 홈에서 멈춤
                # print("start_point : {}, pre_start_point : {}".format(start_point, pre_start_point))
                # if pre_start_point[0] - start_point[0] > 5:
                #     break
                # if frame_index % 2 == 0:
                #     pre_start_point = start_point

                width = len([i for i in extract_line if i > thr_width])
                if width < 12:
                    width = 0

                x = x + capture_frame
                # y = np.append(y, sum(extract_line))[1:]
                y = np.append(y, sum([i for i in extract_line[10:50] if i > (thr_width + 2 * numOfLine)]))[1:]
                thr = sum([thr_width + numOfLine*2 for i in extract_line[10:50] if i > (thr_width + 2 * numOfLine)])
                y_lpf = np.append(y_lpf, sum(y[-numOfMean:]) / numOfMean)[1:]


                # gray_crop = gray[start_point[1]: start_point[1] + numOfLine, start_point[0]: start_point[0] + extract_length]
                # gray_crop = cv2.normalize(gray_crop, None, 0, 255, cv2.NORM_MINMAX)
                # # alpha = 5.0
                # # gray_crop = np.clip((1 + alpha) * gray_crop - 128 * alpha, 0, 255).astype(np.uint8)
                # gray[start_point[1]: start_point[1] + numOfLine, start_point[0]: start_point[0] + extract_length] = gray_crop

                # # subplot
                # ax1.grid(True)
                # ax1.set_xlabel('pixel level')
                # ax1.set_ylabel('number')
                # ax1.set_title('Histogram')
                # ax1.set_xlim([0, 255*numOfLine])
                # ax1.set_ylim([0, line_total])
                # ax1.hist(extract_line)
                #
                # ax2.grid(True)
                # ax2.set_xlabel('pixel position')
                # ax2.set_ylabel('pixel level')
                # # ax2.set_ylim([3400, 4250])
                # ax2.set_title('width : {}pixels (threshold : {})'.format(width, thr_width))
                # ax2.plot([(start_point[0] + i) for i in range(line_total)], extract_line)
                # ax2.plot([start_point[0], start_point[0] + line_total], [thr_width, thr_width], linestyle='--')
                #
                # ax3.grid(True)
                # ax3.set_xlabel('frame')
                # ax3.set_ylabel('sum')
                # ax3.plot(x, y, label="sum")
                # ax3.set_facecolor("whitesmoke") # "https://ehclub.net/674"
                # if sum(y[-numOfMean:]) / numOfMean > thr:
                #     ax3.set_facecolor("indianred")
                # ax3.set_title('Defect Score : {0:.2f} (threshold : {1})'.format(sum(y[-numOfMean:] / numOfMean), thr))
                # ax3.plot(x, y_lpf, label="LPF")
                # ax3.plot([x[0], x[-1]], [thr, thr], linestyle = '--', linewidth = 2)
                # ax3.legend(loc='upper left')

                # image save
                count_save += 1
                save_img.append(gray)
                print(" sum(y[-numOfMean:]) / numOfMean : ", sum(y[-numOfMean:]) / numOfMean)
                if frame_index - start_frame >= 12:
                    pre_gray = save_img.pop(0)
                    if thr < (sum(y[-numOfMean:]) / numOfMean) and count_save >= 20:
                        print("===========================================")
                        cv2.imwrite(PATH + r"\{}\gray\{}_{}.png".format(file, frame_index, (sum(y[-numOfMean:]) / numOfMean)), pre_gray)
                        cv2.imwrite(PATH + r"\{}\rgb\{}_{}.png".format(file, frame_index, (sum(y[-numOfMean:]) / numOfMean)), frame)
                        count_save = 0

                gray = cv2.rectangle(gray, start_point, (start_point[0] + line_total, start_point[1] + numOfLine), color, thickness)
                cv2.imshow('gray', cv2.resize(gray, (int(gray.shape[1]/2), int(gray.shape[0]/2))))
                # cv2.imshow('pre_gray', cv2.resize(pre_gray, (int(gray.shape[1]/2), int(gray.shape[0]/2))))
                # cv2.waitKey(0)

                frame_index += 1
                # clear()

            else:
                frame_index += 1
                continue

            # plt.draw()
            # plt.tight_layout()
            # plt.pause(0.001)
            # ax1.clear()
            # ax2.clear()
            # ax3.clear()

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