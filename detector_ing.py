import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from glob import glob

clear = lambda : os.system('cls')


def run(cap, PATH, file):
    # Set the frame index
    frame_index = 0
    capture_frame = 1
    start_frame = 0

    # draw line
    color = (200, 200, 200)
    thickness = 2
    line_check = 700
    line_sample = 150
    line_backgroud = 0
    line_total = line_sample + line_backgroud

    # others
    numOfMean = 3
    numOfLine = 30
    count_save = 0
    save_img = []
    check_area_thr = (5, 50)
    check_area_thr_width = 100
    conv = [0, 1, 2, 3, 4, 5, 6, 7, 8, 102, 103, 104]
    isHole = False

    # threshold
    thr_upper = 80 * len(conv)
    limit_upper = 150
    thr_lower = 110 * len(conv)
    limit_lower = 150
    thr_mid = 100
    limit_mid = 255

    # fig = plt.figure(None ,(15, 8))
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax3 = fig.add_subplot(2, 2, (3, 4))
    # ax4 = ax3.twinx()
    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()
        # check_line = (20, 0)
        upper_point = (30, 10)
        lower_point = (30, 1050)
        start_point = (70, 600)
        print(frame_index)

        if ret:
            if frame_index % capture_frame == 0 and frame_index >= start_frame:
                # Save the frame as an image file
                # cv2.imwrite("frame_{}.jpg".format(frame_index), frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blue = frame[:,:,0].copy()

                # update start point
                # upper_point = get_startPoint(gray, conv, upper_point, line_check, thr_upper, limit_upper)
                # print()
                # lower_point = get_startPoint(gray, conv, lower_point, line_check, thr_lower, limit_lower)
                # start_point = (lower_point[0] + int((upper_point[0] - lower_point[0])*3/5), lower_point[1] + int((upper_point[1]-lower_point[1])*3/5))

                # start_point = get_startPoint(gray, conv, start_point, line_check, thr_mid, limit_mid)

                # # get pixel level of line
                # line_level = [0 for i in range(line_total)]
                # for i in range(line_total):
                #     for row in range(numOfLine):
                #         line_level[i] += int(gray[start_point[1] + row, start_point[0] + i])

                # update start point
                # tic_line = time.time()
                # extract_line_BR = [0 for i in range(line_check)]
                # extract_line_B = extract_line_BR.copy()
                # extract_line_R = extract_line_BR.copy()
                # blue_line = extract_line_BR.copy()
                # for i in range(line_check):
                #     for row in range(numOfLine):
                #         extract_line_B[i] += frame[start_point[1] + row - int(numOfLine/2), start_point[0] + i][0]
                #         extract_line_R[i] += frame[start_point[1] + row - int(numOfLine/2), start_point[0] + i][2]
                #         blue_line[i] += blue[start_point[1] + row - int(numOfLine/2), start_point[0] + i] / numOfLine
                #     extract_line_BR[i] = int((2 * extract_line_B[i] - extract_line_R[i]) / numOfLine)

                extract_line_B = np.sum(frame[start_point[1] - int(numOfLine / 2): start_point[1] + int(numOfLine / 2), start_point[0]: start_point[0] + line_check, 0] / numOfLine, axis=0)
                extract_line_R = np.sum(frame[start_point[1] - int(numOfLine / 2): start_point[1] + int(numOfLine / 2), start_point[0]: start_point[0] + line_check, 2] / numOfLine, axis=0)
                blue_line = np.sum(blue[start_point[1] - int(numOfLine / 2): start_point[1] + int(numOfLine / 2), start_point[0]: start_point[0] + line_check] / numOfLine, axis=0)
                extract_line_BR = list(2 * extract_line_B - extract_line_R)
                for i in range(line_check):
                    if extract_line_BR[i] > thr_mid:
                        start_point = (i, start_point[1])
                        break
                line_level = blue_line[start_point[0]:  start_point[0] + line_sample]
                # toc_line = time.time()
                # print(toc_line - tic_line)

                # get threshold
                # thr_width_accumulate = 0
                # for idx in range(check_area_thr_width, line_sample, 1):
                thr_width = np.mean(line_level[check_area_thr_width : line_sample])
                # thr_width = thr_width_accumulate / (line_sample - check_area_thr_width)

                # 초기 선언
                if frame_index == start_frame:
                    pre_start_point = start_point
                    x = np.linspace(start_frame - 200, start_frame, 201)
                    y = np.ones(shape=(x.shape[0],)) * 0
                    y_lpf = np.ones(shape=(x.shape[0],)) * 0
                    thr = np.ones(shape=(x.shape[0],)) * thr_width
                    pixel_level = np.ones(shape=(x.shape[0],)) * 0

                # 홈에서 멈춤
                print("start_point : {}, pre_start_point : {}".format(start_point, pre_start_point))
                if abs(pre_start_point[0] - start_point[0]) > 10:
                    isHole = True
                if frame_index % 2 == 0:
                    pre_start_point = start_point

                tmp = [i for i in line_level[check_area_thr[0]:check_area_thr[1]] if i > thr_width + 10]
                defect_pixelWidth = len(tmp)
                #TODO : defect_pixelWidth가 n개 이상일 때 불량
                x = np.append(x, frame_index)[1:]
                thr = np.append(thr, thr_width + 10)[1:]
                if defect_pixelWidth < 1:
                    y = np.append(y, 0)[1:]
                    # y_lpf = np.append(y_lpf, 0)[1:]
                else:
                    y = np.append(y, sum(tmp) / defect_pixelWidth)[1:]
                    # y_lpf = np.append(y_lpf, sum(y[-numOfMean:]) / numOfMean)[1:]

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
                # ax1.set_xlim([0, 255])
                # ax1.set_ylim([0, line_total])
                # ax1.hist(line_level)
                #
                # ax2.grid(True)
                # ax2.set_xlabel('pixel position')
                # ax2.set_ylabel('pixel level')
                # ax2.set_title('defect_width : {0}pixels (threshold : {1:.2f})'.format(defect_pixelWidth, thr_width))
                # ax2.plot([(start_point[0]+70+ i) for i in range(line_total)], line_level, linewidth = 4)
                # # ax2.plot([i+70 for i in range(line_check)], extract_line_BR)
                # # ax2.plot([i+70 for i in range(line_check)], extract_line_B)
                # # ax2.plot([i+70 for i in range(line_check)], extract_line_R)
                # ax2.vlines(start_point[0] + 70+check_area_thr[0], 100, 180, linestyle=':', color="red")
                # ax2.vlines(start_point[0] + 70+check_area_thr[1], 100, 180, linestyle=':', color="red")
                # ax2.plot([start_point[0], start_point[0] + line_total], [thr_width, thr_width], linestyle='--', color="green")
                #
                # tmp = [i for i in line_level[check_area_thr[0]:check_area_thr[1]] if i > thr_width]
                # if len(tmp) >= 1:
                #     pixel_level = np.append(pixel_level, sum(tmp)/len(tmp))[1:]
                # ax3.set_xlabel('frame')
                # ax3.set_ylabel('Defect Score')
                # ax4.set_ylabel('plxel level', color='red')
                # ax3.set_ylim([0, 255])
                # # ax4.set_ylim([0, 255])
                # ax3.set_facecolor("whitesmoke")  # "https://ehclub.net/674"
                # ax3.grid(True)
                # if y[-1] > thr[-1] and defect_pixelWidth >= 3:
                #     ax3.set_facecolor("indianred")
                # ax3.set_title('Defect Score : {0:.2f} (threshold : {1:.2f})'.format(y[-1], thr[-1]))
                # ax3.plot(x, y, label="Defect score")
                # # ax3.plot(x, y_lpf, label="LPF")
                # ax3.plot(x, thr, label="threshold", linestyle='--')
                # # ax4.plot(x, pixel_level, 'r-', label="pixel level", linewidth = 1)
                # ax3.set_ylim(bottom=0)
                # ax3.legend(loc='upper left')

                # image save
                x1 = start_point[0] - 40 + 70
                y1 = start_point[1] - 240
                x2 = start_point[0] + 150 + 70
                y2 = start_point[1] + 470
                count_save += 1
                save_img.append(gray)
                if frame_index - start_frame >= 1:
                    pre_gray = save_img.pop(0)
                    if isHole:
                        start_frame = frame_index + 5
                        isHole = False
                        # try:
                        #     frame = cv2.rectangle(frame, (start_point[0] + 70, start_point[1] - int(numOfLine/2)),
                        #                           (start_point[0] + 70 + 50, start_point[1] + 270), color, thickness)
                        #     gray = cv2.rectangle(gray, (start_point[0] + 70, start_point[1] - int(numOfLine)),
                        #                          (start_point[0] + 70 + 50, start_point[1] + 270), color, thickness)
                        #     cv2.imwrite(
                        #         PATH + r"\{0}\760\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        #         .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                        #         cv2.resize(frame[y1:y2, x1:x2], (int(760), int(710))))
                        #     cv2.imwrite(
                        #         PATH + r"\{0}\380\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        #         .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                        #         cv2.resize(frame[y1:y2, x1:x2], (int(380), int(710))))
                        #     cv2.imwrite(
                        #         PATH + r"\{0}\760\gray\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        #         .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                        #         cv2.resize(gray[y1:y2, x1:x2], (int(760), int(710))))
                        #     cv2.imwrite(
                        #         PATH + r"\{0}\380\gray\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        #         .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                        #         cv2.resize(gray[y1:y2, x1:x2], (int(380), int(710))))
                        # except:
                        #     cv2.imwrite(PATH + r"\error\{0}_{1:.2f}_{2:.2f}_({3},{4})({5},{6}).png"
                        #                 .format(frame_index, y[-1], thr_width, x1, y1, x2, y2), frame)

                    elif y[-1] > thr[-1] and defect_pixelWidth >= 3 and count_save >= 12:
                        print("===========================================")
                        # cv2.imwrite(PATH + r"\{0}\gray\{1}_{2:.2f}_{3}.png".format(file, frame_index, y[-1], defect_pixelWidth), pre_gray)
                        # cv2.imwrite(PATH + r"\{0}\rgb\{1}_{2:.2f}_{3}.png".format(file, frame_index, y[-1], defect_pixelWidth), frame)
                        try:
                            # frame = cv2.rectangle(frame, (start_point[0] + check_area_thr[0] + 70, start_point[1] - 15),
                            #                       (start_point[0] + 70 + check_area_thr[0] + defect_pixelWidth + 5, start_point[1] + 15), color, thickness)
                            # gray = cv2.rectangle(gray, (start_point[0] + check_area_thr[0] + 70, start_point[1] - 15),
                            #                      (start_point[0] + 70 + check_area_thr[0] + defect_pixelWidth + 5, start_point[1] + 15), color, thickness)

                            # cv2.imwrite(
                            #     PATH + r"\{0}\760\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                            #     .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                            #     cv2.resize(frame[y1:y2, x1:x2], (int(760), int(710))))
                            # cv2.imwrite(
                            #     PATH + r"\{0}\380\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                            #     .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                            #     cv2.resize(frame[y1:y2, x1:x2], (int(380), int(710))))
                            # cv2.imwrite(
                            #     PATH + r"\{0}\760\gray\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                            #     .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                            #     cv2.resize(gray[y1:y2, x1:x2], (int(760), int(710))))
                            # cv2.imwrite(
                            #     PATH + r"\{0}\380\gray\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                            #     .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                            #     cv2.resize(gray[y1:y2, x1:x2], (int(380), int(710))))

                            cv2.imwrite(
                                PATH + r"\{0}\total\rgbTest_defect\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                                .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2), frame)
                            cv2.imwrite(
                                PATH + r"\{0}\380\rgbTest_defect\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                                .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                                cv2.resize(frame[y1:y2, x1:x2], (int(380), int(710))))

                            count_save = 0

                        except:
                            cv2.imwrite(PATH + r"\error\{0}_{1:.2f}_{2:.2f}_({3},{4})({5},{6}).png"
                                        .format(frame_index, y[-1], thr_width, x1, y1, x2, y2), frame)

                    elif y[-1] < thr[-1] and frame_index % 1 == 0 and defect_pixelWidth < 3:
                        cv2.imwrite(
                            PATH + r"\{0}\total\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                            .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2), frame)
                        cv2.imwrite(
                            PATH + r"\{0}\good_380\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                            .format(file, frame_index, y[-1], defect_pixelWidth, thr_width, x1, y1, x2, y2),
                            cv2.resize(frame[y1:y2, x1:x2], (int(380), int(710))))

                # frame = cv2.rectangle(frame, (upper_point[0], upper_point[1] - 10), (upper_point[0] + line_total, upper_point[1] + 10), color, thickness)
                # frame = cv2.rectangle(frame, (lower_point[0], lower_point[1] - 10), (lower_point[0] + line_total, lower_point[1] + 10), color, thickness)
                # cv2.imshow('frame', cv2.resize(frame[y1:y2, x1:x2], (int(760), int(710))))
                # cv2.imshow('frame1', cv2.resize(frame[y1:y2, x1:x2], (int(380), int(710))))
                # frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.imshow('gray', cv2.resize(frame, (int(frame.shape[1] / 4 * 3), int(frame.shape[0] / 4 * 3))))
                cv2.waitKey(1)
                frame_index += 1
            else:
                frame_index += 1
                continue

            # plt.draw()
            # plt.tight_layout()
            # plt.pause(0.001)
            # ax1.cla()
            # ax2.cla()
            # ax3.cla()
            # ax4.cla()
            toc = time.time()
            print("inference time : ", (toc - tic))
        else:
            break

    # Release the video file and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

def get_startPoint(gray, conv, start_point, line_check, thr, limit):
    check_level = [gray[start_point[1], start_point[0] + i] for i in range(line_check)]
    for i in range(len(check_level)):
        if check_level[i] > limit:
            check_level[i] = limit
    for i in range(line_check):
        # sum = 0
        # for j in conv:
        #     sum += check_level[j+i]
        # print(sum, end=' ')
        if check_level[i] > thr:
            point = (start_point[0] + i, start_point[1])
            break

    return point



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