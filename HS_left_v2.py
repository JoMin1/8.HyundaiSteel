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
    capture_frame = 1
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
    line_sample = 150
    line_backgroud = 0
    line_total = line_sample + line_backgroud

    files = []
    PATH = r"C:\Users\mjlee\Downloads\tmp"
    file_list = os.listdir(PATH)
    for file in file_list:  # 코드간결화 작업전
        if file.endswith(".bmp"):
            files.append(file)

    fig = plt.figure(None ,(15, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    # ax3 = fig.add_subplot(2, 2, (3, 4))
    plt.tight_layout()
    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()
        # start_point = (20, 0)
        start_point = (60, 400)
        print(frame_index)

        if ret:
            if frame_index % capture_frame == 0 and frame_index >= start_frame:
                for file in files:
                    # Save the frame as an image file
                    # cv2.imwrite("frame_{}.jpg".format(frame_index), frame)
                    # gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # print(gray1.shape)
                    tic = time.time()
                    blue = frame[:, :, 0].copy()
                    blue = blue.astype(np.int16)
                    green = frame[:, :, 1].copy()
                    green = green.astype(np.int16)
                    red = frame[:,:,2].copy()
                    red = red.astype(np.int16)
                    input = (3*blue - red - green).copy()
                    input = cv2.imread(r"C:\Users\mjlee\Downloads\tmp\135457_9451_5.bmp", cv2.IMREAD_GRAYSCALE)
                    # input = cv2.imread(r"C:\Users\mjlee\Downloads\tmp\BC (10).bmp", cv2.IMREAD_GRAYSCALE)
                    # input = cv2.imread(r"C:\Users\mjlee\Downloads\tmp\BC (10).bmp", cv2.IMREAD_GRAYSCALE)
                    # input = cv2.imread(r"C:\Users\mjlee\Downloads\tmp\clipboardImage_23_0222_142429_673.jpeg", cv2.IMREAD_GRAYSCALE)
                    # input = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                    # input = np.where(input < 100, 0, input)
                    input = input.astype(np.uint8)
                    # CLAHE 객체 생성
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                    # CLAHE 객체에 원본 이미지 입력하여 CLAHE가 적용된 이미지 생성
                    # input = clahe.apply(input)
                    ori = input[800:1550, 900:1650]
                    mask = np.zeros_like(ori)


                    # for i in range(input.shape[0]):
                    #     for j in range(input.shape[1]):
                    #         if 2 * blue[i][j] - red[i][j] < min_value:
                    #             input[i][j] = min_value
                    #         else:
                    #             input[i][j] = 2 * blue[i][j] - red[i][j]



                    # kernel_left = np.array([[-1,-1,0,0,0,1,1], [-1,-1,0,0,0,1,1], [-1,-1,0,0,0,1,1], [-1,-1,0,0,0,1,1], [-1,-1,0,0,0,1,1], [-1,-1,0,0,0,1,1]], dtype=np.float64) / 16.
                    # kernel_right = np.array([[1,1,0,0,0,-1,-1], [1,1,0,0,0,-1,-1], [1,1,0,0,0,-1,-1], [-1,-1,0,0,0,1,1], [-1,-1,0,0,0,1,1], [-1,-1,0,0,0,1,1]], dtype=np.float64) / 16.
                    # kernel_left = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]],dtype=np.float64) / 4.
                    # kernel_right = np.array([[1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1]],dtype=np.float64) / 4.
                    # kernel_left = np.array([[-1, -1, -1, -1, 0, 1, 1, 1, 1], [-1, -1, -1, -1, 0,1, 1, 1, 1], [-1, -1, -1, -1, 0,1, 1, 1, 1], [-1, -1, -1, -1, 0,1, 1, 1, 1]],
                    #                        dtype=np.float64) / 2.
                    # kernel_right = np.array([[1, 1, 1, 0, -1, -1, -1], [1, 1, 1, 0,-1, -1, -1], [1, 1, 1, 0,-1, -1, -1], [1, 1, 1, 0,-1, -1, -1], [1, 1, 1, 0,-1, -1, -1], [1, 1, 1, 0,-1, -1, -1], [1, 1, 1, 0,-1, -1, -1]],
                    #                         dtype=np.float64) / 16.
                    kernel_right = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0.1, 0, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                             [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                                            dtype=np.float64) / 25.
                    kernel_right1 = np.full((69, 9), fill_value=0.1, dtype=np.float64) / 36.

                    # dst_left = cv2.filter2D(input[0:int(input.shape[0]), 0:int(input.shape[1]/2)], -1, kernel_left)
                    # edges = cv2.filter2D(input[0:int(input.shape[0]), int(input.shape[1]/2):input.shape[1]], -1, kernel_right)

                    # input = np.where(ori <= 50, 255, mask)
                    # edge = cv2.filter2D(input, -1, kernel_right).copy()
                    # dst = np.where(edge > 70, 255, mask).copy()
                    #
                    # input = np.where(ori <= 50, 255, mask)
                    # edge1 = cv2.filter2D(input, -1, kernel_right1).copy()
                    # dst1 = np.where(edge1 > 70, 255, mask).copy()
                    #
                    # print("count white : ", len(dst[dst == 255]), len(dst1[dst1 == 255]))

                    kernel = np.ones((5,5), np.uint8)
                    result = cv2.morphologyEx(ori, cv2.MORPH_OPEN, kernel)
                    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                    cv2.imshow('result', cv2.resize(result, (int(result.shape[1]), int(result.shape[0]))))

                    # subplot
                    ax1.grid(True)
                    ax1.set_xlabel('pixel level')
                    ax1.set_ylabel('number')
                    ax1.set_title('Histogram')
                    ax1.set_xlim([0, 255])
                    ax1.hist(ori)

                    ax2.grid(True)
                    ax2.set_xlabel('pixel level')
                    ax2.set_ylabel('number')
                    ax2.set_title('Histogram')
                    ax2.set_xlim([0, 255])
                    ax2.hist(result)
                    #
                    # ax2.grid(True)
                    # ax2.set_xlabel('pixel position')
                    # ax2.set_ylabel('pixel level')
                    # ax2.set_ylim(ylim)
                    # ax2.set_title('defect_width : {0}pixels (threshold : {1:.2f})'.format(defect_width, thr_width))
                    # ax2.plot([(start_point[0] + i) for i in range(line_total)], extract_line)
                    # ax2.vlines(start_point[0] + check_area_thr[0], 0, 10000, linestyle=':', color="red")
                    # ax2.vlines(start_point[0] + check_area_thr[1], 0, 10000, linestyle=':', color="red")
                    # ax2.plot([start_point[0], start_point[0] + line_total], [thr_width, thr_width], linestyle='--', color="green")
                    #
                    #
                    # ax3.grid(True)
                    # ax3.set_xlabel('frame')
                    # ax3.set_ylabel('sum')
                    # ax3.set_facecolor("whitesmoke") # "https://ehclub.net/674"
                    # if y[-1] > thr:
                    #     ax3.set_facecolor("indianred")
                    # ax3.set_title('Defect Score : {0:.2f} (threshold : {1:.2f})'.format(y[-1], thr))
                    # ax3.plot(x, y, label="sum")
                    # ax3.plot(x, y_lpf, label="LPF")
                    # ax3.plot([x[0], x[-1]], [thr, thr], linestyle = '--', linewidth = 2)
                    # ax3.legend(loc='upper left')
                    #
                    # # image save
                    # count_save += 1
                    # save_img.append(gray)
                    # if frame_index - start_frame >= 12:
                    #     pre_gray = save_img.pop(0)
                    #     if thr < y[-1] and count_save >= 12:
                    #         print("===========================================")
                    #         cv2.imwrite(PATH + r"\{0}\gray\{1}_{2:.2f}_{3}.png".format(file, frame_index, sum(y[-numOfMean:] / numOfMean), defect_width), pre_gray)
                    #         cv2.imwrite(PATH + r"\{0}\rgb\{1}_{2:.2f}_{3}.png".format(file, frame_index, sum(y[-numOfMean:] / numOfMean), defect_width), frame)
                    #         cv2.imshow('gray', cv2.resize(gray, (int(gray.shape[1]), int(gray.shape[0]))))
                    #         cv2.imshow('pre_gray', cv2.resize(pre_gray, (int(gray.shape[1]), int(gray.shape[0]))))
                    #         cv2.waitKey(0)
                    #         count_save = 0

                    # dst = cv2.rectangle(dst, start_point, (start_point[0] + line_total, start_point[1] + numOfLine), color, thickness)
                    # cv2.imshow('edges', cv2.resize(edges, (int(edges.shape[1]), int(edges.shape[0]))))
                    plt.draw()
                    plt.tight_layout()
                    plt.pause(0.001)
                    # ax1.clear()
                    # ax2.clear()
                    # ax3.clear()

                    # cv2.imshow('input', cv2.resize(input, (int(input.shape[1]), int(input.shape[0]))))
                    # cv2.imshow('edge', cv2.resize(edge, (int(edge.shape[1]), int(edge.shape[0]))))
                    # cv2.imshow('edge1', cv2.resize(edge1, (int(edge.shape[1]), int(edge.shape[0]))))
                    # cv2.imshow('dst', cv2.resize(dst, (int(dst.shape[1]), int(dst.shape[0]))))
                    # cv2.imshow('dst1', cv2.resize(dst1, (int(dst1.shape[1]), int(dst1.shape[0]))))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    frame_index += 1
                    # clear()

            else:
                frame_index += 1
                continue



            toc = time.time()
            print("inference time : ", (toc - tic))

        else:
            break


    # Release the video file and destroy all windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

# input = input - edges
                    # edges = cv2.Canny(input, 0, 200)
                    # edges = cv2.Sobel(input, -1, 0, 1, delta=128)

                    # lines = cv2.HoughLines(input, rho=1, theta=np.pi / 180., threshold=50)
                    # print(lines.shape)
                    # dst = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
                    # # Draw the lines on the original image
                    # if lines is not None:
                    #     for line in lines:
                    #         rho, theta = line[0]
                    #         a = np.cos(theta)
                    #         b = np.sin(theta)
                    #         x0 = a * rho
                    #         y0 = b * rho
                    #         x1 = int(x0 + 1000 * (-b))
                    #         y1 = int(y0 + 1000 * (a))
                    #         x2 = int(x0 - 1000 * (-b))
                    #         y2 = int(y0 - 1000 * (a))
                    #         cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    # toc = time.time()
                    # print("time : ", toc - tic)
