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

    # draw line
    color = (200, 200, 200)
    thickness = 2
    line = 700
    line_sample = 130
    line_backgroud = 0

    # threshold

    # others
    numOfLine = 400
    margin = 30

    # fig0 = plt.figure(0 ,(15, 8))
    # ax1 = fig0.add_subplot(3, 1, 1)
    # ax2 = fig0.add_subplot(3, 1, 2)
    # ax3 = fig0.add_subplot(3, 1, 3)
    # fig1 = plt.figure(1, (15, 8))
    # ax1 = fig1.add_subplot(3, 1, 1)
    # ax2 = fig1.add_subplot(3, 1, 2)
    # ax3 = fig1.add_subplot(3, 1, 3)
    # plt.tight_layout()



    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()
        print(frame_index)

        if ret:
            if frame_index % capture_frame == 0 and frame_index >= start_frame:
                fig0 = plt.figure(0, (15, 8))
                plt.tight_layout()
                fig1 = plt.figure(1, (15, 8))
                plt.tight_layout()

                # 두알
                # frame = cv2.imread(r"C:\Users\mjlee\Downloads\tmp\135457_9451_5.bmp")
                # cv2.imshow("frame", frame)
                # cv2.waitKey(0)

                ax0 = fig0.add_subplot(4, 1, 1)
                ax1 = fig0.add_subplot(4, 1, 2)
                ax2 = fig0.add_subplot(4, 1, 3)
                ax3 = fig0.add_subplot(4, 1, 4)

                start_point = (margin, 15)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax0.set_ylim([0, 255])
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="y")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)

                start_point = (margin, 415)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax1.set_ylim([0, 255])
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="y")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)

                start_point = (margin, 800)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax2.set_ylim([0, 255])
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="y")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)

                start_point = (margin, 1000)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax3.set_ylim([0, 255])
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="y")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)


                ax0 = fig1.add_subplot(4, 1, 1)
                ax1 = fig1.add_subplot(4, 1, 2)
                ax2 = fig1.add_subplot(4, 1, 3)
                ax3 = fig1.add_subplot(4, 1, 4)

                start_point = (1920-margin-line, 15)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax0.set_ylim([0, 255])
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax0.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="black")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)

                start_point = (1920-margin-line, 415)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax1.set_ylim([0, 255])
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax1.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="black")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)

                # print(extract_line_B[130:line-170])
                # for B in extract_line_B[130:line-170]:
                #     if B > 120:
                #         print("defect")
                #         print(B)
                #         cv2.imshow('gray', cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0]))))
                #         cv2.waitKey(0)

                start_point = (1920-margin-line, 800)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax2.set_ylim([0, 255])
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax2.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="black")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)

                start_point = (1920-margin-line, 1000)
                extract_line_B = [frame[start_point[1], start_point[0] + i][0] for i in range(line)]
                extract_line_G = [frame[start_point[1], start_point[0] + i][1] for i in range(line)]
                extract_line_R = [frame[start_point[1], start_point[0] + i][2] for i in range(line)]
                ax3.set_ylim([0, 255])
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_B))], extract_line_B, color="blue")
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_G))], extract_line_G, color="green")
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_R))], extract_line_R, color="red")
                BGR = [3 * extract_line_B[i] - extract_line_R[i] - extract_line_G[i] for i in range(line)]
                extract_line_BGR = [BGR[i] for i in range(line)]
                ax3.plot([(start_point[0] + i) for i in range(len(extract_line_BGR))], extract_line_BGR, color="black")
                gray = cv2.rectangle(frame, (start_point[0], start_point[1] - 10),
                                     (start_point[0] + line, start_point[1] + 10), color,
                                     thickness)

                cv2.imshow('gray', cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))))
                frame_index += 1

            else:
                frame_index += 1
                continue

            plt.draw()
            plt.tight_layout()
            plt.pause(0.001)
            cv2.waitKey(0)
            # plt.clf()
            ax1.clear()
            ax2.clear()
            ax3.clear()

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