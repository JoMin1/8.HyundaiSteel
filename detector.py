import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from glob import glob

class Detector():
    def __init__(self, start_frame, cur_frame):
        # Set the frame index
        self.cur_frame = cur_frame
        self.start_frame = start_frame
        self.pre_start_point = 0

        # check area
        self.check_length = 700
        self.numOfLine = 200
        self.line_sample = 130
        self.check_area_defectScore = (4, 80)
        self.check_area_thr = 50

        # threshold
        self.edge_threshold = 100

        # others
        self.isHole = False
        self.initial = True
        self.margine = 6
        self.ignore_edge = 3

        # plot
        self.color = (200, 200, 200)
        self.thickness = 2

    def detect(self, frame, gray, initial_start_point, isLeft):
        # 배경과 강판의 경계 위치 추출 (3Blue - 1Red - 1Green)
        start_point, target_line = self.get_start_point(frame, gray, initial_start_point, isLeft=isLeft)

        # 경계 위치에부터 line_sample만큼 grayscale level(target_level) 추출
        # Adaptive threshold : line_sample의 마지막 check_area_thr만큼을 사용해 threshold 획득
        if isLeft:
            self.target_level = target_line[start_point[0]:  start_point[0] + self.line_sample]
            self.cur_thr = np.mean(self.target_level[self.line_sample - self.check_area_thr: self.line_sample])
            DefectPixel = [i for i in self.target_level[self.check_area_defectScore[0]: self.check_area_defectScore[1]] if
                           i > self.cur_thr + self.margine]
        else:
            self.target_level = target_line[- start_point[0] - self.line_sample: -start_point[0]]
            self.cur_thr = np.mean(self.target_level[0: self.check_area_thr])
            DefectPixel = [i for i in self.target_level[-self.check_area_defectScore[1] - 1: -self.check_area_defectScore[0] - 1] if
                           i > self.cur_thr + self.margine]

        # 초기 선언
        if self.initial:
            self.pre_start_point = start_point
            self.x = np.linspace(self.start_frame - 200, self.start_frame, 201)
            self.y = np.ones(shape=(self.x.shape[0],)) * 0
            self.thr = np.ones(shape=(self.x.shape[0],)) * self.cur_thr
            self.initial = False

        # 용접 추출, 2프레임 전 경계와 10pixel 이상 차이날 경우 용접으로 판단
        if abs(self.pre_start_point[0] - start_point[0]) > 10:
            self.isHole = True
        if self.cur_frame % 2 == 0:
            self.pre_start_point = start_point

        numOfDefectPixel = len(DefectPixel)  # defect 픽셀 개수
        self.x = np.append(self.x, self.cur_frame)[1:]
        self.thr = np.append(self.thr, self.cur_thr + self.margine)[1:]  # adaptive threshold
        if numOfDefectPixel == 0:
            self.y = np.append(self.y, 0)[1:]
        else:
            self.y = np.append(self.y, sum(DefectPixel) / numOfDefectPixel)[1:]

        self.initial_start_point = initial_start_point
        self.cur_frame += 1

        return start_point, numOfDefectPixel


    def get_start_point(self, frame, gray, start_point, isLeft=True):
        # 높이 폭 numOfLine과 너비 폭 check_length 영역 crop
        if isLeft:
            self.line_B = np.sum(frame[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                            start_point[0]: start_point[0] + self.check_length, 0] / self.numOfLine, axis=0)
            self.line_G = np.sum(frame[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                            start_point[0]: start_point[0] + self.check_length, 1] / self.numOfLine, axis=0)
            self.line_R = np.sum(frame[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                            start_point[0]: start_point[0] + self.check_length, 2] / self.numOfLine, axis=0)
            line_gray = np.sum(
                gray[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                start_point[0]: start_point[0] + self.check_length] / self.numOfLine, axis=0)
            self.line_BGR = list(3 * self.line_B - self.line_G - self.line_R)
            for i in range(self.check_length):
                # edge_threshold 이상인 위치에서 배경과 강판의 경계
                if self.line_BGR[i] > self.edge_threshold:
                    start_point = (i, start_point[1])
                    break

        # 높이 폭 numOfLine과 너비 폭 check_length 영역 crop
        else:
            self.line_B = np.sum(frame[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                            start_point[0] - self.check_length: start_point[0], 0] / self.numOfLine, axis=0)
            self.line_G = np.sum(frame[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                            start_point[0] - self.check_length: start_point[0], 1] / self.numOfLine, axis=0)
            self.line_R = np.sum(frame[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                            start_point[0] - self.check_length: start_point[0], 2] / self.numOfLine, axis=0)
            line_gray = np.sum(
                gray[start_point[1] - int(self.numOfLine / 2): start_point[1] + int(self.numOfLine / 2),
                start_point[0] - self.check_length: start_point[0]] / self.numOfLine, axis=0)
            self.line_BGR = list(3 * self.line_B - self.line_G - self.line_R)
            for i in range(self.check_length):
                # edge_threshold 이상인 위치에서 배경과 강판의 경계
                if self.line_BGR[self.check_length - i - 1] > self.edge_threshold:
                    start_point = (i, start_point[1])
                    break

        return start_point, line_gray

    def visualize(self, start_point, numOfDefectPixel, ax1, ax2, ax3, isLeft=True):
        # self.ax1.grid(True)
        # self.ax1.set_xlabel('pixel level')
        # self.ax1.set_ylabel('number')
        # self.ax1.set_title('Histogram')
        # self.ax1.set_xlim([0, 255])
        # self.ax1.set_ylim([0, self.line_sample])
        # self.ax1.hist(self.target_level)

        if isLeft:
            ax1.grid(True)
            ax1.set_xlabel('pixel position')
            ax1.set_ylabel('pixel level')
            ax1.set_title('defect_width : {0}pixels (threshold : {1:.2f})'.format(numOfDefectPixel, self.cur_thr))
            ax1.plot([(start_point[0] + self.initial_start_point[0] + i) for i in range(self.line_sample)], self.target_level, linewidth = 4)
            ax1.plot([self.initial_start_point[0] + i for i in range(self.check_length)], self.line_BGR, linewidth=1, color="black")
            ax1.plot([self.initial_start_point[0] + i for i in range(self.check_length)], self.line_B, linewidth=1, color="blue")
            ax1.plot([self.initial_start_point[0] + i for i in range(self.check_length)], self.line_G, linewidth=1, color="green")
            ax1.plot([self.initial_start_point[0] + i for i in range(self.check_length)], self.line_R, linewidth=1, color="red")
            ax1.vlines(self.initial_start_point[0] + start_point[0] + self.check_area_defectScore[0], 0, 200, linestyle=':', color="red")
            ax1.vlines(self.initial_start_point[0] + start_point[0] + self.check_area_defectScore[1], 0, 200, linestyle=':', color="red")
            ax1.plot([self.initial_start_point[0] + start_point[0], self.initial_start_point[0] + start_point[0] + self.line_sample], [self.cur_thr, self.cur_thr], linestyle='--', color="green")

        else:
            ax1.grid(True)
            ax1.set_xlabel('pixel position')
            ax1.set_ylabel('pixel level')
            ax1.set_title('defect_width : {0}pixels (threshold : {1:.2f})'.format(numOfDefectPixel, self.cur_thr))
            ax1.plot([(self.initial_start_point[0] - start_point[0] - self.line_sample + i) for i in range(self.line_sample)], self.target_level, linewidth=4)
            ax1.plot([self.initial_start_point[0] - self.check_length + i for i in range(self.check_length)], self.line_BGR, linewidth=1, color="black")
            ax1.plot([self.initial_start_point[0] - self.check_length + i for i in range(self.check_length)], self.line_B, linewidth=1, color="blue")
            ax1.plot([self.initial_start_point[0] - self.check_length + i for i in range(self.check_length)], self.line_G, linewidth=1, color="green")
            ax1.plot([self.initial_start_point[0] - self.check_length + i for i in range(self.check_length)], self.line_R, linewidth=1, color="red")
            ax1.vlines(self.initial_start_point[0] - start_point[0] - self.check_area_defectScore[0], 0, 200, linestyle=':', color="red")
            ax1.vlines(self.initial_start_point[0] - start_point[0] - self.check_area_defectScore[1], 0, 200, linestyle=':', color="red")
            ax1.plot([self.initial_start_point[0] - start_point[0] - self.line_sample, self.initial_start_point[0] -  start_point[0]], [self.cur_thr, self.cur_thr], linestyle='--', color="green")

        ax2.set_xlabel('frame')
        ax2.set_ylabel('Defect Score')
        ax3.set_ylabel('plxel level', color='red')
        ax2.set_ylim([0, 255])
        ax2.set_facecolor("whitesmoke")  # "https://ehclub.net/674"
        ax2.grid(True)
        if numOfDefectPixel > self.ignore_edge:
            ax2.set_facecolor("indianred")
        ax2.set_title('Defect Score : {0:.2f} (threshold : {1:.2f})'.format(self.y[-1], self.thr[-1]))
        ax2.plot(self.x, self.y, label="Defect score")
        ax2.plot(self.x, self.thr, label="threshold", linestyle='--')
        # ax3.set_ylim([0, 255])
        # ax3.plot(x, pixel_level, 'r-', label="pixel level", linewidth = 1)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc='upper left')

        return ax1, ax2, ax3

    def save(self, PATH, file, frame, gray, start_point, numOfDefectPixel, color=(0, 0, 200), thickness = 3, isLeft=True):
        if self.cur_frame - self.start_frame >= 0:
            try:
                cv2.imwrite(
                    PATH + r"\{0}\380_total\rgb\{1}_{2:.2f}_{3}_{4:.2f}.png"
                    .format(file, self.cur_frame, self.y[-1], numOfDefectPixel, self.cur_thr), frame)
                cv2.imwrite(
                    PATH + r"\{0}\380_total\gray\{1}_{2:.2f}_{3}_{4:.2f}.png"
                    .format(file, self.cur_frame, self.y[-1], numOfDefectPixel, self.cur_thr), gray)
                if isLeft:
                    # 저장할 영상의 crop 영역
                    x1 = start_point[0] - 40 + self.initial_start_point[0]
                    y1 = start_point[1] - 240
                    x2 = start_point[0] + 150 + self.initial_start_point[0]
                    y2 = start_point[1] + 470
                    print(start_point, x1, y1, x2, y2)
                    cv2.imwrite(
                        PATH + r"\{0}\380_left\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        .format(file, self.cur_frame, self.y[-1], numOfDefectPixel, self.cur_thr, x1, y1, x2, y2),
                        cv2.resize(frame[y1:y2, x1:x2], (int(380), int(710))))
                    cv2.imwrite(
                        PATH + r"\{0}\380_left\gray\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        .format(file, self.cur_frame, self.y[-1], numOfDefectPixel, self.cur_thr, x1, y1, x2, y2),
                        cv2.resize(gray[y1:y2, x1:x2], (int(380), int(710))))
                else:
                    # 저장할 영상의 crop 영역
                    x1 = self.initial_start_point[0] - start_point[0] - 150
                    y1 = start_point[1] - 240
                    x2 = self.initial_start_point[0] - start_point[0] + 40
                    y2 = start_point[1] + 470
                    print(start_point, x1, y1, x2, y2, frame.shape)
                    cv2.imwrite(
                        PATH + r"\{0}\380_right\rgb\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        .format(file, self.cur_frame, self.y[-1], numOfDefectPixel, self.cur_thr, x1, y1, x2, y2),
                        cv2.resize(frame[y1:y2, x1:x2], (int(380), int(710))))
                    cv2.imwrite(
                        PATH + r"\{0}\380_right\gray\{1}_{2:.2f}_{3}_{4:.2f}_({5},{6})({7},{8}).png"
                        .format(file, self.cur_frame, self.y[-1], numOfDefectPixel, self.cur_thr, x1, y1, x2, y2),
                        cv2.resize(gray[y1:y2, x1:x2], (int(380), int(710))))
            except:
                cv2.imwrite(PATH + r"\error\{0}_{1:.2f}_{2:.2f}_({3},{4})({5},{6}).png"
                            .format(self.cur_frame, self.y[-1], self.cur_thr, x1, y1, x2, y2), frame)

            # 용접 검출 후 5프레임 동안 sleep
            if self.isHole:
                self.start_frame = self.cur_frame + 5
                self.isHole = False
            # 불량 검출 후 5프레임 동안 sleep
            else:
                self.start_frame = self.cur_frame + 5
