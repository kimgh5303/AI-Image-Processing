import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from tkinter import messagebox
from tkinter import Tk
from PIL import Image as im
root= Tk()
root.withdraw()
pre_picture_path = []
add_picture_path = []
def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

form1 = resource_path('Main.ui')
form1, base1 = uic.loadUiType(form1)
W=0.5
class MainWindow(base1, form1):
    
    # 버튼 이벤트 처리
    def __init__(self):
        try:
            super(base1, self).__init__()
            self.setupUi(self)
            self.pushButton_filepath.clicked.connect(self.fileInsert)
            self.pushButton_filepath2.clicked.connect(self.fileInsert_add)
            self.pushButton_Plus.clicked.connect(self.Plus)
            self.pushButton_Minus.clicked.connect(self.Minus)
            self.pushButton_Multi.clicked.connect(self.Multi)
            self.pushButton_Div.clicked.connect(self.Div)
            self.pushButton_AND.clicked.connect(self.AND)
            self.pushButton_OR.clicked.connect(self.OR)
            self.pushButton_Histogram.clicked.connect(self.Histogram)
            self.pushButton_Thresholding.clicked.connect(self.Thresholding)
            self.pushButton_GlobalThresholding.clicked.connect(self.GlobalThresholding)
            self.pushButton_AdaptiveThresholding.clicked.connect(self.AdaptiveThresholding)
            self.pushButton_HistogramStretching.clicked.connect(self.HistogramStretching)
            self.pushButton_HistogramEqualization.clicked.connect(self.HistogramEqualization)
            self.pushButton_MeanFiltering.clicked.connect(self.MeanFiltering)
            self.pushButton_MedianFiltering.clicked.connect(self.MedianFiltering)
            self.pushButton_GaussianFiltering.clicked.connect(self.GaussianFiltering)
            self.pushButton_ConservativeSmoothing.clicked.connect(self.ConservativeSmoothing)
            self.pushButton_UnsharpFiltering_edge.clicked.connect(self.UnsharpFiltering_edge)
            self.pushButton_UnsharpFiltering.clicked.connect(self.UnsharpFiltering)
            self.pushButton_RobertCrossEdgeDetector.clicked.connect(self.RobertCrossEdgeDetector)
            self.pushButton_SobelEdgeDetector.clicked.connect(self.SobelEdgeDetector)
            self.pushButton_PrewittEdgeDetector.clicked.connect(self.PrewittEdgeDetector)
            self.pushButton_CannyEdgeDetector.clicked.connect(self.CannyEdgeDetector)
            self.pushButton_LaplacianEdgeDetector.clicked.connect(self.LaplacianEdgeDetector)
            self.pushButton_GaussianLaplacianEdgeDetector.clicked.connect(self.GaussianLaplacianEdgeDetector)
            self.pushButton_NearestNeighborInterpoliation.clicked.connect(self.NearestNeighborInterpoliation)
            self.pushButton_LinearNeighborInterpoliation.clicked.connect(self.LinearNeighborInterpoliation)
            self.pushButton_Rotation.clicked.connect(self.Rotation)
            self.pushButton_Flipping.clicked.connect(self.Flipping)
            self.pushButton_Translation.clicked.connect(self.Translation)
            self.pushButton_Affine.clicked.connect(self.Affine)
            self.pushButton_Dilation.clicked.connect(self.Dilation)
            self.pushButton_Erosion.clicked.connect(self.Erosion)
            self.pushButton_Opening.clicked.connect(self.Opening)
            self.pushButton_Closing.clicked.connect(self.Closing)
            self.pushButton_fileClear.clicked.connect(self.fileClear)
            self.pushButton_fileClear2.clicked.connect(self.fileClear2)
            self.pushButton_exit.clicked.connect(self.logout)           
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 이미지 저장과 출력
    def filesave(self,output_img):
        try:
            global after_picture_path
            data = im.fromarray(output_img)
            newfilename=pre_picture_path[0].split('.')
            newfilename[1]
            data.save(newfilename[0]+"_change."+newfilename[1])
            self.picture_after.setPixmap(QPixmap(newfilename[0]+"_change."+newfilename[1]))
            after_picture_path=newfilename[0]+"_change."+newfilename[1]
            print(after_picture_path)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)   

    # plt 이미지 출력
    def filePrint(self):
        self.picture_after.setPixmap(QPixmap('plt.png'))

    # 이미지 2개 삽입
    def fileInsert(self): # 파일 경로 지정하는 부분(파일 탐색기)
        try:
            global pre_picture_path
            pre_picture_path=QFileDialog.getOpenFileName(self,'','C:/Users/kimgh/Desktop/image','All File(*)')    
            print(type(pre_picture_path))
            self.picture_pre.setPixmap(QPixmap(pre_picture_path[0]))
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
    def fileInsert_add(self): # 파일 경로 지정하는 부분(파일 탐색기)
        try:
            global add_picture_path
            add_picture_path=QFileDialog.getOpenFileName(self,'','C:/Users/kimgh/Desktop/image','All File(*)')      
            self.picture_add.setPixmap(QPixmap(add_picture_path[0]))
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 출력 이미지 제거
    def fileClear(self):
        try:
            global pre_picture_path
            self.picture_pre.clear()
            pre_picture_path=list(pre_picture_path)
            pre_picture_path.clear()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
    def fileClear2(self):
        try:
            global add_picture_path
            self.picture_add.clear()
            add_picture_path=list(add_picture_path)
            add_picture_path.clear()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
#-----------------------------------------------------------
    # 3장
    # 픽셀 더하기
    def Plus(self):
        try:
            global pre_picture_path
            global add_picture_path
            img1 = cv.imread(pre_picture_path[0])
            img2 = cv.imread(add_picture_path[0])
            output_img=cv.add(img1,img2)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 픽셀 빼기
    def Minus(self):
        try:
            global pre_picture_path
            global add_picture_path
            img1 = cv.imread(pre_picture_path[0])
            img2 = cv.imread(add_picture_path[0])
            output_img = cv.cvtColor(img1,cv.COLOR_BGR2RGB) 
            # BGR채널순서를 RGB채널로 변경
            RGB_img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
            RGB_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB) 
            # RGB 채널 나누기
            R_img1,G_img1,B_img1=cv.split(RGB_img1)
            R_img2,G_img2,B_img2=cv.split(RGB_img2)

            # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
            R_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            G_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            B_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)

            #for문을 돌며 픽셀 빼기 연산 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    R_plus[h,w] = np.abs(np.int32(R_img1[h,w]) - np.int32(R_img2[h,w])) 
                    G_plus[h,w] = np.abs(np.int32(G_img1[h,w]) - np.int32(G_img2[h,w])) 
                    B_plus[h,w] = np.abs(np.int32(B_img1[h,w]) - np.int32(B_img2[h,w]))
            output_img[:,:,0]=R_plus
            output_img[:,:,1]=G_plus
            output_img[:,:,2]=B_plus
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 픽셀 곱하기
    def Multi(self):
        try:
            global pre_picture_path
            global add_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            RGB_img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
            output_img = cv.cvtColor(img1,cv.COLOR_BGR2RGB) 
            # RGB 채널 나누기
            R_img1,G_img1,B_img1=cv.split(RGB_img1)

            # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
            R_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            G_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            B_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            C=2.0  #곱하기 상수값 → 클수록 명암비가 커짐
            def saturation(value): #saturation함수로 정의하기 → 255가 넘으면 255로 채워줌
                if(value>255):
                    value = 255
                return value 
            #for문을 돌며 픽셀 곱하기 연산 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    R_plus[h,w] = saturation(np.int32(R_img1[h,w])*C) 
                    G_plus[h,w] = saturation(np.int32(G_img1[h,w])*C) 
                    B_plus[h,w] = saturation(np.int32(B_img1[h,w])*C) 
            output_img[:,:,0]=R_plus
            output_img[:,:,1]=G_plus
            output_img[:,:,2]=B_plus
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
    
    # 픽셀 나누기
    def Div(self):
        try:
            global pre_picture_path
            global add_picture_path
            img1 = cv.imread(pre_picture_path[0])
            img2 = cv.imread(add_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            RGB_img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
            RGB_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            output_img = cv.cvtColor(img1,cv.COLOR_BGR2RGB) 
            # RGB 채널 나누기
            R_img1,G_img1,B_img1=cv.split(RGB_img1)
            R_img2,G_img2,B_img2=cv.split(RGB_img2)

            # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
            R_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            G_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            B_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            def saturation(value): #saturation함수로 정의하기
                if(value>255):
                    value = 255
                return value 
            #for문을 돌며 픽셀 나누기 연산 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    R_plus[h,w] = saturation(np.fabs(np.float32(R_img1[h,w])/ np.float32(R_img2[h,w]+1))*255.0) 
                    G_plus[h,w] = saturation(np.fabs(np.float32(G_img1[h,w])/ np.float32(G_img2[h,w]+1))*255.0) 
                    B_plus[h,w] = saturation(np.fabs(np.float32(B_img1[h,w])/ np.float32(B_img2[h,w]+1))*255.0) 
            output_img[:,:,0]=R_plus
            output_img[:,:,1]=G_plus
            output_img[:,:,2]=B_plus
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
    
    # 픽셀 AND
    def AND(self):
        try:
            global pre_picture_path
            global add_picture_path
            img1 = cv.imread(pre_picture_path[0])
            img2 = cv.imread(add_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            RGB_img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
            RGB_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            output_img = cv.cvtColor(img1,cv.COLOR_BGR2RGB) 
            # RGB 채널 나누기
            R_img1,G_img1,B_img1=cv.split(RGB_img1)
            R_img2,G_img2,B_img2=cv.split(RGB_img2)
            # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
            R_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            G_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            B_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)

            def saturation(value): #saturation함수로 정의하기
                if(value>255):
                    value = 255
                return value 

            #영상 이진화 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    if(np.int32(R_img1[h,w])>180):
                        R_img1[h,w]=G_img1[h,w]=B_img1[h,w]=255
                    else:
                        R_img1[h,w]=G_img1[h,w]=B_img1[h,w]=0
                    if(np.int32(G_img2[h,w])>50):
                        R_img2[h,w]=G_img2[h,w]=B_img2[h,w]=255
                    else:
                        R_img2[h,w]=G_img2[h,w]=B_img2[h,w]=0 

            #for문을 돌며 픽셀 비트 AND 연산 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    R_plus[h,w] = saturation(np.int32(R_img1[h,w])& np.int32(R_img2[h,w])) 
                    G_plus[h,w] = saturation(np.int32(G_img1[h,w])& np.int32(G_img2[h,w])) 
                    B_plus[h,w] = saturation(np.int32(B_img1[h,w])& np.int32(B_img2[h,w]))
            #영상 다시 넣어주기  
            RGB_img1[:,:,0] = R_img1
            RGB_img1[:,:,1] = G_img1
            RGB_img1[:,:,2] = B_img1  
            RGB_img2[:,:,0] = R_img2
            RGB_img2[:,:,1] = G_img2
            RGB_img2[:,:,2] = B_img2
            output_img[:,:,0]=R_plus
            output_img[:,:,1]=G_plus
            output_img[:,:,2]=B_plus  
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 픽셀 OR
    def OR(self):
        try:
            global pre_picture_path
            global add_picture_path
            img1 = cv.imread(pre_picture_path[0])
            img2 = cv.imread(add_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            RGB_img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB) 
            RGB_img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            output_img = cv.cvtColor(img1,cv.COLOR_BGR2RGB) 
            # RGB 채널 나누기
            R_img1,G_img1,B_img1=cv.split(RGB_img1)
            R_img2,G_img2,B_img2=cv.split(RGB_img2)

            # 출력 array 생성하고 0으로 초기화, unsigned byte (0~255)로 설정
            R_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            G_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            B_plus=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]),dtype=np.ubyte)
            def saturation(value): #saturation함수로 정의하기
                if(value>255):
                    value = 255
                return value 

            #영상 이진화 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    if(np.int32(R_img1[h,w])>180):
                        R_img1[h,w]=G_img1[h,w]=B_img1[h,w]=255
                    else:
                        R_img1[h,w]=G_img1[h,w]=B_img1[h,w]=0
                    if(np.int32(G_img2[h,w])>50):
                        R_img2[h,w]=G_img2[h,w]=B_img2[h,w]=255
                    else:
                        R_img2[h,w]=G_img2[h,w]=B_img2[h,w]=0 

            #for문을 돌며 픽셀 비트 OR 연산 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    R_plus[h,w] = saturation(np.int32(R_img1[h,w])| np.int32(R_img2[h,w])) 
                    G_plus[h,w] = saturation(np.int32(G_img1[h,w])| np.int32(G_img2[h,w])) 
                    B_plus[h,w] = saturation(np.int32(B_img1[h,w])| np.int32(B_img2[h,w]))  
            output_img[:,:,0]=R_plus
            output_img[:,:,1]=G_plus
            output_img[:,:,2]=B_plus 
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
#-----------------------------------------------------------
#-----------------------------------------------------------
    # 4장
    # 히스토그램
    def Histogram(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            RGB_img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB) 
            # RGB 채널 나누기
            R_img1,G_img1,B_img1=cv.split(RGB_img1)
            # 히스토그램 출력, 채널은 0으로 표시
            hist = cv.calcHist([R_img1],[0],None,[256],[0,256])  # images / channel:gray / mask : 전체 영상 / histsize : 히스토그램 bin 개수 (영상 전체 색상) / ranges : 색상 범위
            plt.subplot(111)
            plt.plot(hist,color = 'r') # R색상 히스토그램 표현 → 각 색상의 약자를 따름
            plt.xlim([0,256])
            hist = cv.calcHist([G_img1],[0],None,[256],[0,256])
            plt.plot(hist,color = 'g')
            plt.xlim([0,256])
            hist = cv.calcHist([B_img1],[0],None,[256],[0,256])
            plt.plot(hist,color = 'b')
            plt.xlim([0,256])
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 임계값 적용
    def Thresholding(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            RGB_img1 = cv.cvtColor(img1,cv.COLOR_BGR2RGB)
            # 결과 영상을 담을 기억 장소 생성 
            output_img=np.zeros((RGB_img1.shape[0],RGB_img1.shape[1]))      # 0으로 채워진 array 생성    
            # 영상 임계값 적용 하기
            for h in range(RGB_img1.shape[0]):
                for w in range(RGB_img1.shape[1]):
                    if(np.int32(RGB_img1[h,w][0])<180):
                        output_img[h,w]=255
                    else:
                        output_img[h,w]=0
            #그림을 화면에 출력
            plt.subplot(111)
            plt.imshow(output_img, cmap='gray')
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 전역 임계값 적용
    def GlobalThresholding(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            gray_img = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
            # 초기 임계값, 이전 임계값 저장 변수, 종료 조건 임계값 설정   
            T1=50;T2=0;T0=1
            # 임계값 적용 후 이진 영상을 담을 기억 장소 생성 
            output_img =np.zeros((gray_img.shape[0],gray_img.shape[1]))
            # 각 그룹별 평균값으로 새로운 임계값 계산하는 함수
            # 영상을 두 개의 그룹으로 분할하고 그룹마다 평균값을 계산하여 이 값들을 임계값 계산에 적용
            def threshold_update(h, w, img1, output,T1): #saturation함수로 정의하기
                sum1=0;count1=1;sum2=0;count2=1
                for h in range(img1.shape[0]):
                    for w in range(img1.shape[1]):
                    # 그룹별 픽셀 총합 계산
                        if(output[h,w]==255):
                            sum1 = sum1+ img1[h,w]
                            count1 = count1+1
                        else:
                            sum2 = sum2+ img1[h,w]
                            count2 = count2+1
                # 그룹별 픽셀 평균 계산    
                ave1 = sum1/count1 
                ave2 = sum2/count2
                # 평균으로 새로운 임계값 계산
                T2 = np.int32(ave1+ave2)/2
                return T2
            #1) 영상 초기 임계값 적용 하기
            for h in range(gray_img.shape[0]):
                for w in range(gray_img.shape[1]):
                    if(gray_img[h,w]>T1):
                        output_img[h,w]=255
                    else:
                        output_img[h,w]=0 

            #2) 종료 조건을 만족할 때 까지 계속 반복
            while True:
                #새로운 임계값 생성을 위한 함수 호출
                T2 = threshold_update(gray_img.shape[0], gray_img.shape[1], gray_img, output_img,T1)
                #새로운 임계값과 이전 임계값의 변화 측정
                if(np.abs(T1-T2)<T0):
                    # 종료 조건을 만족하면 새로운 임계값으로 영상 이진화 후 출력
                    for h in range(gray_img.shape[0]):
                        for w in range(gray_img.shape[1]):
                            if(gray_img[h,w]>T2):
                                output_img[h,w]=255
                            else:
                                output_img[h,w]=0
                    break
                #종료 조건을 만족하지 않는다면
                else:
                    # 새로운 임계값으로 다시 이진화 작업
                    T1 = T2
                    for h in range(gray_img.shape[0]):
                        for w in range(gray_img.shape[1]):
                            if(gray_img[h,w]>T1):
                                output_img[h,w]=255
                            else:
                                output_img[h,w]=0
            #그림을 화면에 출력
            plt.subplot(111)
            plt.imshow(output_img, cmap='gray')
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 적응적 임계값 적용
    def AdaptiveThresholding(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # BGR채널순서를 RGB채널로 변경
            gray_img = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
            #가로세로 블록의 개수 
            N=18
            #블록 당 가로와 세로 크기 계산 
            dimh=np.int32(gray_img.shape[0]/N)
            dimw=np.int32(gray_img.shape[1]/N)
            #연산에서 제외될 영상 가장자리 크기 계산 
            dh_rest = np.int32(gray_img.shape[0]%N)
            dw_rest =np.int32(gray_img.shape[1]%N)
            # 임계값 적용 후 이진 영상을 담을 기억 장소 생성 
            mean_img = np.zeros((N,N)) #블록의 평균값 저장 배열
            output_img = np.zeros((gray_img.shape[0],gray_img.shape[1]))
            # 각 블록의 평균값 계산을 위한 함수
            def mean_function(img,dimh,dimw,h,w): 
                count=1;sum=0;ave=0
                for y in range(h, h+dimh):
                    for x in range(w, w+dimw):
                        sum = sum+img[y,x]
                        count = count+1
                # 블록별 픽셀 평균 계산    
                ave = np.int32(sum/count) 
                return ave
            #각 블록의 평균값 계산
            for h in range(0,img1.shape[0]-dh_rest,dimh):
                for w in range(0,img1.shape[1]-dw_rest,dimw):
                    if(h+dimh <img1.shape[0] and w+dimw<img1.shape[1]):
                                mean_img[np.int32(h/dimh),np.int32(w/dimw)]= mean_function(gray_img,dimh,dimw,h,w)       
            #각 블록에 대해 임계값 적용 및 이진화 작업 수행
            for h in range(0,gray_img.shape[0]-dh_rest):
                for w in range(0,gray_img.shape[1]-dw_rest):
                    if(gray_img[h,w]>= mean_img[np.int32(h/dimh),np.int32(w/dimw)]):
                        output_img[h,w]=255
                    else:
                        output_img[h,w]=0
            #그림을 화면에 출력
            plt.subplot(111)
            plt.title("Segmented Image")
            plt.imshow(output_img, cmap='gray')
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 히스토그램 스트레칭
    def HistogramStretching(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0], cv.IMREAD_GRAYSCALE)
            output_img=img1.copy()
            height,width=img1.shape
            high=img1.max()
            low=img1.min()
            for i in range(width):
                for j in range(height):
                    output_img[i][j]=((img1[i][j]-low)*255/(high-low))
            #그림을 화면에 출력
            plt.subplot(2,2,1),plt.axis("off"),plt.imshow(img1,cmap='gray')
            plt.subplot(2,2,2),plt.axis("off"),plt.imshow(output_img,cmap='gray')
            plt.subplot(2,2,3),plt.hist(img1.ravel(), 256, [0,256])
            plt.subplot(2,2,4),plt.hist(output_img.ravel(), 256, [0,256])
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 히스토그램 스트레칭
    def HistogramEqualization(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0], 0)
            img2 = cv.equalizeHist(img1)
            img1_hist=cv.calcHist(img1,[0],None,[256],[0,255])
            plt.subplot(2,1,1), plt.plot(img1_hist)
            img2_hist=cv.calcHist(img2,[0],None,[256],[0,255])
            plt.subplot(2,1,2), plt.plot(img2_hist)
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
#-----------------------------------------------------------
#-----------------------------------------------------------
    # 5장
    # 평균 필터링
    def MeanFiltering(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 계수가 1로 구성된 3x3커널 만들기
            kernel = np.ones((3,3),np.float32)/9
            output_img = cv.filter2D(gray_img,-1,kernel) 
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 중간값 필터링
    def MedianFiltering(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 5x5 중간값 커널 적용하기
            output_img = cv.medianBlur(gray_img,5)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 가우시안 필터링
    def GaussianFiltering(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 5x5 가우시안 커널 적용하기
            output_img = cv.GaussianBlur(gray_img,(5,5),1)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 보존 스무딩
    def ConservativeSmoothing(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            output_img = np.zeros((img1.shape[0],img1.shape[1]))
            center=0;current=0;min=255;max=0;ed=1 # 3x3커널일 경우 1, 5x5 커널일 경우 2
            for h in range(ed,img1.shape[0]-ed,1):
                for w in range(ed, img1.shape[1]-ed,1):
                    #초기값 설정
                    center = gray_img[h,w]
                    min = gray_img[h-ed,w-ed]
                    max = gray_img[h-ed,w-ed]
                    #최대, 최소 구하기
                    for m in range(-ed,ed,1):
                        for n in range(-ed,ed,1):
                            if( m==0 and n==0):
                                continue
                            else:
                                current = gray_img[h+m,w+n]
                            if (min > current):
                                min = current
                            if (max < current):
                                max = current   
                    if (center> min and center < max):
                        output_img[h,w] = center
                    elif (center > max):
                        center = max
                    elif (center < min):
                        center = min
                    output_img[h,w] = center
            #그림을 화면에 출력
            plt.subplot(111)
            plt.imshow(output_img, cmap='gray')
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
 
    # 언샤프 필터링 edge
    def UnsharpFiltering_edge(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 5x5커널 적용하기
            mean_img = cv.blur(gray_img,(5,5))
            edge_img = cv.addWeighted(gray_img, 1.0, mean_img, -1.0, 0)
            self.filesave(edge_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 언샤프 필터링
    def UnsharpFiltering(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 5x5커널 적용하기
            mean_img = cv.blur(gray_img,(5,5))
            edge_img = cv.addWeighted(gray_img, 1.0, mean_img, -1.0, 0)
            output_img = cv.addWeighted(gray_img, 1.0, edge_img, 3.0, 0)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
#-----------------------------------------------------------
#-----------------------------------------------------------
    # 6장
    # 로버트 크로스 에지 검출
    def RobertCrossEdgeDetector(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 로버트 크로스 필터
            gx=np.array([[-1, 0], [0, 1]], dtype=int)
            gy=np.array([[0, -1], [1, 0]], dtype=int)
            # 로버트 크로스 컨벌루션
            x=cv.filter2D(gray_img, -1, gx)
            y=cv.filter2D(gray_img, -1, gy)
            # 절대값 취하기
            absX=cv.convertScaleAbs(x)
            absY=cv.convertScaleAbs(y)
            output_img=cv.addWeighted(absX, 0.5, absY, 0.5, 0)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 소벨 에지 검출
    def SobelEdgeDetector(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # Sobel operator
            x=cv.Sobel(gray_img, -1, 1, 0)
            y=cv.Sobel(gray_img, -1, 0, 1)
            # Turn uint8, image fusion
            absX=cv.convertScaleAbs(x)
            absY=cv.convertScaleAbs(y)
            output_img=cv.addWeighted(absX, 0.5, absY, 0.5, 0)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 프르윗 에지 검출
    def PrewittEdgeDetector(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 프르윗 필터
            gx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
            gy = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
            # 프르윗 필터 컨벌루션
            x = cv.filter2D(gray_img, -1, gx)
            y = cv.filter2D(gray_img, -1, gy)
            # uint8 타입(0~255)로 변경하고 영상 합하기
            absX = cv.convertScaleAbs(x)
            absY = cv.convertScaleAbs(y)
            output_img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 캐니 에지 검출
    def CannyEdgeDetector(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 케니 에지 컨벌루션 연산하기
            output_img = cv.Canny(gray_img,100,250)     # 임계값 100,250
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 라플라시안 에지 검출
    def LaplacianEdgeDetector(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # 라플라시안 에지 컨벌루션 연산하기
            laplacian = cv.Laplacian(gray_img,-1,1)
            output_img = laplacian/laplacian.max()
            #그림을 화면에 출력
            plt.subplot(111)
            plt.imshow(output_img, cmap='gray')
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 가우시안-라플라시안 에지 검출
    def GaussianLaplacianEdgeDetector(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # color영상을 gray영상으로 만들기
            gray_img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            # Apply Gaussian Blur
            blur = cv.GaussianBlur(gray_img,(3,3),1)
            # 라플라시안 에지 컨벌루션 연산하기
            laplacian = cv.Laplacian(blur,-1,1)
            output_img = laplacian/laplacian.max()
            #그림을 화면에 출력
            plt.subplot(111)
            plt.imshow(output_img, cmap='gray')
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
#-----------------------------------------------------------
    # 7장
    # 최근접 이웃 보간법
    def NearestNeighborInterpoliation(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            output_img = cv.resize(img1,None,fx=2, fy=2, interpolation = cv.INTER_NEAREST)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 양선형 보간법
    def LinearNeighborInterpoliation(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            output_img = cv.resize(img1,None,fx=0.5, fy=0.5, interpolation = cv.INTER_LINEAR)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 회전 변환
    def Rotation(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            rows,cols = img1.shape[:2]
            # 회전점을 영상 모서리 -> 영상의 중심으로 변경
            M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),180,1)
            output_img = cv.warpAffine(img1,M,(cols*1,rows*1),flags = cv.INTER_LINEAR)
            self.filesave(output_img)
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 대칭 변환
    def Flipping(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # BGR영상을 RGB영상으로 변환
            img1= cv.cvtColor(img1, cv.COLOR_BGR2RGB)
            flipVertical = cv.flip(img1, 0)
            flipHorizontal = cv.flip(img1, 1)
            flipBoth = cv.flip(img1, -1)
            #그림을 화면에 출력
            plt.subplot(2,2,1), plt.imshow(img1)
            plt.axis("off")
            plt.subplot(2,2,2), plt.imshow(flipVertical)
            plt.axis("off")
            plt.subplot(2,2,3), plt.imshow(flipHorizontal)
            plt.axis("off")
            plt.subplot(2,2,4), plt.imshow(flipBoth)
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
    
    # 이동 변환
    def Translation(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            # BGR영상을 RGB영상으로 변환
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
            height,width = img1.shape[:2]
            M = np.float32([[1,0,100],[0,1,50]])
            out1 = cv.warpAffine(img1,M,(width,height))
            M = np.float32([[1,0,-50],[0,1,-50]])
            out2 = cv.warpAffine(img1,M,(width,height))
            #그림을 화면에 출력
            plt.subplot(2,1,1), plt.imshow(out1)
            plt.axis("off")
            plt.subplot(2,1,2), plt.imshow(out2)
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 어파인 변환
    def Affine(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            #영상 크기변환
            resize = cv.resize(img1,None,fx=0.5, fy=0.5, interpolation = cv.INTER_LINEAR)
            height,width = resize.shape[:2]
            # 회전점을 영상 모서리 -> 영상의 중심으로 변경
            M = cv.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0),-15,1)
            # 회전점을 영상 모서리로 한 경우
            rotate = cv.warpAffine(resize,M,(width,height))
            M = np.float32([[1,0,30],[0,1,+20]])
            translate = cv.warpAffine(rotate,M,(width,height))
            #그림을 화면에 출력
            plt.subplot(2,2,1), plt.imshow(img1)
            plt.axis("off")
            plt.subplot(2,2,2), plt.imshow(resize)
            plt.axis("off")
            plt.subplot(2,2,3), plt.imshow(rotate)
            plt.axis("off")
            plt.subplot(2,2,4), plt.imshow(translate)
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
#-----------------------------------------------------------
#-----------------------------------------------------------
    # 8장
    # 팽창 연산
    def Dilation(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            kernel = np.ones((3,3),np.uint8)
            dilation = cv.dilate(img1,kernel,iterations = 1)
            #그림을 화면에 출력
            plt.subplot(2,1,1), plt.imshow(dilation)
            plt.axis("off")
            plt.subplot(2,1,2), plt.imshow(dilation-img1)
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)

    # 침식 연산
    def Erosion(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            kernel = np.ones((3,3),np.uint8)
            erode1 = cv.erode(img1,kernel,iterations = 1)
            erode3 = cv.erode(img1,kernel,iterations = 3)
            #그림을 화면에 출력
            plt.subplot(2,1,1), plt.imshow(erode1)
            plt.axis("off")
            plt.subplot(2,1,2), plt.imshow(erode3)
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 열림 연산
    def Opening(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            kernel = np.ones((5,5),np.uint8)
            opening = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
            #그림을 화면에 출력
            plt.subplot(111), plt.imshow(opening)
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
            
    # 닫힘 연산
    def Closing(self):
        try:
            global pre_picture_path
            img1 = cv.imread(pre_picture_path[0])
            kernel = np.ones((5,5),np.uint8)
            closing = cv.morphologyEx(img1, cv.MORPH_CLOSE, kernel)
            #그림을 화면에 출력
            plt.subplot(111), plt.imshow(closing)
            plt.axis("off")
            plt.savefig('plt.png', bbox_inches='tight', pad_inches=0)
            self.filePrint()
        except Exception as e:
            messagebox.showinfo("예외가 발생했습니다", e)
#-----------------------------------------------------------
    # 로그아웃
    def logout(self):
        exit()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.setFixedWidth(1040)
    widget.setFixedHeight(710)
    widget.addWidget(win)
    widget.show()
    app.exec_()