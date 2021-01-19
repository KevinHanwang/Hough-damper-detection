/*
用途：防振锤检测
*/
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

/// 全局变量

Mat src, src_gray;
Mat dst, detected_edges;
Mat closed;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 2;
int kernel_size = 3;

int delay = 250;

// 腐蚀膨胀处理参数
int g_nTrackbarNumer = 3;//0表示腐蚀erode, 1表示膨胀dilate
int g_nStructElementSize = 15; //结构元素(内核矩阵)的尺寸

/** @函数 main */
int main( int argc, char** argv )
{
    //读取视频或摄像头
	VideoCapture capture(argv[1]);
    if(!capture.isOpened())
    {
        std::cout<<"video not open."<<std::endl;
        return 1;
    }
    // //获取当前视频帧率
    // double rate = capture.get(CV_CAP_PROP_FPS);
    namedWindow("video", 0);
	cvResizeWindow("video", 1200, 2000);

	while (true)
	{
		Mat frame;
		capture >> frame;

        // 手机拍摄视频导致旋转90度
        Mat dst;
        transpose(frame, dst);
        Mat dst2;
        flip(dst, dst2, 1);

        Mat src_cut = dst2(Range(0,5*dst2.rows/7), Range(dst2.cols/2 - 300, dst2.cols/2 + 300));

        Mat element = getStructuringElement(MORPH_RECT, Size(33,33),Point(16, 16));
        morphologyEx(src_cut, closed, MORPH_CLOSE, element);
        /// 原图像转换为灰度图像
        cvtColor( closed, src_gray, CV_BGR2GRAY);

        // 使用 3x3内核降噪
        blur( src_gray, detected_edges, Size(3,3));

        // Scharr滤波器
        Mat grad_x, abs_grad_x;
        Scharr(detected_edges, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(grad_x, abs_grad_x);

        vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
        HoughLinesP(abs_grad_x, lines, 1, CV_PI/90, 1000, 100, 300);

        cout << "number of lines : " << lines.size() << endl;

        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            //根据斜率留下属于输电线的直线
            // double k;
            // if(l[0]==l[2]) k = 0.0; 
            // else k = (l[1]-l[3])/(l[0]-l[2]);
            
            // if(abs(k) < 7.0 || abs(k) > 8.5){
            //     continue;
            // } 
            // cout << k << endl;

            line( src_cut, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 5, CV_AA);
        }

        vector<Vec3f> circles;
        HoughCircles( abs_grad_x, circles, CV_HOUGH_GRADIENT,1.5, 300, 200, 150, 20, 200);
        cout << "number of circles : " << circles.size() << endl;

        for( size_t i = 0; i < circles.size(); i++ ){
            //参数定义
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            //绘制圆心
            circle( src_cut, center, 3, Scalar(0,255,0), -1, 8, 0 );
            //绘制圆轮廓
            circle( src_cut, center, radius, Scalar(255,0,0), 40, 8, 0 );
        }

        imshow("video", dst2);

        if(delay>=0&&waitKey (delay)>=32)
            waitKey(0);
    }

    return 0;
}