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

// 腐蚀膨胀处理参数
int g_nTrackbarNumer = 3;//0表示腐蚀erode, 1表示膨胀dilate
int g_nStructElementSize = 15; //结构元素(内核矩阵)的尺寸

/** @函数 main */
int main( int argc, char** argv )
{
    /// 装载图像
    src = imread( argv[1] );
    // cout << "width :" << src.rows << ", height :" << src.cols << endl;

    Mat src_cut = src(Range(0, 5*src.rows/7), Range(src.cols/2 -300, src.cols/2 + 300));

    if( !src.data )
    { return -1; }

    namedWindow( "original",0);
    cvResizeWindow("original", 1200, 800);
    // imshow("original", src);

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

    // cout << "M = " << endl << " " << abs_grad_x << endl << endl;

    vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
    HoughLinesP(abs_grad_x, lines, 1, CV_PI/90, 1500, 100, 300);

    cout << "number of lines : " << lines.size() << endl;

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        //根据斜率留下属于输电线的直线
        double k;
        if(l[0]==l[2]) k = 0.0; 
        else k = (l[1]-l[3])/(l[0]-l[2]);
        
        // if(abs(k) < 7.0 || abs(k) > 8.5){
        //     continue;
        // } 
        cout << k << endl;

        line( src, Point(l[0]+1200, l[1]), Point(l[2]+1200, l[3]), Scalar(0,0,255), 1, CV_AA);
    }

    vector<Vec3f> circles;
    HoughCircles( abs_grad_x, circles, CV_HOUGH_GRADIENT,1.5, 300, 200, 150, 20, 200);
    cout << "number of circles : " << circles.size() << endl;

    for( size_t i = 0; i < circles.size(); i++ ){
        //参数定义
        Point center(cvRound(circles[i][0]+1200), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        //绘制圆心
        circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        //绘制圆轮廓
        circle( src, center, radius, Scalar(255,0,0), 10, 8, 0 );
    }

    imshow("original", src);

    namedWindow( "cutted",0);
    cvResizeWindow("cutted", 1200, 800);
    imshow("cutted", src_cut);

    namedWindow( "closed",0);
    cvResizeWindow("closed", 1200, 800);
    imshow("closed", abs_grad_x);

    /// 等待用户反应
    waitKey(0);

    return 0;
}