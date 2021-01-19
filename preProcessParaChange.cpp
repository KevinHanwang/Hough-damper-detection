/*
用途：形态学处理&canny边缘检测可视化调参
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
int g_nStructElementSize = 16; //结构元素(内核矩阵)的尺寸
 
void Process();//膨胀和腐蚀的处理函数
void on_TrackbarNumChange(int, void *);//回调函数
void on_ElementSizeChange(int, void *);//回调函数

 void Process()
{
       //获取自定义核
       Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));
 
       //进行腐蚀或膨胀操作
       if(g_nTrackbarNumer== 0) {     
              erode(src, closed, element);
       }
       else if(g_nTrackbarNumer== 1){
              dilate(src, closed, element);
       }
       else if(g_nTrackbarNumer== 2){
              morphologyEx(src, closed, MORPH_OPEN, element);
       }
       else if(g_nTrackbarNumer== 3){
              morphologyEx(src, closed, MORPH_CLOSE, element);
       }
       else if(g_nTrackbarNumer== 4){
              morphologyEx(src, closed, MORPH_GRADIENT, element);
       }
       else if(g_nTrackbarNumer== 5){
              morphologyEx(src, closed, MORPH_TOPHAT, element);
       }
       else {
              morphologyEx(src, closed, MORPH_BLACKHAT, element);
       }
       cvtColor( closed, src_gray, CV_BGR2GRAY );

       /// 使用 3x3内核降噪
       blur( src_gray, detected_edges, Size(3,3) );

       namedWindow( "3_gray",0);
       cvResizeWindow("3_gray", 1200, 800);
       imshow("3_gray", detected_edges);

       /// 运行Canny算子
       // Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
       
       // Scharr滤波器
       Mat grad_x, abs_grad_x;
       Scharr(detected_edges, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
       convertScaleAbs(grad_x, abs_grad_x);
 
       //显示效果图
       imshow("closed", abs_grad_x);
}
 
void on_TrackbarNumChange(int, void *)
{
       //腐蚀和膨胀之间效果已经切换，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
       Process();
}

void on_ElementSizeChange(int, void *)
{
       //内核尺寸已改变，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
       Process();
}

void on_CannyChange(int, void *)
{
       //内核尺寸已改变，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
       Process();
}


/** @函数 main */
int main( int argc, char** argv )
{
  /// 装载图像
  src = imread( argv[1] );

  if( !src.data )
  { return -1; }

  namedWindow( "original",0);
  cvResizeWindow("original", 1200, 800);
  imshow("original", src);

  Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));
  morphologyEx(src, closed, MORPH_CLOSE, element);
  /// 原图像转换为灰度图像
  cvtColor( closed, src_gray, CV_BGR2GRAY );

  namedWindow( "closed",0);
  cvResizeWindow("closed", 1200, 800);
  imshow("closed", closed);

  //创建轨迹条
  createTrackbar("Erotion/Dilation", "closed", &g_nTrackbarNumer, 6, on_TrackbarNumChange);
  createTrackbar("kernel_size", "closed",&g_nStructElementSize, 21, on_ElementSizeChange);
  // createTrackbar( "Min Threshold:", "closed", &lowThreshold, max_lowThreshold, on_CannyChange );

  /// 等待用户反应
  waitKey(0);

  return 0;
}