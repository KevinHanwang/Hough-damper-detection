#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
//-----------------------------------【全局变量声明部分】--------------------------------------
//		描述：全局变量声明
//-----------------------------------------------------------------------------------------------
Mat g_srcImage, g_dstImage, g_midImage, img;//原始图、中间图和效果图
vector<Vec4i> g_lines;//定义一个矢量结构g_lines用于存放得到的线段矢量集合
//变量接收的TrackBar位置参数
int g_nthreshold=100;
int g_merge = 10;
int g_cannyLowThreshold=1;

static void on_HoughLines(int, void*);//回调函数

int main(int argc, char** argv)
{
	
	//载入原始图和Mat变量定义   
	g_srcImage = imread(argv[1]);
    // img = g_srcImage;

	//显示原始图 
    namedWindow("before",0);
    cvResizeWindow("before", 2000, 1500); 
	imshow("before", g_srcImage);  

	//创建滚动条
	namedWindow("after",0);
    cvResizeWindow("after", 2000, 1500);

    //定义核
    Mat element = getStructuringElement(MORPH_RECT, Size(11, 11)); 
    //进行形态学操作
    morphologyEx(g_srcImage,g_midImage, MORPH_CLOSE, element);

    // imshow("before1", g_srcImage);
    // system("pause"); 

	// //调用一次回调函数，调用一次HoughLinesP函数

    //进行边缘检测和转化为灰度图
    cvtColor(g_midImage,g_midImage, CV_BGR2GRAY); //转化边缘检测后的图为灰度图
    GaussianBlur( g_midImage, g_midImage, Size(15, 15), 2, 2);

    createTrackbar("value", "after", &g_nthreshold,200,on_HoughLines);
    createTrackbar("value_merge", "after", &g_merge,200,on_HoughLines);
    createTrackbar( "canny_thre", "after", &g_cannyLowThreshold, 120, on_HoughLines);

	on_HoughLines(0,0);

    // cout << "g_lines.size = " << g_lines.size() << endl;

	//显示效果图  
	imshow("before", g_srcImage);  
	waitKey(0);  
	return 0;  

}
//-----------------------------------【on_HoughLines( )函数】--------------------------------
//		描述：【顶帽运算/黑帽运算】窗口的回调函数
//----------------------------------------------------------------------------------------------
static void on_HoughLines(int, void*)
{
	//定义局部变量储存全局变量
	// Mat img_=img.clone();
    Mat srcImage=g_srcImage.clone();
	Mat midImage=g_midImage.clone();
	//调用HoughLinesP函数
	vector<Vec4i> mylines;

    Canny( midImage, midImage, g_cannyLowThreshold, g_cannyLowThreshold*3, 3);

	HoughLinesP(midImage, mylines, 1, CV_PI/180, g_nthreshold+1, 50, g_merge+1);
    
	//循环遍历绘制每一条线段
	for( size_t i = 0; i < mylines.size(); i++ )
	{
		Vec4i l = mylines[i];
		line( srcImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 5, CV_AA);
	}
	//显示图像
    imshow("before",srcImage);
	imshow("after",midImage);
}