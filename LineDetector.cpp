#include <iostream>
#include <opencv2/opencv.hpp>

#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

Mat g_srcImage, g_midImage, g_midImage2;//原始图、中间图和效果图
vector<Vec4i> g_lines;//定义一个矢量结构g_lines用于存放得到的线段矢量集合
//变量接收的TrackBar位置参数
int g_nthreshold=116;
int g_merge = 120;
int g_cannyLowThreshold=40;

static void on_HoughLines(int, void*);//回调函数

int main(int argc, char** argv)
{
    VideoCapture capture(argv[1]);
    if(!capture.isOpened()){
        cout << "video not open" << endl;
        return 1;
    }
    int delay = 30;
	
    while(true){
		Mat frame;
        capture >> frame;
		frame = frame(Range(0,frame.rows), Range(0, 3*frame.cols/5));
        namedWindow("after",0);
        cvResizeWindow("after", 1200, 800);

		g_srcImage = frame(Range(frame.rows/2 -50,frame.rows/2 + 150), Range(0, frame.cols));
		// g_srcImage = frame(Range(0,frame.rows/2 + 50), Range(0, frame.cols));

        //定义核
        Mat element = getStructuringElement(MORPH_RECT, Size(35, 35)); 
		// 开运算
		morphologyEx(g_srcImage,g_midImage, MORPH_OPEN, element);
		Mat element1 = getStructuringElement(MORPH_RECT, Size(3,3));
        morphologyEx(g_midImage, g_midImage, MORPH_GRADIENT, element1);
        //进行边缘检测和转化为灰度图
        cvtColor(g_midImage,g_midImage, CV_BGR2GRAY); //转化边缘检测后的图为灰度图
        // GaussianBlur( g_midImage, g_midImage, Size(15, 15), 2, 2);
        vector<Vec4i> mylines;
		Canny( g_midImage, g_midImage, g_cannyLowThreshold, g_cannyLowThreshold*3, 3);
		HoughLinesP(g_midImage, mylines, 1, CV_PI/180, g_nthreshold+1, 50, g_merge+1);

		//进行形态学操作 闭运算
		morphologyEx(g_srcImage,g_midImage2, MORPH_CLOSE, element);
		morphologyEx(g_midImage2, g_midImage2, MORPH_GRADIENT, element1);
		cvtColor(g_midImage2,g_midImage2, CV_BGR2GRAY); //转化边缘检测后的图为灰度图
        vector<Vec4i> mylines2;
		Canny( g_midImage2, g_midImage2, g_cannyLowThreshold, g_cannyLowThreshold*3, 3);
		HoughLinesP(g_midImage2, mylines2, 1, CV_PI/180, g_nthreshold+1, 50, g_merge+1);

		cout << "********************************" << endl;
		
		//循环遍历绘制每一条线段
		for( size_t i = 0; i < mylines.size(); i++ )
		{
			Vec4i l = mylines[i];
			//根据斜率留下属于输电线的直线
            double k;
            if(l[0]==l[2]) k = 1000; 
            else k = static_cast<double> ((l[1]-l[3]))/(l[0]-l[2]);   
			if(abs(k) > 0.3 || abs(k) < 0.05 || k < 0) continue;     
            // if(abs(k) < 0.3 || abs(k) > 6) continue;
            cout << k << endl;
			line( g_srcImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		}
		for( size_t i = 0; i < mylines2.size(); i++ )
		{
			Vec4i l = mylines2[i];
			//根据斜率留下属于输电线的直线
            double k;
            if(l[0]==l[2]) k = 1000; 
            else k = static_cast<double> ((l[1]-l[3]))/(l[0]-l[2]);
			if(abs(k) > 0.3 || abs(k) < 0.05 || k < 0) continue;  
            // if(abs(k) < 0.3 || abs(k) > 6) continue;
            cout << k << endl;
			line( g_srcImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		}
		//显示图像
		imshow("after",frame);

        if(delay>=0 && waitKey(delay)>=32) waitKey(0);
    } 
}

// int threshold_value = 50;
// int max_value = 255;
// const char* output_title = "detect lines";

// int main(int argc, char** argv)
// {
// 	//【1】载入原始图和Mat变量定义   
// 	Mat Image = imread(argv[1]);
//     int height = Image.rows;
//     int width = Image.cols;
//     cout<< height << "x" << width << endl; 

//     Mat srcImage = Image(Range(0,height), Range(width/2 -200, width/2 + 200));
// 	Mat midImage,dstImage=srcImage;//临时变量和目标图的定义

//     // namedWindow(output_title, CV_WINDOW_AUTOSIZE);
// 	// createTrackbar("threshold", output_title, &threshold_value, max_value, houghlinedetect);
 
// 	//【2】进行边缘检测和转化为灰度图
// 	Canny(srcImage, midImage, 50, 200, 3);//进行一此canny边缘检测
// 	// cvtColor(midImage,dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
 
// 	//【3】进行霍夫线变换
// 	vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
// 	HoughLinesP(midImage, lines, 1, CV_PI/180, 130, 80, 30);
 
// 	//【4】依次在图中绘制出每条线段
// 	for( size_t i = 0; i < lines.size(); i++ )
// 	{
// 		Vec4i l = lines[i];
// 		line( dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 1, CV_AA);
// 	}
//     cout << lines.size() << endl;
 
// 	//【5】显示原始图  
//     namedWindow("Original", 0);
// 	cvResizeWindow("Original", 1200, 800); 
// 	imshow("Original", srcImage);  
 
// 	//【6】边缘检测后的图 
//     namedWindow("edge", 0);
// 	cvResizeWindow("edge", 1200, 800); 
// 	imshow("edge", midImage);  
 
// 	//【7】显示效果图  
//     namedWindow("after", 0);
// 	cvResizeWindow("after", 1200, 800); 
// 	imshow("after", dstImage);  
 
// 	waitKey(0);  
 
// 	return 0;  
// }