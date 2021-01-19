#include <opencv2/opencv.hpp>

#include <math.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

// int max_centers = 20;
// Point *pp = new Point[max_centers*2];

// struct Rad {
// 	int rad ;
// 	int countR ;
// };
 
int main()
{
	//读取视频或摄像头
	VideoCapture capture(0);
 
	while (true)
	{
		Mat frame;
		capture >> frame;

        //【1】Mat变量定义   
        Mat midImage;//目标图的定义

        //【2】转为灰度图并进行图像平滑
        cvtColor(frame,midImage, CV_BGR2GRAY);//转化边缘检测后的图为灰度图
        GaussianBlur( midImage, midImage, Size(9, 9), 2, 2 );

        //【3】进行霍夫圆变换
        vector<Vec3f> circles;
        HoughCircles( midImage, circles, CV_HOUGH_GRADIENT,1.5, 10, 200, 70, 0, 100 );

        //【4】依次在图中绘制出圆
        // for( size_t i = 0; i < circles.size(); i++ )
        if(circles.size() != 0){
            for( size_t i = 0; i < 1; i++ )
            {
            //参数定义
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            //绘制圆心
            circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
            //绘制圆轮廓
            circle( frame, center, radius, Scalar(155,50,255), 3, 8, 0 );
            //打印圆心坐标
            // printf("x = %d,y = %d\n",cvRound(circles[i][0]),cvRound(circles[i][1]));
            }
        }

        imshow("video", frame);

		waitKey(100);	//延时30
	}
	return 0;
}