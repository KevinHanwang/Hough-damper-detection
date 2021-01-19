#include <opencv2/opencv.hpp>

#include <math.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

int DISTANCE = -10; //判断圆心距与半径的差值，用来筛选圆
int delay = 50;
int mid_line;
cv::Scalar scalarL= cv::Scalar(0,0,0);
cv::Scalar scalarH= cv::Scalar(180,23,130); 

int main(int argc, char** argv)
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
	cvResizeWindow("video", 1800, 1200);

	while (true)
	{
		Mat frame;
		capture >> frame;

        Mat src = frame(Range(0,3*frame.rows/5), Range(frame.cols/2 -300, frame.cols/2 +200));
        // Mat l_midImage = frame(Range(0,3*frame.rows/5), Range(frame.cols/2 -300, frame.cols/2 +200));

        //【1】Mat变量定义   
        Mat midImage,l_midImage, mid;//目标图的定义

        //【2】转为灰度图并进行图像平滑
        // cvtColor(src,midImage, CV_BGR2GRAY);//转化边缘检测后的图为灰度图
        // GaussianBlur(midImage, midImage, Size(15, 15), 2, 2 );
        // Canny(midImage, midImage, 45, 120, 3);
        
        // cvtColor(midImage,dstImage, COLOR_GRAY2BGR);//转化边缘检测后的图为灰度图

        //检测直线
        Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
        morphologyEx(src, l_midImage, MORPH_GRADIENT, element);
        
        Canny(l_midImage, mid, 45, 120, 3);
        vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
	    HoughLinesP(mid, lines, 1, CV_PI/180, 150, 70, 50);
        // 存放离圆最近的两个点{x1,y1,x2,y2}
        vector<int> closedToCir = {0,0,0,0};
        //【4】依次在图中绘制出每条线段
        cout << lines.size() << endl;

        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            //根据斜率留下属于输电线的直线
            double k;
            if(l[0]==l[2]) k = 0.0; 
            else k = (l[1]-l[3])/(l[0]-l[2]);
            
            if(abs(k) < 7.0 || abs(k) > 8.5){
                continue;
            } 
            cout << k << endl;

            if(k>0){
                if(l[1] > l[3]){
                    if(l[1] > closedToCir[3]){
                        closedToCir[3] = l[1];
                        closedToCir[2] = l[0];
                    }
                }
                else{
                    if(l[3] > closedToCir[3]){
                        closedToCir[3] = l[3];
                        closedToCir[2] = l[2];
                    }
                }
            }
            else{
                if(l[1] > l[3]){
                    if(l[1] > closedToCir[3]){
                        closedToCir[1] = l[1];
                        closedToCir[0] = l[0];
                    }
                }
                else{
                    if(l[3] > closedToCir[3]){
                        closedToCir[1] = l[3];
                        closedToCir[0] = l[2];
                    }
                }
            }

            line( src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 5, CV_AA);
        }

        if(closedToCir[0] && closedToCir[2]){
            mid_line = (closedToCir[2]+closedToCir[0])/2;
            line( src, Point(mid_line, 0), Point(mid_line,max(closedToCir[1],closedToCir[3])), Scalar(0,255,0), 3, CV_AA);
        }

        //【3】进行霍夫圆变换
        // cvtColor(l_midImage,midImage, CV_BGR2GRAY);//转化边缘检测后的图为灰度图
        // GaussianBlur(midImage, midImage, Size(3, 3), 2, 2 );
        // Canny(midImage, midImage, 45, 120, 3);
        //hsv颜色分离
        cvtColor(src, midImage, CV_BGR2HSV);
        inRange(midImage,scalarL,scalarH,mid);

        vector<Vec3f> circles;
        HoughCircles( mid, circles, CV_HOUGH_GRADIENT,1.5, 150, 200, 30, 20, 200);
        cout << circles.size() << endl;

         //【4】依次在图中绘制出圆
        for( size_t i = 0; i < circles.size(); i++ ){
        // if(circles.size() != 0){
            // for( size_t i = 0; i < 1; i++ )
            // {
            //参数定义
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);

            //圆心不与输电线中心线对称的话，跳出
            int bias = cvRound(circles[i][0]) - mid_line;
            if(abs(bias) > 30) continue;

            //判断圆与直线是否太近，如果太近，那么舍弃
            int distance1 = pow((circles[i][0] - closedToCir[0]),2) + pow((circles[i][1] - closedToCir[1]),2);
            int distance2 = pow((circles[i][0] - closedToCir[2]),2) + pow((circles[i][1] - closedToCir[3]),2);
            int r_2 = pow(circles[i][2],2);
            int dis = min(distance1-r_2, distance2 -r_2);
            if(dis < DISTANCE) continue;

            // //通过判断圆心与最近端点的高度来筛选圆
            int y_bias1 = circles[i][1] - closedToCir[1];
            int y_bias2 = circles[i][1] - closedToCir[3];
            int y_bias = min(y_bias1, y_bias2);
            if(y_bias < circles[i][2]) continue;

            //绘制圆心
            circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
            //绘制圆轮廓
            circle( src, center, radius, Scalar(255,0,0), 10, 8, 0 );
            //打印圆心坐标
            // printf("x = %d,y = %d\n",cvRound(circles[i][0]),cvRound(circles[i][1]));
            // }
        }

        imshow("video", frame);

        if(delay>=0&&waitKey (delay)>=32)
            waitKey(0);
	}
	return 0;
}