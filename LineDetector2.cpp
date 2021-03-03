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
int height, width;
vector<Vec4i> g_lines;//定义一个矢量结构g_lines用于存放得到的线段矢量集合
//变量接收的TrackBar位置参数
int g_nthreshold=120;
int g_merge = 50;
int g_cannyLowThreshold=6;

static void on_HoughLines(int, void*);//回调函数
vector<double> line_fit(vector<Vec4i>& h);

vector<int> kmeans(vector<int>& sample,int k);

int main(int argc, char** argv)
{
	
	//载入原始图和Mat变量定义   
	g_srcImage = imread(argv[1]);
    height = g_srcImage.rows;
    width = g_srcImage.cols;

	// g_srcImage = src(Range(0, 5*height/7), Range(width/2 -100, width/2 + 100));

	//创建滚动条
	namedWindow("after",0);
    cvResizeWindow("after", 2000, 1500);

    //定义核
    // Mat element = getStructuringElement(MORPH_RECT, Size(11, 11)); 
    // //进行形态学操作
    // morphologyEx(g_srcImage,g_midImage, MORPH_CLOSE, element);

    // imshow("before1", g_srcImage);
    // system("pause"); 

	// //调用一次回调函数，调用一次HoughLinesP函数

    //进行边缘检测和转化为灰度图
    cvtColor(g_srcImage,g_midImage, CV_BGR2GRAY); //转化边缘检测后的图为灰度图

	//显示原始图 
    namedWindow("before",0);
    cvResizeWindow("before", 2000, 1500); 
	imshow("before", g_srcImage);  

	// equalizeHist(g_midImage, g_midImage);
	//显示原始图 
    // namedWindow("equalize",0);
    // cvResizeWindow("equalize", 2000, 1500); 
	// imshow("equalize", g_midImage);  

    GaussianBlur( g_midImage, g_midImage, Size(15, 15), 2, 2);

    // // Scharr滤波器
    // Mat grad_x, abs_grad_x;
    // Scharr(g_midImage, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    // convertScaleAbs(grad_x, abs_grad_x);

    createTrackbar("threshod", "after", &g_nthreshold,200,on_HoughLines);
    createTrackbar("maxLineGap", "after", &g_merge,200,on_HoughLines);
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
    Mat cutImage;

    Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3)); 
    morphologyEx(midImage,midImage, MORPH_GRADIENT, element1);
    namedWindow("gradient",0);
    cvResizeWindow("gradient", 2000, 1500); 
    imshow("gradient", midImage);

	//调用HoughLinesP函数
	vector<Vec4i> mylines;
    vector<Vec4i> mylines_left;
    vector<Vec4i> mylines_right;

	Canny( midImage, midImage, g_cannyLowThreshold, g_cannyLowThreshold*3, 3);

    Mat mask = Mat::zeros(midImage.size(), CV_8UC1);
    Point PointArray[4];
    PointArray[0] = Point(width/2 -100, 0);
    PointArray[1] = Point(width/2 +100, 0);
    PointArray[2] = Point(width/2 +70 ,5*height/7);
    PointArray[3] = Point(width/2 -70 ,5*height/7);
    fillConvexPoly(mask,PointArray,4,Scalar(255));
    bitwise_and(mask,midImage,cutImage);

    // imshow("mask", mask);

	cout << "start 1" << endl;
    HoughLinesP(cutImage, mylines, 1, CV_PI/180, g_nthreshold+1, 50, g_merge+1);
    vector<int> slopeset;
    
	//循环遍历绘制每一条线段,斜率小于10的直接删除，并区分左右直线
	for(auto it = mylines.begin(); it != mylines.end(); ++it){
		Vec4i L = *it;
		int k;
        if(abs(L[2] - L[0]) < 1){
            k = 150;
        }
		else 
            k = (L[3]-L[1])/(L[2]-L[0]);
		// putText(srcImage,to_string(k),Point(L[2], L[3]),FONT_HERSHEY_SIMPLEX,0.8,Scalar(255,23,0),2,8);
		if(abs(k) < 10){
			it = mylines.erase(it);
            --it;
            continue;
		}
		slopeset.push_back(k);
	}
    cout << "slopeset : " << slopeset.size() << endl;
    for( int i=0; i<slopeset.size(); ++i){
        cout << slopeset[i] << " " ;
    }
    cout << endl;

    if(slopeset.size() > 1){
        vector<int> slopes = kmeans(slopeset, 2);
        if(slopes[0] != INT_MIN) {
            cout << "slope of two sides : " << slopes[0] << " " << slopes[1] << endl;
            float slopeThresh = (slopes[0] + slopes[1]) / 2;
            // 如果左右斜率同方向，那么flag为0，否则为1;
            int flag = 0;
            if( slopes[0] * slopes[1] < 0 )
                flag = 1;
            cout << "flag : " << flag << endl;

            for(auto it = mylines.begin(); it != mylines.end(); ++it){
                Vec4i L = *it;
                int k;
                if(abs(L[2] - L[0]) < 1){
                    k = 150;
                }
                else 
                    k = (L[3]-L[1])/(L[2]-L[0]);
                // putText(srcImage,to_string(k),Point(L[2], L[3]),FONT_HERSHEY_SIMPLEX,0.8,Scalar(255,23,0),2,8);
                // line( srcImage, Point(L[0], L[1]), Point(L[2], L[3]), Scalar(0,0,255), 5, CV_AA);
                if(flag){
                    // 斜率不同符号，大斜率在左边
                    if(k < slopeThresh)
                        mylines_left.push_back(L);
                    else
                        mylines_right.push_back(L);
                }
                else{
                    // 斜率同符号，大斜率在右边
                    if(k > slopeThresh)
                        mylines_left.push_back(L);
                    else
                        mylines_right.push_back(L);
                }
            }

            cout << mylines_left.size() << "  " << mylines_right.size() << endl;
            vector<double> lineFitLeft = line_fit(mylines_left);
            vector<double> lineFitRight = line_fit(mylines_right);

            int crossPoint_y;
            if(mylines_left.size() > 0 && mylines_right.size() > 0){
                crossPoint_y = (lineFitLeft[0]*lineFitRight[1] + lineFitLeft[1]*lineFitRight[0]) / (lineFitRight[0] - lineFitLeft[0]);
                int y1r = 0;
                int y2r = min(5*height/7, crossPoint_y);
                int y1l = 0;
                int y2l = min(5*height/7, crossPoint_y);

                int x1r = (y1r - lineFitRight[1]) / lineFitRight[0];
                int x2r = (y2r - lineFitRight[1]) / lineFitRight[0];
                int x1l = (y1l - lineFitLeft[1]) / lineFitLeft[0];
                int x2l = (y2l - lineFitLeft[1]) / lineFitLeft[0];

                cout << x1r << " " << x2r << " " << x1l << " " << x2l << endl;
                line(srcImage, Point(x1r,y1r),Point(x2r,y2r), Scalar(0,0,255), 5, CV_AA);
                line(srcImage, Point(x1l,y1l),Point(x2l,y2l), Scalar(0,0,255), 5, CV_AA);
            }
        }
    }

	//显示图像
    imshow("before",srcImage);
	imshow("after",cutImage);
}

//最小二乘法拟合直线
vector<double> line_fit(vector<Vec4i>& h)
{
    vector<double> res(2, 0);
    int n = 2*h.size();
    double k;                       //目标直线斜率
    double b;                       //目标直线截距
    vector<Vec4i>::iterator it = h.begin();
    double sumx=0,sumy=0,sumxy=0,sumxsq=0;
    while(it != h.end())
    {
        Vec4i L = *it;
        sumx += L[0];
        sumx += L[2];
        sumy += L[1];
        sumy += L[3];
        sumxy += L[0] * L[1];
        sumxy += L[2] * L[3];
        sumxsq += L[0] * L[0];
        sumxsq += L[2] * L[2];
        it++;
    }

    if(sumxsq == (sumx*sumx/n))
        k = atan2(0,1.0);
    else
        k = (sumxy-((sumx*sumy)/n))/(sumxsq-(sumx*sumx/n));
    b = (sumy-k*sumx)/n;
    res[0] = k;
    res[1] = b;
    return res;
}

vector<int> kmeans(vector<int> &sample,int k)
{
    vector<int> twoSideK(2, INT_MIN);
    int len = sample.size();
    vector<int> meanValue(k,0);
    //初始化均值
    if(len == 2) {
        twoSideK[0] = sample[0];
        twoSideK[1] = sample[1];
        return twoSideK;
    }
    meanValue[0] = sample[0];
    int i = 1;
    int n = sample.size();
    for( ;i < n; ++i){
        if(sample[i] != meanValue[0])
            meanValue[1] = sample[i];
    }
    if(i == n-1 && sample[0] == sample[n-1])
        return twoSideK;
    int cnt = 100;
    while(cnt > 0)
    {
        vector<vector<int> > C(k,vector<int>(k,0));  //用于存储类别
        vector<int> numC(k,0);
        //计算每个样本与各个均值的距离
        for(int i = 0; i < len;++i)
        {
            int minDis = abs(sample[i] - meanValue[0]);
            int minDisIndex = 0;
            for(int j = 1; j < k; ++j)
            {
                int dis = abs(sample[i] - meanValue[j]);
                if(dis < minDis)
                {
                    minDis = dis;
                    minDisIndex = j;
                }
            }
            //每个样本属于哪个类
            C[minDisIndex][numC[minDisIndex]] =  sample[i];
            numC[minDisIndex]++;
        }
        //均值更新
        int ifBreak = 0;
        for(int i = 0; i < k;++i)
        {
            int Sum = 0;
            for(int j = 0; j < numC[i];++j)
            {
                Sum += C[i][j];
            }
            int lastMeanValue = meanValue[i];
            meanValue[i] = Sum/numC[i];
            if(abs(lastMeanValue - meanValue[i]) <= 1)  ifBreak++;
        }
        cnt -= 1;
        //判断均值是否被更新
        if(ifBreak == k || cnt == 0)  
        {
            for(int i = 0;i < k; ++i)
            {
                twoSideK[i] =  meanValue[i];
            }
            return twoSideK;
        }
        
    }
}


// // Scharr滤波器
// static void on_HoughLines(int, void*)
// {
// 	//定义局部变量储存全局变量
// 	// Mat img_=img.clone();
//     Mat srcImage=g_srcImage.clone();
// 	Mat midImage=g_midImage.clone();
// 	//调用HoughLinesP函数
// 	vector<Vec4i> mylines;

// 	// Scharr滤波器
//     Mat grad_x, abs_grad_x;
//     Scharr(midImage, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
//     convertScaleAbs(grad_x, abs_grad_x);

// 	Canny( abs_grad_x, abs_grad_x, g_cannyLowThreshold, g_cannyLowThreshold*3, 3);

// 	HoughLinesP(abs_grad_x, mylines, 1, CV_PI/180, g_nthreshold+1, 50, g_merge+1);
    
// 	//循环遍历绘制每一条线段
// 	for( size_t i = 0; i < mylines.size(); i++ )
// 	{
// 		Vec4i l = mylines[i];
// 		line( srcImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 5, CV_AA);
// 	}
// 	//显示图像
//     imshow("before",srcImage);
// 	imshow("after",abs_grad_x);
// }