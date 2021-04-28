#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;

// const int height = 567, width = 1008;
int height, width;
// cutImage是梯形感兴趣区域的图片
Mat srcImage, midImage, grayImage, cutImage, cutImage2, dst;

// GW0426 left : close 8
int g_nStructElementSize = 4; //形态学闭运算结构元素(内核矩阵)的尺寸
int g_nthreshold=30;
int g_merge = 50;
int g_cannyLowThreshold= 40;
int b_value = 100;
int c_p = 20;

//H、S通道
int channels[] = { 0, 1 };
int histSize[] = { 30, 32 };
float HRanges[] = { 0, 180 };   
float SRanges[] = { 0, 256 };
const float *ranges[] = { HRanges, SRanges };

vector<Point> CalROI(Mat dst,Mat src);
double compHist(const MatND srcHist,Mat compareImage);
vector<Point> LineDetector( Mat srcImage, Mat dst);
vector<double> line_fit(vector<Vec4i>& h);
vector<int> kmeans(vector<int>& sample,int k);

// input  : test_image and dst_l2.image
// output : detection of the power line 
int main(int argc, char** argv){

    srcImage = imread(argv[1]);
    dst = imread(argv[2]);
    // resize(src, srcImage, srcSize, 0, 0, INTER_LINEAR);
    imshow("Original", srcImage);
    imshow("dst", dst);

    height = srcImage.rows;
    width  = srcImage.cols;

    vector<Point> lineOutput(4);
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    lineOutput = LineDetector(srcImage,dst);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract power line cost = " << time_used.count() << " seconds. " << endl;
    
    line(srcImage, Point(lineOutput[0].x,lineOutput[0].y),Point(lineOutput[1].x,lineOutput[1].y), Scalar(0,0,255), 2, CV_AA);
    line(srcImage, Point(lineOutput[2].x,lineOutput[2].y),Point(lineOutput[3].x,lineOutput[3].y), Scalar(0,0,255), 2, CV_AA);
    
    imshow("result", srcImage);
    waitKey(0);
    return 0;
}

vector<Point> LineDetector(Mat srcImage, Mat dst){
    Mat draw = srcImage.clone();

    vector<Point> twoLines(4);
    int x1u, x1d, x2u, x2d;
    int y1u, y1d, y2u, y2d;

	cvtColor(srcImage, srcImage, CV_BGR2GRAY);
    GaussianBlur(srcImage, srcImage, Size(61, 61), 2, 2);
    // 直方图均衡化
    equalizeHist(srcImage, srcImage);

    midImage = Mat::zeros(srcImage.size(), srcImage.type());

    for (int y = 0; y < srcImage.rows; y++)
	{
		for (int x = 0; x < srcImage.cols; x++)
		{
            double t = ((srcImage.at<uchar>(y, x) - 127) / 225.00)*c_p*0.1;
			midImage.at<uchar>(y, x) = saturate_cast<uchar>(srcImage.at<uchar>(y, x) * 
                    ((1.00 / (1.00 + exp(-t))) + 0.3) + b_value - 100);
		}
	}
    // 进行形态学close操作
    Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),
        Point( g_nStructElementSize, g_nStructElementSize ));
    morphologyEx(midImage, midImage, MORPH_CLOSE, element);
    imshow("morph_close", midImage);

	//调用HoughLinesP函数
	vector<Vec4i> mylines;
    vector<Vec4i> mylines_up;
    vector<Vec4i> mylines_down;

    // Laplacian( midImage, midImage, CV_8U, 3, 1, 0, BORDER_DEFAULT );
    // convertScaleAbs(midImage, midImage);
    // imshow("laplacian", midImage);

    // Mat element1 = getStructuringElement(MORPH_RECT, Size(3,3),Point(1,1));
    // morphologyEx(midImage, midImage, MORPH_GRADIENT, element1);
    // imshow("morph_gradient", midImage);
    // morphologyEx(midImage, midImage, MORPH_ERODE, element1);
    // imshow("morph_erode", midImage);

    GaussianBlur( midImage, midImage, Size(15, 15), 2, 2);

    Canny(midImage, midImage, g_cannyLowThreshold, g_cannyLowThreshold*3, 3);
    imshow("canny", midImage);

    vector<Point> ROIpoints;
    ROIpoints = CalROI(dst, draw);


    // 掩膜
    Mat mask = Mat::zeros(srcImage.size(), CV_8UC1);
    Point PointArray[4];

    // GW
    // if(ROIpoints[1].x != 0 && ROIpoints[1].y != 0){
    //     PointArray[0] = Point(max(8*width/10,ROIpoints[0].x), max(1*height/10, ROIpoints[0].y - 10));
    //     PointArray[1] = Point(width,  max(1*height/10, ROIpoints[0].y - 10));
    //     PointArray[2] = Point(width,  min(2*height/5 ,ROIpoints[1].y + 10));
    //     PointArray[3] = Point(max(8*width/10,ROIpoints[0].x), min(2*height/5 ,ROIpoints[1].y + 10));
    // }

    // GW0426left
    if(ROIpoints[1].x != 0 && ROIpoints[1].y != 0){
        cout << "ROI found!" << endl;
        PointArray[0] = Point(0, ROIpoints[0].y - 40);
        PointArray[1] = Point(width/3,  ROIpoints[0].y - 10);
        PointArray[2] = Point(width/3,  ROIpoints[1].y + 15);
        PointArray[3] = Point(0, ROIpoints[1].y + 15);
    }
    else{
        PointArray[0] = Point(0, 2*height/10);
        PointArray[1] = Point(width/2, 2*height/10);
        PointArray[2] = Point(width/2, 5*height/10);
        PointArray[3] = Point(0,5*height/10);
    }

    fillConvexPoly(mask,PointArray,4,Scalar(255));
    bitwise_and(mask,midImage,cutImage);

    imshow("cutImage", cutImage);

	HoughLinesP(cutImage, mylines, 1, CV_PI/180, g_nthreshold+1, 50, g_merge+1);
    vector<int> slopeset; //提取的直线到斜率集合

    cout << "num of lines : " << mylines.size() << endl;

    int cnt = 0;
    cout << "slopes : ";
    for(auto it = mylines.begin(); it != mylines.end(); ++it){
        Vec4i L = *it;
        double k = (L[3]-L[1]) * 1.0 /(L[2]-L[0]);
        // cout << k << " " ;
        if(abs(k) < 0.3){
            line(draw, Point(L(0), L(1)), Point(L(2), L(3)), Scalar(0,0,255), 5, CV_AA);
            cnt += 1;
        }
    }
    cout << endl;
    cout << "num of the res lines : " << cnt << endl;
    imshow("multi_line_result", draw);
    
	//循环遍历绘制每一条线段,斜率小于10的直接删除，并区分左右直线
	for(auto it = mylines.begin(); it != mylines.end(); ++it){
		Vec4i L = *it;
		double k;
        if(abs(L[2] - L[0]) < 1)
            k = 150; //设置垂直线的斜率为150
		else 
            k = (L[3]-L[1]) * 1.0/(L[2]-L[0]);
		if(abs(k) > 0.3){
			it = mylines.erase(it);
            --it;
            continue;
		}
        int k_ = k * 1000;
        cout << k_ << " ";
		slopeset.push_back(k_);
	}

    cout << "slopeset num : " << slopeset.size() << endl;

    if(slopeset.size() > 1){
        vector<int> slopes = kmeans(slopeset, 2);
        if(slopes[0] != INT_MIN) {
            // cout << "slope of two sides : " << slopes[0] << " " << slopes[1] << endl;
            float slopeThresh = (slopes[0] + slopes[1]) / 2;

            for(auto it = mylines.begin(); it != mylines.end(); ++it){
                Vec4i L = *it;
                double k;
                if(abs(L[2] - L[0]) < 1){
                    k = 150;
                }
                else 
                    k = (L[3]-L[1]) * 1.0 /(L[2]-L[0]);

                k *= 1000;

                if(k < slopeThresh){
                    mylines_down.push_back(L);
                }
                else{
                    mylines_up.push_back(L);
                }
            }

            cout << mylines_up.size() << "  " << mylines_down.size() << endl;
            vector<double> lineFitUp = line_fit(mylines_up);
            vector<double> lineFitDown = line_fit(mylines_down);

            int crossPoint_x;
            if(mylines_up.size() > 0 && mylines_down.size() > 0){
                crossPoint_x = (lineFitUp[1] - lineFitDown[1]) / (lineFitDown[0] - lineFitUp[0]);
                x1u = 0;
                x1d = 0;
                x2u = min(3*width/7, crossPoint_x);
                x2d = min(3*width/7, crossPoint_x);

                y1u = lineFitUp[0] * x1u + lineFitUp[1];
                y1d = lineFitDown[0] * x1d + lineFitDown[1];
                y2u = lineFitUp[0] * x2u + lineFitUp[1];
                y2d = lineFitDown[0] * x2d + lineFitDown[1];
            }
        }
    }
    twoLines[0].x = x1d;
    twoLines[0].y = y1d;
    twoLines[1].x = x2d;
    twoLines[1].y = y2d;
    twoLines[2].x = x1u;
    twoLines[2].y = y1u;
    twoLines[3].x = x2u;
    twoLines[3].y = y2u;

    return twoLines;
}

void sharpenImage1(const cv::Mat &image, cv::Mat &result)
{
     //创建并初始化滤波模板
     cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
     kernel.at<float>(1,1) = 5.0;
     kernel.at<float>(0,1) = -1.0;
     kernel.at<float>(1,0) = -1.0;
     kernel.at<float>(1,2) = -1.0;
     kernel.at<float>(2,1) = -1.0;
 
     result.create(image.size(),image.type());
     
     //对图像进行滤波
     cv::filter2D(image,result,image.depth(),kernel);
 }

vector<Point> CalROI(Mat dst,Mat src){
    vector<Point> res(2);

    int dst_height = dst.rows;
    int dst_width  = dst.cols;

    cvtColor(dst, dst, CV_BGR2HSV);
	//采用H-S直方图进行处理
	//首先得配置直方图的参数
	MatND srcHist;
	//进行原图直方图的计算
	calcHist(&dst, 1, channels, Mat(), srcHist, 2, histSize, ranges, true, false);
	//归一化
	normalize(srcHist, srcHist, 0, 1, NORM_MINMAX);

    // 定义图片中到搜索起始坐标
    int X1, X2, Y1, Y2;
    // GW
    // X1 = 3*width/5;
    // Y1 = height/10;
    // X2 = width;
    // Y2 = 4*height/10;

    // GW0426left
    X1 = 0;
    Y1 = 2* height/10;
    X2 = width/3;
    Y2 = 5 * height/10;

    Point preStart;
    Point preEnd;

	Point get1(0, 0);
	Point get2(0, 0);

    //直方图对比的相似值
    double comnum = 1.0;
    //初始化搜索区域
    for (int Cy = Y1; Cy <= Y2; Cy += 1) {
        for (preStart.x = X1, preStart.y = Cy; preStart.x <= X2; preStart.x += 1) {

            if ((preStart.x + dst_width) < X2)
                preEnd.x = preStart.x + dst_width;
            else
                preEnd.x = X2 - 1;
            if ((preStart.y + dst_height) < Y2)
                preEnd.y = preStart.y + dst_height;
            else
                preEnd.y = Y2 - 1;

            Mat compareImg;
            compareImg = src(Rect(preStart, preEnd));
            double c = compHist(srcHist, compareImg);
            if (comnum > c) {
                get1 = preStart;
                get2 = preEnd;
                comnum = c;
            }
        }
    }
    cout << "comnum : " << comnum << endl;

    res[0] = get1;
    res[1] = get2;

    if(comnum>0.6){
        Point p(0,0);
        res[1] = p;
    }
    return res;
}

double compHist(const MatND srcHist,Mat compareImage)
{
	//在比较直方图时，最佳操作是在HSV空间中操作，所以需要将BGR空间转换为HSV空间
	Mat compareHsvImage;
	cvtColor(compareImage, compareHsvImage, CV_BGR2HSV);
	//采用H-S直方图进行处理
	//首先得配置直方图的参数
	MatND  compHist;
	//进行原图直方图的计算
	
	//对需要比较的图进行直方图的计算
	calcHist(&compareHsvImage, 1, channels, Mat(), compHist, 2, histSize, ranges, true, false);
	//注意：这里需要对两个直方图进行归一化操作
	
	normalize(compHist, compHist, 0, 1, NORM_MINMAX);
	//对得到的直方图对比
	double g_dCompareRecult = compareHist(srcHist, compHist, 3);//3表示采用巴氏距离进行两个直方图的比较
	return g_dCompareRecult;
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
    // if(i == n && sample[0] == sample[n-1]){
    //     return twoSideK;
    // }
    while(1)
    {
        vector<vector<int> > C(k,vector<int>(n,0));  //用于存储类别
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
            if(lastMeanValue == meanValue[i] || lastMeanValue == meanValue[i] -1 || lastMeanValue == meanValue[i] +1) 
                ifBreak++;
        }
        //判断均值是否被更新
        if(ifBreak == k) 
        {
            for(int i = 0;i < k; ++i)
            {
                twoSideK[i] =  meanValue[i];
            }
            return twoSideK;
        }
        
    }
}