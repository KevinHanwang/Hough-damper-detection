#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

const int height = 1084, width = 813;
// cutImage是梯形感兴趣区域的图片
Mat srcImage, midImage, grayImage, cutImage, cutImage2;
int g_nStructElementSize = 12; //形态学闭运算结构元素(内核矩阵)的尺寸
int g_nthreshold=120;
int g_merge = 50;
int g_cannyLowThreshold=10;

vector<Point> LineDetector(Mat src, Mat srcImage,Size srcSize);
vector<double> line_fit(vector<Vec4i>& h);
vector<int> kmeans(vector<int>& sample,int k);

int main(int argc, char** argv){
    // 载入原始图,原始图像太大，所以对原始图像进行分辨率重设 
	Mat src = imread(argv[1]);
    Size srcSize = Size(width, height);  //填入任意指定尺寸
    resize(src, srcImage, srcSize, 0, 0, INTER_LINEAR);
    imshow("Original", srcImage);

    vector<Point> lineOutput(4); 
    lineOutput = LineDetector(src, srcImage, srcSize);
    
    cout << lineOutput[0].x << " " << lineOutput[1].x << " " << lineOutput[2].x << " " << lineOutput[3].x << endl; 
    line(srcImage, Point(lineOutput[0].x,lineOutput[0].y),Point(lineOutput[1].x,lineOutput[1].y), Scalar(0,0,255), 5, CV_AA);
    line(srcImage, Point(lineOutput[2].x,lineOutput[2].y),Point(lineOutput[3].x,lineOutput[3].y), Scalar(0,0,255), 5, CV_AA);
    
    imshow("result", srcImage);
    waitKey(0);
    return 0;
}

vector<Point> LineDetector(Mat src, Mat srcImage, Size srcSize){
    vector<Point> twoLines(4);
    int x1l, x1r, x2l, x2r;
    int y1l, y1r, y2l, y2r;

	cvtColor(src, src, CV_BGR2GRAY);
    GaussianBlur(src, src, Size(61, 61), 2, 2);
    // 直方图均衡化
    equalizeHist(src, src);
    // 进行形态学闭运算操作
    Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),
        Point( g_nStructElementSize, g_nStructElementSize ));
    morphologyEx(src, midImage, MORPH_CLOSE, element);
    resize(midImage, midImage, srcSize, 0, 0, INTER_LINEAR);

	//调用HoughLinesP函数
	vector<Vec4i> mylines;
    vector<Vec4i> mylines_left;
    vector<Vec4i> mylines_right;

    GaussianBlur( midImage, midImage, Size(15, 15), 2, 2);
    grayImage = midImage.clone();

	Canny(midImage, midImage, g_cannyLowThreshold, g_cannyLowThreshold*3, 3);

    // 掩膜
    Mat mask = Mat::zeros(srcImage.size(), CV_8UC1);
    Point PointArray[4];
    PointArray[0] = Point(width/2 -100, 0);
    PointArray[1] = Point(width/2 +100, 0);
    PointArray[2] = Point(width/2 + 50 ,5*height/7);
    PointArray[3] = Point(width/2 - 50 ,5*height/7);
    fillConvexPoly(mask,PointArray,4,Scalar(255));
    bitwise_and(mask,midImage,cutImage);

	HoughLinesP(cutImage, mylines, 1, CV_PI/180, g_nthreshold+1, 50, g_merge+1);
    vector<int> slopeset; //提取的直线到斜率集合
    
	//循环遍历绘制每一条线段,斜率小于10的直接删除，并区分左右直线
	for(auto it = mylines.begin(); it != mylines.end(); ++it){
		Vec4i L = *it;
		int k;
        if(abs(L[2] - L[0]) < 1)
            k = 150; //设置垂直线的斜率为150
		else 
            k = (L[3]-L[1])/(L[2]-L[0]);
		if(abs(k) < 10){
			it = mylines.erase(it);
            --it;
            continue;
		}
		slopeset.push_back(k);
	}

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
                if(flag){
                    // 斜率不同符号，大斜率在左边
                    if(k < slopeThresh){
                        mylines_left.push_back(L);
                    }
                    else{
                        mylines_right.push_back(L);
                    }
                }
                else{
                    // 斜率同符号，大斜率在右边
                    if(k > slopeThresh){
                        mylines_left.push_back(L);
                    }
                    else{
                        mylines_right.push_back(L);
                    }
                }
            }

            cout << mylines_left.size() << "  " << mylines_right.size() << endl;
            vector<double> lineFitLeft = line_fit(mylines_left);
            vector<double> lineFitRight = line_fit(mylines_right);

            int crossPoint_y;
            if(mylines_left.size() > 0 && mylines_right.size() > 0){
                crossPoint_y = (lineFitLeft[0]*lineFitRight[1] + lineFitLeft[1]*lineFitRight[0]) / (lineFitRight[0] - lineFitLeft[0]);
                y1r = 0;
                y2r = min(4*height/7, crossPoint_y);
                y1l = 0;
                y2l = min(4*height/7, crossPoint_y);

                x1r = (y1r - lineFitRight[1]) / lineFitRight[0];
                x2r = (y2r - lineFitRight[1]) / lineFitRight[0];
                x1l = (y1l - lineFitLeft[1]) / lineFitLeft[0];
                x2l = (y2l - lineFitLeft[1]) / lineFitLeft[0];
            }
        }
    }
    twoLines[0].x = x1r;
    twoLines[0].y = y1r;
    twoLines[1].x = x2r;
    twoLines[1].y = y2r;
    twoLines[2].x = x1l;
    twoLines[2].y = y1l;
    twoLines[3].x = x2l;
    twoLines[3].y = y2l;

    return twoLines;
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
    while(1)
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
            if(lastMeanValue == meanValue[i])  ifBreak++;
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