/*
  对avi视频进行实时腐蚀膨胀参数调整
*/
#include <opencv2/opencv.hpp>

#include <math.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

/// 全局变量

Mat src, src_gray;
Mat dst, detected_edges;

char* window_name = "Edge Map";

int g_nStructElementSize = 12;
int g_nTrackbarNumer = 2;
void Process();//膨胀和腐蚀的处理函数
void on_TrackbarNumChange(int, void *);//回调函数
void on_ElementSizeChange(int, void *);//回调函数

int main(int argc, char** argv){
    VideoCapture capture(argv[1]);
    if(!capture.isOpened()){
        cout << "video not open" << endl;
        return 1;
    }
    int delay =200;

    while(true){
        Mat frame;
        capture >> frame;

        src = frame;

        Mat l_midImage;

        /// 创建显示窗口
        namedWindow( window_name, 0);
        cvResizeWindow(window_name, 1500, 1000);

        Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));
        if(g_nTrackbarNumer== 0) {   
                erode(src,src, element);
        }
        else if(g_nTrackbarNumer== 1){
                dilate(src,src, element);
        }
        else if(g_nTrackbarNumer== 2){
                morphologyEx(src,src, MORPH_OPEN, element);
        }
        else if(g_nTrackbarNumer== 3){
                morphologyEx(src,src, MORPH_CLOSE, element);
        }
        else if(g_nTrackbarNumer== 4){
                morphologyEx(src,src, MORPH_GRADIENT, element);
        }
        else if(g_nTrackbarNumer== 5){
                morphologyEx(src,src, MORPH_TOPHAT, element);
        }
        else {
                morphologyEx(src,src, MORPH_BLACKHAT, element);
        }
        Mat element1 = getStructuringElement(MORPH_RECT, Size(3,3));
        morphologyEx(src, src, MORPH_GRADIENT, element1);

        imshow(window_name, frame);

        createTrackbar("Erotion/Dilation",window_name, &g_nTrackbarNumer, 6, on_TrackbarNumChange);
        createTrackbar("kernel_size", window_name,&g_nStructElementSize, 21, on_ElementSizeChange);

        if(delay>=0&&waitKey (delay)>=32)
            waitKey(0);
    }
}
void Process()
{
       //获取自定义核
       Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));
 
       //进行腐蚀或膨胀操作
       if(g_nTrackbarNumer== 0) {   
              erode(src,src, element);
       }
       else if(g_nTrackbarNumer== 1){
              dilate(src,src, element);
       }
       else if(g_nTrackbarNumer== 2){
              morphologyEx(src,src, MORPH_OPEN, element);
       }
       else if(g_nTrackbarNumer== 3){
              morphologyEx(src,src, MORPH_CLOSE, element);
       }
       else if(g_nTrackbarNumer== 4){
              morphologyEx(src,src, MORPH_GRADIENT, element);
       }
       else if(g_nTrackbarNumer== 5){
              morphologyEx(src,src, MORPH_TOPHAT, element);
       }
       else {
              morphologyEx(src,src, MORPH_BLACKHAT, element);
       }

       Mat element1 = getStructuringElement(MORPH_RECT, Size(3,3));
       morphologyEx(src, src, MORPH_GRADIENT, element1);
       //显示效果图
       imshow(window_name, src);
}
 
 
//-----------------------------【on_TrackbarNumChange( )函数】------------------------------------
//            描述：腐蚀和膨胀之间切换开关的回调函数
//-----------------------------------------------------------------------------------------------------
void on_TrackbarNumChange(int, void *)
{
       //腐蚀和膨胀之间效果已经切换，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
       Process();
}
 
 
//-----------------------------【on_ElementSizeChange( )函数】-------------------------------------
//            描述：腐蚀和膨胀操作内核改变时的回调函数
//-----------------------------------------------------------------------------------------------------
void on_ElementSizeChange(int, void *)
{
       //内核尺寸已改变，回调函数体内需调用一次Process函数，使改变后的效果立即生效并显示出来
       Process();
}
