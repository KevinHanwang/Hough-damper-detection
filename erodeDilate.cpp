#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
 

Mat g_srcImage, g_dstImage;//原始图和效果图
int g_nTrackbarNumer = 0;//0表示腐蚀erode, 1表示膨胀dilate
int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸
 
void Process();//膨胀和腐蚀的处理函数
void on_TrackbarNumChange(int, void *);//回调函数
void on_ElementSizeChange(int, void *);//回调函数
 

int main(int argc, char** argv)
{
       //改变console字体颜色
       system("color5E"); 
 
       //载入原图
       g_srcImage= imread(argv[1]);
       if(!g_srcImage.data ) { printf("Oh，no，读取srcImage错误~！\n"); return false; }
      
       //显示原始图
       namedWindow("Original");
       cvResizeWindow("Original", 1800, 1200);
       imshow("Original", g_srcImage);
      
       //进行初次腐蚀操作并显示效果图
       namedWindow("after1",0);
       cvResizeWindow("after1", 1800, 1200);
       //获取自定义核
       Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));
       erode(g_srcImage,g_dstImage, element);
       imshow("after1", g_dstImage);
 
       //创建轨迹条
       createTrackbar("Erotion/Dilation", "after1", &g_nTrackbarNumer, 6, on_TrackbarNumChange);
       createTrackbar("kernel_size", "after1",&g_nStructElementSize, 21, on_ElementSizeChange);
 
       //输出一些帮助信息
       cout<<endl<<"\t嗯。运行成功，请调整滚动条观察图像效果~\n\n"
              <<"\t按下“q”键时，程序退出~!\n";
 
       //轮询获取按键信息，若下q键，程序退出
       while(char(waitKey(1))!= 'q') {}
 
       return 0;
}
 
//-----------------------------【Process( )函数】------------------------------------
//            描述：进行自定义的腐蚀和膨胀操作
//-----------------------------------------------------------------------------------------
void Process()
{
       //获取自定义核
       Mat element = getStructuringElement(MORPH_RECT, Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),Point( g_nStructElementSize, g_nStructElementSize ));
 
       //进行腐蚀或膨胀操作
       if(g_nTrackbarNumer== 0) {   
              erode(g_srcImage,g_dstImage, element);
       }
       else if(g_nTrackbarNumer== 1){
              dilate(g_srcImage,g_dstImage, element);
       }
       else if(g_nTrackbarNumer== 2){
              morphologyEx(g_srcImage,g_dstImage, MORPH_OPEN, element);
       }
       else if(g_nTrackbarNumer== 3){
              morphologyEx(g_srcImage,g_dstImage, MORPH_CLOSE, element);
       }
       else if(g_nTrackbarNumer== 4){
              morphologyEx(g_srcImage,g_dstImage, MORPH_GRADIENT, element);
       }
       else if(g_nTrackbarNumer== 5){
              morphologyEx(g_srcImage,g_dstImage, MORPH_TOPHAT, element);
       }
       else {
              morphologyEx(g_srcImage,g_dstImage, MORPH_BLACKHAT, element);
       }

 
       //显示效果图
       imshow("after1", g_dstImage);
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