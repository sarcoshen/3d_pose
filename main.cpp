#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "keeppose.h"

using namespace std;
using namespace cv;

// 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
int posePairs[26] = {0,1,1,2,2,3,3,4,1,5,5,6,6,7,1,8,1,11,8,9,9,10,11,12,12,13};

// opencv color
Scalar colorLeftLeg = Scalar(255,255,255);
Scalar colorLeftThigh = Scalar(255,255,0);
Scalar colorRightLeg = Scalar(0,0,255);
Scalar colorRightThigh = Scalar(0,255,255);
Scalar colorElse = Scalar(255,0,0);

const int Height1 = Height;
const int Width1 = Width;

int minmax(float a, int m)
{
    int c = a;
    if (c < 1)
    {
        return 1;
    }
    if (c > m - 2)
    {
        return m - 2;
    }
    return c;
}


int display3d(Mat pano, Mat src, vector<float> pts3d)
{
    if (65 > pts3d.size())
    {
         return 0;
    }

    // x, [-2,2]
    // y, [-1.5,1.5]
    // z, [0,4]
    float scaleX = float(Width1-1) / 4.0;
    float scaleY = float(Height1-1) / 3.0;
    float scaleZ = float(Width1-1) / 4.0;
    Mat src1(Height1,Width1,CV_8UC3,cv::Scalar(0,0,0));
    Mat src2(Height1,Width1,CV_8UC3,cv::Scalar(0,0,0));
 
    for (int i=0; i<13; i++)
    {
         // 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
         // int posePairs[26] = {0,1,1,2,2,3,3,4,1,5,5,6,6,7,1,8,1,11,8,9,9,10,11,12,12,13};
         int idx1 = posePairs[2*i];
         int idx2 = posePairs[2*i+1];
         float x1 = pts3d[idx1*4+3];
         float y1 = pts3d[idx1*4+4];
         float z1 = pts3d[idx1*4+5];
         float x2 = pts3d[idx2*4+3];
         float y2 = pts3d[idx2*4+4];
         float z2 = pts3d[idx2*4+5];
         Scalar colordraw = colorElse;
         if (12==idx1 && 13==idx2)
         {
              colordraw = colorLeftLeg;
         }
         if (11==idx1 && 12==idx2)
         {
              colordraw = colorLeftThigh;
         }
         if (9==idx1 && 10==idx2)
         {
              colordraw = colorRightLeg;
         }
         if (8==idx1 && 9==idx2)
         {
              colordraw = colorRightThigh;
         }
         int col1 = minmax((x1 - (-2.0)) * scaleX, Width1-1);
         int row1 = Height1 - 1 - minmax((y1 - (-1.5)) * scaleY, Height1-1); 
         int col2 = minmax((x2 - (-2.0)) * scaleX, Width1-1);
         int row2 = Height1 - 1 - minmax((y2 - (-1.5)) * scaleY, Height1-1);
         line(src1,Point(col1,row1),Point(col2,row2),colordraw,3,8,0);
         col1 = minmax((z1 - 0) * scaleZ, Width1-1);
         row1 = Height1 - 1 - minmax((y1 - (-1.5)) * scaleY, Height1-1);
         col2 = minmax((z2 - 0) * scaleZ, Width1-1);
         row2 = Height1 - 1 - minmax((y2 - (-1.5)) * scaleY, Height1-1);
         line(src2,Point(col1,row1),Point(col2,row2),colordraw,3,8,0);
    }
    
    Rect rect_roi_1 = Rect(Width1,0,Width1,Height1);
    src1.copyTo(pano(rect_roi_1));
    Rect rect_roi_2 = Rect(Width1,Height1,Width1,Height1);
    src2.copyTo(pano(rect_roi_2));
    
    return 1;
}


int display2d(Mat pano, Mat src, vector<float> pts2d, vector<int> pose_pairs)
{
    //cout << "size: " << pts2d.size() << "," << pose_pairs.size() << endl;   
    for (int i=0; i<13; i++)
    {
         int x = pts2d[3*i];
         int y = pts2d[3*i+1];
         float pro = pts2d[3*i+2];
         //if (pro < pointThd || x < 2 || x > Width-2)
         //{
         //    return 0;
         //}
    }
    

    //cout << "shen" << endl;
    for (int i=0; i<13; i++)
    {
         
         int idx1 = pose_pairs[2*i];
         int idx2 = pose_pairs[2*i+1];
         if (idx1 < 0 || idx2 < 0)
         {
              continue;
         }
         float x1 = pts2d[idx1*3];
         float y1 = pts2d[idx1*3+1];
         float x2 = pts2d[idx2*3];
         float y2 = pts2d[idx2*3+1];         
         Scalar colordraw = colorElse;
         if (12==idx1 && 13==idx2)
         {
              colordraw = colorLeftLeg;
         }
         if (11==idx1 && 12==idx2)
         {
              colordraw = colorLeftThigh;
         }
         if (9==idx1 && 10==idx2)
         {
              colordraw = colorRightLeg;
         }
         if (8==idx1 && 9==idx2)
         {
              colordraw = colorRightThigh;
         }
         line(src,Point(int(x1),int(y1)),Point(int(x2),int(y2)),colordraw,3,8,0);
    }
    
    Rect rect_roi = Rect(0,0,Width1,Height1);
    src.copyTo(pano(rect_roi));
    return 1;
}



/*
int main()
{
    // initial model and camera
    KeepPose *keeppose = new KeepPose();
    cout << "init camera success !" << endl;
    int success = keeppose->initialize();
    cout << "init posemodel success !" << endl;
    int iFrame = 0;
    
    while(true)
    {
         if (keeppose->get_frame())
         {
             
             cout << "IFrame: " << iFrame << endl;            
             Mat src(Height,Width,CV_8UC3,cv::Scalar(0,0,0));
             src.data = keeppose->frameData;
             Mat pano;
             resize(src,pano,Size(Width,Height),2.0,2.0,INTER_LINEAR);
             
             if (keeppose->get_2d_pose())
             {
                 for (int i=0; i<13; i++)
                 {
                      int idx1 = posePairs[2*i];
                      int idx2 = posePairs[2*i+1];
                      float x1 = keeppose->keypoints2d[idx1*3];
                      float y1 = keeppose->keypoints2d[idx1*3+1];
                      float x2 = keeppose->keypoints2d[idx2*3];
                      float y2 = keeppose->keypoints2d[idx2*3+1];
                      float pro1 = keeppose->keypoints2d[idx1*3+2];
                      float pro2 = keeppose->keypoints2d[idx2*3+2];
                      if (pro1 > pointThd && pro2 > pointThd)
                      {
                          line(pano,Point(int(x1),int(y1)),Point(int(x2),int(y2)),Scalar(0,0,255),3,8,0);
                      }
                 }
             }

             //char saveBuf[256];
             //sprintf(saveBuf,"%s%d%s","./images_dst/",iFrame,".jpg");
             //imwrite(saveBuf,pano);
             
             imshow("keeppose",pano);
             waitKey(1);
             
         }
         iFrame++;
         
    }

    delete keeppose;
    keeppose = NULL;
    return 0;
}
*/




int main(int argc, char *argv[])
{
    // initial model and camera
    KeepPose *keeppose = new KeepPose();
    cout << "init camera success !" << endl;
    int success = keeppose->initialize();
    cout << "init posemodel success !" << endl;
    int iFrame = 0;
    /*
    cv::VideoWriter avi;
    cv::Size S = cv::Size(2*Width, 2*Height);
    bool flag = avi.open("./rst.avi", CV_FOURCC('M','J','P', 'G'), 25, S, true);
    if (flag == false) {
        cout << " open log avi failed." <<endl;
    }
    */
    Size S = cv::Size(2*Width, 2*Height);
    VideoWriter avi("./rst.avi", CV_FOURCC('M','J','P', 'G'), 25.0, S);
    while(true)
    {
         if (keeppose->get_frame())
         {
             //cout << "IFrame: " << keeppose->colorTimestamp << endl;
             
             Mat pano(2*Height,2*Width,CV_8UC3,cv::Scalar(0,0,0));
             Mat src1(Height,Width,CV_8UC3,cv::Scalar(0,0,0));
             src1.data = keeppose->frameData;
             Mat src;
             resize(src1,src,Size(Width1,Height1),0.5,0.5,INTER_LINEAR);
             if (keeppose->get_2d_pose())
             {
                 //cout << "xiao" << endl;
                 display2d(pano,src,keeppose->keypoints2d_for_display, keeppose->pose_pair_list);
                 
                 if (keeppose->get_3d_pose())
                 {
                      display3d(pano,src,keeppose->keypoints3d);
                      imshow("keeppose",pano);
                      waitKey(1);
                 }
                 else
                 {
                      imshow("keeppose",pano);
                      waitKey(1);
                 }
                 
                 imshow("keeppose",pano);
                 waitKey(1);
             }
             else
             {
                 imshow("keeppose",pano);
                 waitKey(1);
             }
 
             avi << pano;
         }
          
         iFrame++;
         if (iFrame > 400) break;
    }
    avi.release();
    delete keeppose;
    keeppose = NULL;
    return 0;
}





