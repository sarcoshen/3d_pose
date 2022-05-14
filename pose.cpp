// 3rdparty dependencies
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include "keeppose.h"

using namespace std;
using namespace cv;

// Producer
DEFINE_int32(camera,                    0,             "The camera index for cv::VideoCapture. Integer in the range [0, 9]. Select a negative"
                                                        " number (by default), to auto-detect and open the first available camera.");
DEFINE_string(camera_resolution,        "640x480",     "Size of the camera frames to ask for.");
DEFINE_double(camera_fps,               30.0,           "Frame rate for the webcam (only used when saving video from webcam). Set this value to the"
                                                        " minimum value between the OpenPose displayed speed and the webcam real frame rate.");
DEFINE_uint64(frame_first,              0,              "Start on desired frame number. Indexes are 0-based, i.e. the first frame has index 0.");
DEFINE_uint64(frame_last,               -1,             "Finish on desired frame number. Select -1 to disable. Indexes are 0-based, e.g. if set to"
                                                        " 10, it will process 11 frames (0-10).");
DEFINE_bool(frame_flip,                 false,          "Flip/mirror each frame (e.g. for real time webcam demonstrations).");
DEFINE_int32(frame_rotate,              0,              "Rotate each frame, 4 possible values: 0, 90, 180, 270.");
DEFINE_bool(frames_repeat,              false,          "Repeat frames when finished.");

// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                                        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder,             "models/",      "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution,           "432x368",      "Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased,"
                                                        " the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect"
                                                        " ratio possible to the images or videos to be processed. E.g. the default `656x368` is"
                                                        " optimal for 16:9 videos, e.g. full HD (1980x1080) and HD (1280x720) videos.");
DEFINE_string(resolution,               "640x480",     "The image resolution (display and output). Use \"-1x-1\" to force the program to use the");
 
DEFINE_int32(num_gpu,                   -1,             "The number of GPU devices to use. If negative, it will use all the available GPUs in your machine.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");

    
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                                                        " If you want to change the initial scale, you actually want to multiply the"
                                                        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number,              1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending,           false,          "If blending is enabled, it will merge the results with the original frame. If disabled, it"
                                                        " will only display the results on a black background.");
DEFINE_double(render_threshold,         0.05,           "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                                        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                                        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                                        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                                                        " hide it. Only valid for GPU rendering.");

// outputSize
const auto outputSize = op::flagsToPoint(FLAGS_resolution, "640x480");

// netInputSize
const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "432x368");

// netOutputSize
const auto netOutputSize = netInputSize;

// poseModel
const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);

op::CvMatToOpInput cvMatToOpInput{netInputSize, FLAGS_scale_number, (float)FLAGS_scale_gap};

op::CvMatToOpOutput cvMatToOpOutput{outputSize};



// nearest point in human-mask
void nearest_mask_points(unsigned short *mask, int col, int row, int Width, int Height, int &col1, int &row1)
{
    col1 = col;
    row1 = row;
    float minDist = 1000000.0;
    for (int i = row - nearestR; i < row + nearestR; i++)
    {
        int y = max(float(2),min(float(i),float(Height-3)));
        for (int j = col - nearestR; j < col + nearestR; j++)
        {
            int x = max(float(2),min(float(i),float(Width-3)));
            int flags = 0;
            for (int ii=-2; ii<=2; ii++)
            {
                for (int jj=-2; jj<=2; jj++)
                {
                     if (!mask[(y+ii)*Width+x+jj])
                     {
                         flags++;
                     }
                }
            }
            if (!flags)
            {
                float dx = x - col;
                float dy = y - row;
                float dist2 = dx*dx+dy*dy;
                if (dist2 < minDist)
                {
                    minDist = dist2;
                    col1 = x;
                    row1 = y;
                }
            }
        }
    }
}


KeepPose::KeepPose()
{
    
    // init camera
    sensor.Initialize(YDPeopleSensor::ColorResolution::VGA, YDPeopleSensor::DepthResolution::VGA,true);
    sensor.SetDepthMappedToColor(true);
    sensor.Start();
    

    /*
    ifstream fin("images.txt");
    if (fin.is_open())
    {
         while (!fin.eof())
         {
              string image_path;
              fin >> image_path;
              image_paths.push_back(image_path);
         }
    }
    */
}

int KeepPose::initialize()
{
    // init scoring
    act_type = StatisticCount;  // StatisticTime;
    //act_name = Squat;
    //act_name = HalfSquat;
    //act_name = BasicSquatWithSideLegLift;
    //act_name = SquatWithKneeLiftAndTwist;
    act_name = CossackSquat;
    count_template_folder_path = "/workspace/KeepPoseScoreInterface_2d_3d_track/scoreModels/";
    body_info = {1.60,0.32,0};
    acc[0] = 0;
    acc[1] = 0;
    acc[2] = 0;
    TofApi *tof = new TofApi(act_type, act_name, body_info, count_template_folder_path);
    tofApiPtr = reinterpret_cast<long>(tof);

    // init pose detection
    mFrame = 0;
    calcFrame = 0;
    ifCalculating = true;
    numPeople = 0;
    allNotZero2d = false;
    lenLeftLeg = 0.40;
    lenRightLeg = 0.40;
    lenLeftThigh = 0.40;
    lenRightThigh = 0.40;
    allNotZero = false;
    frameData = new unsigned char[Height*Width*3];

    op::PoseExtractorCaffe *posePtr = new op::PoseExtractorCaffe(netInputSize, netOutputSize, outputSize, FLAGS_scale_number, poseModel,FLAGS_model_folder, FLAGS_num_gpu_start);
    posePtr->initializationOnThread();
    poseExtractorCaffePtr = reinterpret_cast<long>(posePtr);
    return 1;
}

KeepPose::~KeepPose()
{
    TofApi *tof = reinterpret_cast<TofApi*>(tofApiPtr);
    if (NULL != tof)
    {
        delete tof;
        tof = NULL;
    }
    op::PoseExtractorCaffe *poseExtractorCaffe = reinterpret_cast<op::PoseExtractorCaffe*>(poseExtractorCaffePtr);
    if (NULL != poseExtractorCaffe)
    {
        delete poseExtractorCaffe;
        poseExtractorCaffe = NULL;
    }
    if (NULL != frameData)
    {
        delete frameData;
        frameData = NULL;
    }
}


int KeepPose::get_frame()
{
   
    // get frame
    sensor.GetColorFrame(color);
    sensor.GetDepthFrame(depth);
    sensor.GetPublishData(pubData);
    colorWidth = color.Width;
    colorHeight = color.Height;
    colorFrameID = color.FrameID;
    colorTimestamp = color.Timestamp;
    int frameCount = 0;
    for (int i=0; i<colorHeight; i++)
    {
         int linecount = i * colorWidth;
         for (int j=0; j<colorWidth; j++)
         {
               unsigned int bgr = color.Pixels[linecount+j];
               frameData[frameCount++] = bgr & 0XFF;
               frameData[frameCount++] = (bgr & 0XFF00) >> 8;
               frameData[frameCount++] = (bgr & 0XFF0000) >> 16;
        }
    }
    mFrame++;
    if (mFrame > 1000)
    {
        mFrame = 1000;
    }

    return 1;
    
    /*
    colorWidth = 640;
    colorHeight = 480;
    colorFrameID = 0;
    colorTimestamp = 0;
    string image_path;
    if (i_image < image_paths.size() - 3)
    {
         image_path = image_paths[i_image];
         //std::cout << image_path << std::endl;
    }
    i_image++;
    cv::Mat src = imread(image_path.c_str(),1);
    Size dsize = Size(colorWidth,colorHeight);
    Mat dst;
    resize(src, dst, dsize, 0, 0);
    frameData = dst.data;

    return 1;
    */
}



void KeepPose::jointindex_for_draw(vector<float> &temp_points, vector<int> &pose_pairs)
{
    // 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
    // int posePairs[26] = {0,1,1,2,2,3,3,4,1,5,5,6,6,7,1,8,1,11,8,9,9,10,11,12,12,13};       
    for (int part=0; part<temp_points.size()/3; part++)
    {
          float pro = temp_points[3*part+2];
          float y = temp_points[3*part+1];
          float x = temp_points[3*part];
          if (pro < pointThd2 || x < 1.0 || x > Width - 2.0 || y < 1.0 || y > Height - 2.0)
          {
                temp_points[3*part] = -1.0;
                temp_points[3*part+1] = -1.0;
                temp_points[3*part+2] = -1.0;
                if (0 == part)
                {
                     pose_pairs[0] = -1;
                     pose_pairs[1] = -1;
                }
                if (1 == part)
                {
                     pose_pairs[0] = -1;
                     pose_pairs[1] = -1;
                     pose_pairs[2] = -1;
                     pose_pairs[3] = -1;
                     pose_pairs[8] = -1;
                     pose_pairs[9] = -1;
                     pose_pairs[14] = -1;
                     pose_pairs[15] = -1;
                     pose_pairs[16] = -1;
                     pose_pairs[17] = -1;
                }
                if (2 == part)
                {
                     pose_pairs[2] = -1;
                     pose_pairs[3] = -1;
                     pose_pairs[4] = -1;
                     pose_pairs[5] = -1;
                }
                if (3 == part)
                {
                     pose_pairs[4] = -1;
                     pose_pairs[5] = -1;
                     pose_pairs[6] = -1;
                     pose_pairs[7] = -1;
                }
                if (4 == part)
                {
                     pose_pairs[6] = -1;
                     pose_pairs[7] = -1;
                }
                if (5 == part)
                {
                     pose_pairs[8] = -1;
                     pose_pairs[9] = -1;
                     pose_pairs[10] = -1;
                     pose_pairs[11] = -1;
                }
                if (6 == part)
                {
                     pose_pairs[10] = -1;
                     pose_pairs[11] = -1;
                     pose_pairs[12] = -1;
                     pose_pairs[13] = -1;
                }
                if (7 == part)
                {
                     pose_pairs[12] = -1;
                     pose_pairs[13] = -1;
                }
                if (8 == part)
                {
                     pose_pairs[14] = -1;
                     pose_pairs[15] = -1;
                     pose_pairs[18] = -1;
                     pose_pairs[19] = -1;
                }
                if (9 == part)
                {
                     pose_pairs[18] = -1;
                     pose_pairs[19] = -1;
                     pose_pairs[20] = -1;
                     pose_pairs[21] = -1;
                }
                if (10 == part)
                {
                     pose_pairs[20] = -1;
                     pose_pairs[21] = -1;
                }
                if (11 == part)
                {
                     pose_pairs[16] = -1;
                     pose_pairs[17] = -1;
                     pose_pairs[22] = -1;
                     pose_pairs[23] = -1;
                }
                if (12 == part)
                {
                     pose_pairs[22] = -1;
                     pose_pairs[23] = -1;
                     pose_pairs[24] = -1;
                     pose_pairs[25] = -1;
                }
                if (13 == part)
                {
                     pose_pairs[24] = -1;
                     pose_pairs[25] = -1;
                }
          }
    }
}


float KeepPose::bbox_dist(float min_x, float min_y, float max_x, float max_y, float minx, float miny, float maxx, float maxy)
{
    // IoU
    float areaCur = (max_x - min_x) * (max_y - min_y);
    if (minx >= maxx || miny >= maxy || min_x >= max_x || min_y >= max_y)
    {
         return -1;
    }
    float leftX = max(min_x, minx);
    float rightX = min(max_x, maxx);
    float topY = max(min_y, miny);
    float bottomY = min(max_y, maxy);
    if (rightX >= leftX && bottomY >= topY)
    {
         float areaIOU = (rightX - leftX) * (bottomY - topY);
         return areaIOU / areaCur;
    }
    else
    {
         return -1;
    }
}

void KeepPose::calcMinMaxPoint(std::vector<float> &temp_points,float &min_x,float &min_y,float &max_x,float &max_y)
{
     float minHeight = Height;
     float minWidth = Width;
     float maxHeight = 0;
     float maxWidth = 0;
     for (int part = 0; part < temp_points.size(); part++)
     {
            float pro = temp_points[3*part+2];
            float x = temp_points[3*part];
            float y = temp_points[3*part+1];
            if (pro > pointThd && x > 1.0 && x < Width -1 && y > 1.0 && y < Height - 1)
            {
                   if (y > maxHeight)
                   {
                         maxHeight = y;
                         max_y = y;
                   }
                   if (y < minHeight)
                   {
                         minHeight = y;
                         min_y = y;
                   }
                   if (x > maxWidth)
                   {
                         maxWidth = x;
                         max_x = x;
                   }
                   if (x < minWidth)
                   {
                         minWidth = x;
                         min_x = x;
                   }
            }
     }
}


int KeepPose::get_2d_pose()
{
    if (mFrame < startFrameThd - 5)
    {
        return 0;
    }

    keypoints2d.clear();

    if (use2dForDisplay)
    {
         pose_pair_list.clear();
         keypoints2d_for_display.clear();
         int pose_pairs1[26] = {0,1,1,2,2,3,3,4,1,5,5,6,6,7,1,8,1,11,8,9,9,10,11,12,12,13};
         for (int k=0; k<26; k++)
         {
              pose_pair_list.push_back(pose_pairs1[k]);
         }
    }

    // model ptr
    op::PoseExtractorCaffe *poseExtractorCaffe = reinterpret_cast<op::PoseExtractorCaffe*>(poseExtractorCaffePtr);
    if (NULL == poseExtractorCaffe)
    {
        return 0;
    }
    // input image
    cv::Mat frame(colorHeight,colorWidth,CV_8UC3,cv::Scalar(255,255,255));
    frame.data = frameData;
    if (frame.empty())
    {
         return 0;
    }
  
    // detect 2d pose 
    op::Array<float> netInputArray;
    vector<float> scaleRatios;
    tie(netInputArray, scaleRatios) = cvMatToOpInput.format(frame);
    double scaleInputToOutput;
    op::Array<float> outputArray;
    tie(scaleInputToOutput, outputArray) = cvMatToOpOutput.format(frame);
    poseExtractorCaffe->forwardPass(netInputArray, {colorWidth, colorHeight}, scaleRatios);
    const auto poseKeypoints = poseExtractorCaffe->getPoseKeypoints();

    // get people
    numPeople = poseKeypoints.getSize(0);
    numParts = poseKeypoints.getSize(1);
    int numChannels = poseKeypoints.getSize(2);
    if (numPeople > 0)
    {
         int bestIndex = 0;
         if (numPeople > 1)
         {
             // get 2d keypoints
             vector<float> height_vec;
             vector<float> dist_vec;
             float maxHumanHeight = 0;
             float minHumanDist = 10000;
             float maxHeightIndex = 0;
             float minDistIndex = 0;
             float heightPro = 0;
             float distPro = 0;
             
             float maxIOU = 0;
             int iouIndex = 0;
             for (int person = 0; person < numPeople; person++)
             {
                  float maxHeight = 0;
                  float minHeight = 10000.0;
                  float meanX = 0;
                  int valid_count = 0;
                  float sumPro = 0;
                  vector<float> tmp_points;
                  for (int part = 0; part < numParts - 4; part++)
                  {
                        int index = numChannels * (person * numParts + part);
                        float pro = poseKeypoints[index+2];
                        float x = poseKeypoints[index];
                        float y = poseKeypoints[index+1];
                        sumPro += pro;
                        if (pro > pointThd && x > 1.0 && x < colorWidth -1 && y > 1.0 && y < colorHeight - 1)
                        {
                             if (y > maxHeight)
                             {
                                  maxHeight = y;
                             }
                             if (y < minHeight)
                             {
                                  minHeight = y;
                             }
                             meanX += poseKeypoints[index];
                             valid_count++;
                             tmp_points.push_back(x);
                             tmp_points.push_back(y);
                             tmp_points.push_back(pro);
                        }
                  }

                  float tmpHeight = maxHeight + (maxHeight - minHeight); //maxHeight - minHeight;
                  if (tmpHeight > maxHumanHeight)
                  {
                        maxHumanHeight = tmpHeight;
                        maxHeightIndex = person;
                        heightPro = sumPro;
                  }
                  height_vec.push_back(tmpHeight);
                  float tmpDist = 10000.0;
                  if (valid_count)
                  {
                        meanX /= float(valid_count);
                        tmpDist = fabs(meanX - centerX);
                  }
                  if (tmpDist < minHumanDist)
                  {
                        minHumanDist = tmpDist;
                        minDistIndex = person;
                        distPro = sumPro;
                  }
                  dist_vec.push_back(tmpDist);
 
                  /*                
                  if (mFrame >= startFrameThd - 5)
                  {
                        tmp_points.clear();
                        float max_x = Width;
                        float max_y = Height;
                        float min_x = 0;
                        float min_y = 0;
                        calcMinMaxPoint(tmp_points, min_x, min_y, max_x, max_y);
                        float IoU = bbox_dist(min_x, min_y, max_x, max_y, mMinX, mMinY, mMaxX, mMaxY);
                        if (IoU > maxIOU)
                        {
                            maxIOU = IoU;
                            iouIndex = person;
                        }
                  }
                  */
                  tmp_points.clear();
             }

             if (mFrame < startFrameThd - 5)
             {
                  if (maxHeightIndex == minDistIndex)
                  {
                       bestIndex = maxHeightIndex;
                  }
                  else
                  {
                       bestIndex = minDistIndex; //maxHeightIndex;
                       //if (heightPro > distPro)
                       //{
                       //     bestIndex = maxHeightIndex;
                       //}
                       //else
                       //{
                       //     bestIndex = minDistIndex;
                       //}
                  }
                  /*
                  vector<float> tmp_points;
                  for (int part = 0; part < numParts - 4; part++)
                  {
                        int index = numChannels * (bestIndex * numParts + part);
                        float pro = poseKeypoints[index+2];
                        float x = poseKeypoints[index];
                        float y = poseKeypoints[index+1];
                        if (pro > pointThd && x > 1.0 && x < colorWidth -1 && y > 1.0 && y < colorHeight - 1)
                        {
                             tmp_points.push_back(x);
                             tmp_points.push_back(y);
                             tmp_points.push_back(pro);
                        }
                  }
                  float max_x = Width;
                  float max_y = Height;
                  float min_x = 0;
                  float min_y = 0;
                  calcMinMaxPoint(tmp_points, min_x, min_y, max_x, max_y);
                  mMinX = min_x;
                  mMinY = min_y;
                  mMaxX = max_y;
                  mMaxY = max_y;
                  tmp_points.clear();
                  */
             }
             /*
             else if (maxIOU > 0)
             {
                  // IoU
                  bestIndex = iouIndex;
                  vector<float> tmp_points;
                  for (int part = 0; part < numParts - 4; part++)
                  {
                        int index = numChannels * (iouIndex * numParts + part);
                        float pro = poseKeypoints[index+2];
                        float x = poseKeypoints[index];
                        float y = poseKeypoints[index+1];
                        if (pro > pointThd && x > 1.0 && x < colorWidth -1 && y > 1.0 && y < colorHeight - 1)
                        {
                             tmp_points.push_back(x);
                             tmp_points.push_back(y);
                             tmp_points.push_back(pro);  
                        }
                  }               
                  float max_x = Width;
                  float max_y = Height;
                  float min_x = 0;
                  float min_y = 0; 
                  calcMinMaxPoint(tmp_points, min_x, min_y, max_x, max_y);
                  mMinX = min_x;
                  mMinY = min_y;
                  mMaxX = max_y;
                  mMaxY = max_y;
                  tmp_points.clear();
             }
             */
             height_vec.clear();
             dist_vec.clear();
             
         } // end numPeople > 1


         if (allNotZero2d)
         {
             // 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
             // 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
             // int posePairs[26] = {0,1,1,2,2,3,3,4,1,5,5,6,6,7,1,8,1,11,8,9,9,10,11,12,12,13};
             
             for (int part = 0; part < numParts - 4; part++)
             {
                 int indexB = numChannels * (bestIndex * numParts + part);
                 float x_2d = poseKeypoints[indexB];
                 float y_2d = poseKeypoints[indexB+1];
                 float pt_pro = poseKeypoints[indexB+2];
                 if (use2dForDisplay) 
                 {
                       keypoints2d_for_display.push_back(x_2d);
                       keypoints2d_for_display.push_back(y_2d);
                       keypoints2d_for_display.push_back(pt_pro);
                 }
                 if (pt_pro < pointThd2 || x_2d < 1.0 || x_2d > colorWidth - 2.0 || y_2d < 1.0 || y_2d > colorHeight - 2.0)
                 {
                       float d32_x = prepreX2d[part] - preprepreX2d[part];
                       float d21_x = preX2d[part] - prepreX2d[part];
                       float d32_y = prepreY2d[part] - preprepreY2d[part];
                       float d21_y = preY2d[part] - prepreY2d[part];
                       if (d32_x > d21_x > 0 || d32_x < d21_x < 0)
                       {
                           x_2d = preX2d[part] + (d21_x + d32_x) / 2.0;
                       }
                       else
                       {
                           x_2d = preX2d[part] + d21_x;
                       }   
                       if (d32_y > d21_y > 0 || d32_y < d21_y < 0)
                       {
                           y_2d = preY2d[part] + (d21_y + d32_y) / 2.0;
                       }
                       else
                       {
                           y_2d = preY2d[part] + d21_y;
                       }
                 }
                 else if (13==part || 10==part)
                 {
                      if (fabs(x_2d - preX2d[part])>100.0 || fabs(y_2d - preY2d[part])>100.0)
                      {
                             float d32_x = prepreX2d[part] - preprepreX2d[part];
                             float d21_x = preX2d[part] - prepreX2d[part];
                             float d32_y = prepreY2d[part] - preprepreY2d[part];
                             float d21_y = preY2d[part] - prepreY2d[part];
                             if (d32_x > d21_x > 0 || d32_x < d21_x < 0)
                             {
                                 x_2d = preX2d[part] + (d21_x + d32_x) / 2.0;
                             }
                             else
                             {
                                 x_2d = preX2d[part] + d21_x;
                             }
                             if (d32_y > d21_y > 0 || d32_y < d21_y < 0)
                             {
                                 y_2d = preY2d[part] + (d21_y + d32_y) / 2.0;
                             }
                             else
                             {
                                 y_2d = preY2d[part] + d21_y;
                             }
                      }
                 }

                 preprepreX2d[part] = prepreX2d[part];
                 preprepreY2d[part] = prepreY2d[part];
                 prepreX2d[part] = preX2d[part];
                 prepreY2d[part] = preY2d[part];
                 preX2d[part] = x_2d;
                 preY2d[part] = y_2d;
                 
                 keypoints2d.push_back(x_2d);
                 keypoints2d.push_back(y_2d);
                 keypoints2d.push_back(pt_pro);
             }
             // 11*3,8*3
             float hip_x = (keypoints2d[33] + keypoints2d[24]) / 2.0;
             float hip_y = (keypoints2d[34] + keypoints2d[25]) / 2.0;
             float hip_pro = (keypoints2d[35] + keypoints2d[26]) / 2.0;
             keypoints2d.push_back(hip_x);
             keypoints2d.push_back(hip_y);
             keypoints2d.push_back(hip_pro);
             if (use2dForDisplay)
             {
                  // 2d for display
                  jointindex_for_draw(keypoints2d_for_display, pose_pair_list);
             }

         }

         
         // tracking for low threshold point
         if (allNotZero2d)
         {
             bool valid_all_2d = true;
             for (int part = 0; part < numParts - 4; part++)
             {
                 float y = keypoints2d[3*part+1];
                 float x = keypoints2d[3*part];
                 if (x<1.0 || x>colorWidth-2 || y<1.0 || y>colorHeight-2)
                 {
                     allNotZero2d = false;
                     return 0;
                 }
             }
         }
         

         if (!allNotZero2d)
         {
             bool valid_all_2d = true;
             for (int part = 0; part < numParts - 4; part++)
             {
                 int indexB = numChannels * (bestIndex * numParts + part);
                 float pro = poseKeypoints[indexB+2];
                 float y = poseKeypoints[indexB+1];
                 float x = poseKeypoints[indexB];
                 if (13 == part)  // 13,12; 10,9
                 {
                     float xlknee = poseKeypoints[numChannels * (bestIndex * numParts + 12)];
                     float ylknee = poseKeypoints[numChannels * (bestIndex * numParts + 12 + 1)];
                     if (fabs(xlknee - x) > 0.7 * fabs(ylknee - y))
                     {
                           valid_all_2d = false;
                           return 0;
                     }
                 }
                 if (10 == part)  // 13,12; 10,9
                 {
                     float xrknee = poseKeypoints[numChannels * (bestIndex * numParts + 9)];
                     float yrknee = poseKeypoints[numChannels * (bestIndex * numParts + 9 + 1)];
                     if (fabs(xrknee - x) > 0.7 * fabs(yrknee - y))
                     {
                           valid_all_2d = false;
                           return 0;
                     }
                 }
                 if (poseKeypoints[indexB+2] < pointThd2 || fabs(x)<1.0 || fabs(y)<1.0 )
                 {
                     valid_all_2d = false;
                     return 0;
                 }
             }
             if (valid_all_2d)
             {
                 allNotZero2d = true;
                 for (int part = 0; part < numParts - 4; part++)
                 { 
                     int indexB = numChannels * (bestIndex * numParts + part);
                     preX2d[part] = poseKeypoints[indexB];
                     preY2d[part] = poseKeypoints[indexB+1];
                     prepreX2d[part] = preX2d[part];
                     prepreY2d[part] = preY2d[part];
                     preprepreX2d[part] = prepreX2d[part];
                     preprepreY2d[part] = prepreY2d[part];     
                     keypoints2d.push_back(poseKeypoints[indexB]);
                     keypoints2d.push_back(poseKeypoints[indexB+1]);
                     keypoints2d.push_back(poseKeypoints[indexB+2]);       
                     keypoints2d_for_display.push_back(poseKeypoints[indexB]);
                     keypoints2d_for_display.push_back(poseKeypoints[indexB+1]);
                     keypoints2d_for_display.push_back(poseKeypoints[indexB+2]);        
                 }
                 // 11*3,8*3
                 float hip_x = (keypoints2d[33] + keypoints2d[24]) / 2.0;
                 float hip_y = (keypoints2d[34] + keypoints2d[25]) / 2.0;
                 float hip_pro = (keypoints2d[35] + keypoints2d[26]) / 2.0;
                 keypoints2d.push_back(hip_x);
                 keypoints2d.push_back(hip_y);
                 keypoints2d.push_back(hip_pro);
             }

         }

         return 1;   
    }

    return 0;

}



void KeepPose::predict_invalid_knee(std::vector<float> &keypoints3d_temp)
{
    // 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
    // predict bad ank
    float x1 = keypoints3d_temp[45]; // 4*11+1
    float y1 = keypoints3d_temp[46];
    float z1 = keypoints3d_temp[47];
    float x2 = keypoints3d_temp[49]; // 4*12+1
    float y2 = keypoints3d_temp[50];
    float z2 = keypoints3d_temp[51];


    float lenLeftThigh1 = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
    if (lenLeftThigh1 > 1.25*lenLeftThigh || lenLeftThigh1 < 0.75*lenLeftThigh || z2<minDepthBound || z2>maxDepthBound)
    {
          if (!useFourFramesForPredict)
          {
              keypoints3d_temp[51] = 2*prepreZ[12] - preprepreZ[12];
              keypoints3d_temp[50] = 2*prepreY[12] - preprepreY[12];
              keypoints3d_temp[49] = 2*prepreX[12] - preprepreX[12];
          }
          else
          {
              float dz32 = preprepreZ[12] - prepreprepreZ[12];
              float dz21 = prepreZ[12] - preprepreZ[12];
              float dy32 = preprepreY[12] - prepreprepreY[12];
              float dy21 = prepreY[12] - preprepreY[12];
              float dx32 = preprepreX[12] - prepreprepreX[12];
              float dx21 = prepreX[12] - preprepreX[12];
              if (dz32 > dz21 > 0 || dz32 < dz21 < 0)
              {
                   keypoints3d_temp[51] = prepreZ[12] + (dz21 + dz32) / 2.0;
              }
              else
              {   
                   keypoints3d_temp[51] = prepreZ[12] + dz21;
              }
              if (dy32 > dy21 > 0 || dy32 < dy21 < 0)
              {
                   keypoints3d_temp[50] = prepreY[12] + (dy21 + dy32) / 2.0;
              }
              else
              {      
                   keypoints3d_temp[50] = prepreY[12] + dy21;
              }
              if (dx32 > dx21 > 0 || dx32 < dx21 < 0)
              {
                   keypoints3d_temp[49] = prepreX[12] + (dx21 + dx32) / 2.0;
              }
              else
              {      
                   keypoints3d_temp[49] = prepreX[12] + dx21;
              }
          }
          //keypoints3d_temp[49] = (keypoints3d_temp[49] + prepreX[12]) / 2.0;
          //keypoints3d_temp[50] = (keypoints3d_temp[50] + prepreY[12]) / 2.0;
          //keypoints3d_temp[51] = (keypoints3d_temp[51] + prepreZ[12]) / 2.0;          
    }
    x1 = keypoints3d_temp[33]; // 4*8+1
    y1 = keypoints3d_temp[34];
    z1 = keypoints3d_temp[35];
    x2 = keypoints3d_temp[37]; // 4*9+1
    y2 = keypoints3d_temp[38];
    z2 = keypoints3d_temp[39];
    float lenRightThigh1 = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
    if (lenRightThigh1 > 1.25*lenRightThigh || lenRightThigh1 < 0.75*lenRightThigh || z2<minDepthBound || z2>maxDepthBound)
    {
          if (!useFourFramesForPredict)
          {
              keypoints3d_temp[39] = 2*prepreZ[9] - preprepreZ[9];
              keypoints3d_temp[38] = 2*prepreY[9] - preprepreY[9];
              keypoints3d_temp[37] = 2*prepreX[9] - preprepreX[9];
          }
          else
          {
              float dz32 = preprepreZ[9] - prepreprepreZ[9];
              float dz21 = prepreZ[9] - preprepreZ[9];
              float dy32 = preprepreY[9] - prepreprepreY[9];
              float dy21 = prepreY[9] - preprepreY[9];
              float dx32 = preprepreX[9] - prepreprepreX[9];
              float dx21 = prepreX[9] - preprepreX[9];
              if (dz32 > dz21 > 0 || dz32 < dz21 < 0)
              {
                   keypoints3d_temp[39] = prepreZ[9] + (dz21 + dz32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[39] = prepreZ[9] + dz21;
              }
              if (dy32 > dy21 > 0 || dy32 < dy21 < 0) 
              {
                   keypoints3d_temp[38] = prepreY[9] + (dy21 + dy32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[38] = prepreY[9] + dy21;
              }
              if (dx32 > dx21 > 0 || dx32 < dx21 < 0)
              {
                   keypoints3d_temp[37] = prepreX[9] + (dx21 + dx32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[37] = prepreX[9] + dx21;
              }
          }
          //keypoints3d_temp[37] = (keypoints3d_temp[37] + prepreX[9]) / 2.0;
          //keypoints3d_temp[38] = (keypoints3d_temp[38] + prepreY[9]) / 2.0;
          //keypoints3d_temp[39] = (keypoints3d_temp[39] + prepreZ[9]) / 2.0;
    }
}


void KeepPose::predict_invalid_ank(std::vector<float> &keypoints3d_temp)
{
    // 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
    // predict bad ank
    float x1 = keypoints3d_temp[49]; // 4*12+1
    float y1 = keypoints3d_temp[50];
    float z1 = keypoints3d_temp[51];
    float x2 = keypoints3d_temp[53]; // 4*13+1
    float y2 = keypoints3d_temp[54];
    float z2 = keypoints3d_temp[55];
    float lenLeftLeg1 = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
    if (lenLeftLeg1 > 1.25*lenLeftLeg || lenLeftLeg1 < 0.75*lenLeftLeg)
    {
          if (!useFourFramesForPredict)
          {
              keypoints3d_temp[55] = 2*prepreZ[13] - preprepreZ[13];
              keypoints3d_temp[54] = 2*prepreY[13] - preprepreY[13];
              keypoints3d_temp[53] = 2*prepreX[13] - preprepreX[13];
          }
          else
          {
              float dz32 = preprepreZ[13] - prepreprepreZ[13];
              float dz21 = prepreZ[13] - preprepreZ[13];
              float dy32 = preprepreY[13] - prepreprepreY[13];
              float dy21 = prepreY[13] - preprepreY[13];
              float dx32 = preprepreX[13] - prepreprepreX[13];
              float dx21 = prepreX[13] - preprepreX[13];
              if (dz32 > dz21 > 0 || dz32 < dz21 < 0)
              {
                   keypoints3d_temp[55] = prepreZ[13] + (dz21 + dz32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[55] = prepreZ[13] + dz21;
              }
              if (dy32 > dy21 > 0 || dy32 < dy21 < 0)
              {
                   keypoints3d_temp[54] = prepreY[13] + (dy21 + dy32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[54] = prepreY[13] + dy21;
              }
              if (dx32 > dx21 > 0 || dx32 < dx21 < 0)
              {
                   keypoints3d_temp[53] = prepreX[13] + (dx21 + dx32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[53] = prepreX[13] + dx21;
              }

          }
          //keypoints3d_temp[53] = (keypoints3d_temp[53] + prepreX[13]) / 2.0;
          //keypoints3d_temp[54] = (keypoints3d_temp[54] + prepreY[13]) / 2.0;
          //keypoints3d_temp[55] = (keypoints3d_temp[55] + prepreZ[13]) / 2.0;
    }
    x1 = keypoints3d_temp[37]; // 4*9+1
    y1 = keypoints3d_temp[38];
    z1 = keypoints3d_temp[39];
    x2 = keypoints3d_temp[41]; // 4*10+1
    y2 = keypoints3d_temp[42];
    z2 = keypoints3d_temp[43];
    float lenRightLeg1 = sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
    if (lenRightLeg1 > 1.25*lenRightLeg || lenRightLeg1 < 0.75*lenRightLeg)
    {
          if (!useFourFramesForPredict)
          {
              keypoints3d_temp[43] = 2*prepreZ[10] - preprepreZ[10];
              keypoints3d_temp[42] = 2*prepreY[10] - preprepreY[10];
              keypoints3d_temp[41] = 2*prepreX[10] - preprepreX[10];
          }
          else
          {
              float dz32 = preprepreZ[10] - prepreprepreZ[10];
              float dz21 = prepreZ[10] - preprepreZ[10];
              float dy32 = preprepreY[10] - prepreprepreY[10];
              float dy21 = prepreY[10] - preprepreY[10];
              float dx32 = preprepreX[10] - prepreprepreX[10];
              float dx21 = prepreX[10] - preprepreX[10];
              if (dz32 > dz21 > 0 || dz32 < dz21 < 0)
              {
                   keypoints3d_temp[43] = prepreZ[10] + (dz21 + dz32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[43] = prepreZ[10] + dz21;
              }
              if (dy32 > dy21 > 0 || dy32 < dy21 < 0)
              {
                   keypoints3d_temp[42] = prepreY[10] + (dy21 + dy32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[42] = prepreY[10] + dy21;
              }
              if (dx32 > dx21 > 0 || dx32 < dx21 < 0)
              {
                   keypoints3d_temp[41] = prepreX[10] + (dx21 + dx32) / 2.0;
              }
              else
              {
                   keypoints3d_temp[41] = prepreX[10] + dx21;
              }
          }
          //keypoints3d_temp[41] = (keypoints3d_temp[41] + prepreX[10]) / 2.0;
          //keypoints3d_temp[42] = (keypoints3d_temp[42] + prepreY[10]) / 2.0;
          //keypoints3d_temp[43] = (keypoints3d_temp[43] + prepreZ[10]) / 2.0;
    }

}


void KeepPose::predict_invalid_3d_knee(std::vector<float> &points3d, std::vector<float> &points2d)
{
    int index_lank = 13;
    int index_rank = 10;
    int index_lknee = 12;
    int index_rknee = 9;
    int index_lhip = 11;
    int index_rhip = 8;
    float p_lank = points3d[4*index_lank];
    float x_lank = points3d[4*index_lank+1];
    float y_lank = points3d[4*index_lank+2];
    float z_lank = points3d[4*index_lank+3];
    float p_rank = points3d[4*index_rank];
    float x_rank = points3d[4*index_rank+1];
    float y_rank = points3d[4*index_rank+2];
    float z_rank = points3d[4*index_rank+3];
    float p_lknee = points3d[4*index_lknee];
    float x_lknee = points3d[4*index_lknee+1];
    float y_lknee = points3d[4*index_lknee+2];
    float z_lknee = points3d[4*index_lknee+3];
    float p_rknee = points3d[4*index_rknee];
    float x_rknee = points3d[4*index_rknee+1];
    float y_rknee = points3d[4*index_rknee+2];
    float z_rknee = points3d[4*index_rknee+3];
    float p_lhip = points3d[4*index_lhip];
    float x_lhip = points3d[4*index_lhip+1];
    float y_lhip = points3d[4*index_lhip+2];
    float z_lhip = points3d[4*index_lhip+3];
    float p_rhip = points3d[4*index_rhip];
    float x_rhip = points3d[4*index_rhip+1];
    float y_rhip = points3d[4*index_rhip+2];
    float z_rhip = points3d[4*index_rhip+3];
    float x_lhip_2d = points2d[3*index_lhip];
    float y_lhip_2d = points2d[3*index_lhip+1];
    float x_rhip_2d = points2d[3*index_rhip];
    float y_rhip_2d = points2d[3*index_rhip+1];
    float x_lank_2d = points2d[3*index_lank];
    float y_lank_2d = points2d[3*index_lank+1];
    float x_rank_2d = points2d[3*index_rank];
    float y_rank_2d = points2d[3*index_rank+1];
    float x_lknee_2d = points2d[3*index_lknee];
    float y_lknee_2d = points2d[3*index_lknee+1];
    float x_rknee_2d = points2d[3*index_rknee];
    float y_rknee_2d = points2d[3*index_rknee+1];
    
    float lenLeftThigh1 = sqrt((x_lhip-x_lknee)*(x_lhip-x_lknee)+(y_lhip-y_lknee)*(y_lhip-y_lknee)+(z_lhip-z_lknee)*(z_lhip-z_lknee));   
    float lenLeftLeg1 = sqrt((x_lank-x_lknee)*(x_lank-x_lknee)+(y_lank-y_lknee)*(y_lank-y_lknee)+(z_lank-z_lknee)*(z_lank-z_lknee));
    int cond_1 = lenLeftLeg1 / lenLeftLeg < 0.4 || lenLeftThigh1 / lenLeftThigh < 0.65;
    int cond_2 = (z_lknee>=z_lank-0.02&&z_lknee<=z_lhip+0.02) || (z_lknee<=z_lank+0.02&&z_lknee>=z_lhip-0.02);
    int cond_3 = (x_lknee_2d>=x_lank_2d&&x_lknee_2d<=x_lhip_2d) || (x_lknee_2d<=x_lank_2d&&x_lknee_2d>=x_lhip_2d);
    int cond_4 = (y_lknee_2d<y_lank_2d&&y_lknee_2d>y_lhip_2d);
    int cond_5 =  y_rknee - y_rank > lenLeftLeg * 0.5  && y_rknee -y_lank > lenLeftLeg * 0.5;
    if (cond_1 && cond_2 && cond_3 && cond_4 && cond_5)
    {
         points3d[4*index_lknee] = points3d[4*index_lank];
         points3d[4*index_lknee+1] = points3d[4*index_lank+1];
         points3d[4*index_lknee+2] = points3d[4*index_lank+2];
         points3d[4*index_lknee+3] = points3d[4*index_lank+3];
         points2d[3*index_lknee] = points2d[3*index_lank];
         points2d[3*index_lknee+1] = points2d[3*index_lank+1];
         std::cout << "knee=ank" << std::endl;
    }

    float lenRightThigh1 = sqrt((x_rhip-x_rknee)*(x_rhip-x_rknee)+(y_rhip-y_rknee)*(y_rhip-y_rknee)+(z_rhip-z_rknee)*(z_rhip-z_rknee));
    float lenRightLeg1 = sqrt((x_rank-x_rknee)*(x_rank-x_rknee)+(y_rank-y_rknee)*(y_rank-y_rknee)+(z_rank-z_rknee)*(z_rank-z_rknee));
    cond_1 = lenRightLeg1 / lenRightLeg < 0.4 || lenRightThigh1 / lenRightThigh < 0.65;
    cond_2 = (z_rknee>=z_rank-0.02&&z_rknee<=z_rhip+0.02) || (z_rknee<=z_rank+0.02&&z_rknee>=z_rhip-0.02);
    cond_3 = (x_rknee_2d>=x_rank_2d&&x_rknee_2d<=x_rhip_2d) || (x_rknee_2d<=x_rank_2d&&x_rknee_2d>=x_rhip_2d);
    cond_4 = (y_rknee_2d<y_rank_2d&&y_rknee_2d>y_rhip_2d);
    cond_5 =  y_lknee - y_lank > lenRightLeg * 0.5 && y_lknee -y_rank > lenRightLeg * 0.5;
    if (cond_1 && cond_2 && cond_3 && cond_4 && cond_5)
    {
         points3d[4*index_rknee] = points3d[4*index_rank];
         points3d[4*index_rknee+1] = points3d[4*index_rank+1];
         points3d[4*index_rknee+2] = points3d[4*index_rank+2];
         points3d[4*index_rknee+3] = points3d[4*index_rank+3];
         points2d[3*index_rknee] = points2d[3*index_rank];
         points2d[3*index_rknee+1] = points2d[3*index_rank+1];
         std::cout << "knee=ank" << std::endl;
    }
}



void KeepPose::predict_invalid_3d_ank(vector<float> &points3d, vector<float> &points2d)
{
    int index_lank = 13;
    int index_rank = 10;
    int index_lknee = 12;
    int index_rknee = 9;
    int index_lhip = 11;
    int index_rhip = 8;
    float p_lank = points3d[4*index_lank];
    float x_lank = points3d[4*index_lank+1];
    float y_lank = points3d[4*index_lank+2];
    float z_lank = points3d[4*index_lank+3];
    float p_rank = points3d[4*index_rank];
    float x_rank = points3d[4*index_rank+1];
    float y_rank = points3d[4*index_rank+2];
    float z_rank = points3d[4*index_rank+3];
    float p_lknee = points3d[4*index_lknee];
    float x_lknee = points3d[4*index_lknee+1];
    float y_lknee = points3d[4*index_lknee+2];
    float z_lknee = points3d[4*index_lknee+3];
    float p_rknee = points3d[4*index_rknee];
    float x_rknee = points3d[4*index_rknee+1];
    float y_rknee = points3d[4*index_rknee+2];
    float z_rknee = points3d[4*index_rknee+3];
    float p_lhip = points3d[4*index_lhip];
    float x_lhip = points3d[4*index_lhip+1];
    float y_lhip = points3d[4*index_lhip+2];
    float z_lhip = points3d[4*index_lhip+3];
    float p_rhip = points3d[4*index_rhip];
    float x_rhip = points3d[4*index_rhip+1];
    float y_rhip = points3d[4*index_rhip+2];
    float z_rhip = points3d[4*index_rhip+3];

    float x_lhip_2d = points2d[3*index_lhip];
    float y_lhip_2d = points2d[3*index_lhip+1];
    float x_rhip_2d = points2d[3*index_rhip];
    float y_rhip_2d = points2d[3*index_rhip+1];
    float x_lank_2d = points2d[3*index_lank];
    float y_lank_2d = points2d[3*index_lank+1];
    float x_rank_2d = points2d[3*index_rank];
    float y_rank_2d = points2d[3*index_rank+1];
    float x_lknee_2d = points2d[3*index_lknee];
    float y_lknee_2d = points2d[3*index_lknee+1];
    float x_rknee_2d = points2d[3*index_rknee];
    float y_rknee_2d = points2d[3*index_rknee+1];

    float thd_z = 0.05;
    float thd_2d = 15.0; // 1.75,640-480 0.05 
    float thd_y1 = 0.15;
    float thd_y2 = 0.10; 
    
    // lank, blocked by lthigh
    int cond_1 = (z_lank>=z_lknee-thd_z&&z_lank<=z_lhip+thd_z) || (z_lank<=z_lknee+thd_z&&z_lank>=z_lhip-thd_z);
    int cond_2 = (x_lank_2d>=x_lknee_2d-thd_2d&&x_lank_2d<=x_lhip_2d+thd_2d) || (x_lank_2d<=x_lknee_2d+thd_2d&&x_lank_2d>=x_lhip_2d-thd_2d);
    int cond_3 = (y_lank_2d<y_lknee_2d+thd_2d&&y_lank_2d>y_lhip_2d-thd_2d);
    int cond_5 =  y_rknee - y_rank > thd_y1 && y_rknee - y_lknee > thd_y2 && (z_lank < z_lknee + thd_z || z_lank > maxDepthBound || z_lknee < minDepthBound);
    if (cond_5)
    {
         float dy_1 = y_lhip - y_lknee;
         float dz_1 = z_lhip - z_lknee;
         float dy_2 = y_lank - y_lknee;
         float dz_2 = z_lank - z_lknee;
         float theta_1 = 180.0 * atan2(double(dy_1),double(dz_1)) / 3.1415926; // -180~180
         float theta_2 = 180.0 * atan2(double(dy_2),double(dz_2)) / 3.1415926; // -180~180
         if (theta_1 < 0) theta_1 += 360.0;
         if (theta_2 < 0) theta_2 += 360.0;
         int cond_6 = theta_2 - theta_1 < 200.0;
         if (cond_6)
         {
              points3d[4*index_lank+1] = points3d[4*index_lknee+1];
              points3d[4*index_lank+2] = points3d[4*index_lknee+2];
              points3d[4*index_lank+3] = points3d[4*index_lknee+3] + lenLeftLeg;
         }
         std::cout << ">180" << std::endl;
    }
    else 
    {
         if (cond_1 && cond_2 && cond_3)
         {
             // line_lknee_lhip, x = ky + b, or: dx * y - dy * x + dy * b = 0; 
             float dx = x_lhip_2d - x_lknee_2d;
             float dy = y_lhip_2d - y_lknee_2d;
             float len_2d = sqrt(dx*dx+dy*dy);
             if (len_2d > 2.0)
             {
                   float line_lank = fabs(dx * y_lank_2d - dy * x_lank_2d + x_lhip_2d * dy - y_lank_2d * dx) / len_2d; 
                   if (fabs(line_lank) < 15.0)
                   {
                       float len_temp_2d = sqrt((x_lank_2d-x_lknee_2d)*(x_lank_2d-x_lknee_2d)+(y_lank_2d-y_lknee_2d)*(y_lank_2d-y_lknee_2d));
                       points3d[4*index_lank+3] = points3d[4*index_lknee+3] + lenLeftLeg * (len_2d - len_temp_2d) / len_2d;
                       std::cout << "blocking" << std::endl;
                   }
             }
         }
    }
    
    
    // rank, blocked by rthigh
    cond_1 = (z_rank>=z_rknee-thd_z&&z_rank<=z_rhip+thd_z) || (z_rank<=z_rknee+thd_z&&z_rank>=z_rhip-thd_z);
    cond_2 = (x_rank_2d>=x_rknee_2d-thd_2d&&x_rank_2d<=x_rhip_2d+thd_2d) || (x_rank_2d<=x_rknee_2d+thd_2d&&x_rank_2d>=x_rhip_2d-thd_2d);
    cond_3 = (y_rank_2d<y_rknee_2d + thd_2d && y_rank_2d>y_rhip_2d - thd_2d);
    cond_5 = y_lknee - y_lank > thd_y1 && y_lknee - y_rknee > thd_y2 && (z_rank < z_rknee + thd_z || z_rank > maxDepthBound || z_rknee < minDepthBound);
    if (cond_5)
    {
         float dy_1 = y_rhip - y_rknee;
         float dz_1 = z_rhip - z_rknee;
         float dy_2 = y_rank - y_rknee;
         float dz_2 = z_rank - z_rknee;
         float theta_1 = 180.0 * atan2(double(dy_1),double(dz_1)) / 3.1415926; // -180~180
         float theta_2 = 180.0 * atan2(double(dy_2),double(dz_2)) / 3.1415926; // -180~180
         if (theta_1 < 0) theta_1 += 360.0;
         if (theta_2 < 0) theta_2 += 360.0;
         int cond_6 = theta_2 - theta_1 < 200.0;
         if (cond_6)
         {
               points3d[4*index_rank+1] = points3d[4*index_rknee+1];
               points3d[4*index_rank+2] = points3d[4*index_rknee+2];
               points3d[4*index_rank+3] = points3d[4*index_rknee+3] + lenRightLeg;
               std::cout << ">180" << std::endl;
         }
    }

    else 
    {
         if (cond_1 && cond_2 && cond_3)
         {
               // line_rknee_rhip, x = ky + b, or: dx * y - dy * x + dy * b = 0;
               float dx = x_rhip_2d - x_rknee_2d;
               float dy = y_rhip_2d - y_rknee_2d;
               float len_2d = sqrt(dx*dx+dy*dy);
               if (len_2d > 2.0)
               {
                   float line_rank = fabs(dx * y_rank_2d - dy * x_rank_2d + x_rhip_2d * dy - y_rank_2d * dx) / len_2d;
                   if (fabs(line_rank) < 15.0)
                   {
                       float len_temp_2d = sqrt((x_rank_2d-x_rknee_2d)*(x_rank_2d-x_rknee_2d)+(y_rank_2d-y_rknee_2d)*(y_rank_2d-y_rknee_2d));
                       points3d[4*index_rank+3] = points3d[4*index_rknee+3] + lenRightLeg * (len_2d - len_temp_2d) / len_2d;
                       std::cout << "blocking" << std::endl;
                   }
               }
          }
    }

}



int KeepPose::get_3d_pose()
{
    if (mFrame < startFrameThd)
    {
         return 0;
    }

    keypoints2d = keypoints2d_for_display;
    keypoints3d.clear();
    
    if (numPeople < 1)
    {
         return 0;
    }
    
    /*
    for (int part = 0; part < numParts-3; part++)
    {
         float fcol = keypoints2d[3*part];
         float frow = keypoints2d[3*part+1];
         if (fcol<padW || fcol>colorWidth-padW || frow<padH || frow>colorHeight-padH)
         {
              return 0;
         }
    }
    */

    // reconstruct 3d keypoints
    vector<float> keypoints3d_temp;
    for (int part = 0; part < numParts-3; part++)
    {
         float fcol = keypoints2d[3*part];
         float frow = keypoints2d[3*part+1];
         int col = max(float(0),min(fcol,float(colorWidth-1)));
         int row = max(float(0),min(frow,float(colorHeight-1)));
         float proba = keypoints2d[3*part+2];
         float pointDepth = float(depth.Pixels[row*colorWidth+col])/1000.0;
         float x_3d, y_3d, z_3d, p_3d;

         if (pointDepth < minDepthBound || pointDepth > maxDepthBound)
         {
              int col1 = col;
              int row1 = row;
              nearest_mask_points(pubData.UserMask.Mask,col,row,colorWidth,colorHeight,col1,row1);
              pointDepth = float(depth.Pixels[row1*colorWidth+col1])/1000.0;
         }

         if ((pointDepth < minDepthBound || pointDepth > maxDepthBound) && allNotZero)
         {
              if (!useFourFramesForPredict)
              {
                   pointDepth = 2.0*preZ[part]-prepreZ[part];
              }
              else
              {
                   float d32 = prepreZ[part] - preprepreZ[part];
                   float d21 = preZ[part] - prepreZ[part];
                   if (d32 > d21 > 0 || d32 < d21 < 0)
                   {
                       pointDepth = preZ[part] + (d21 + d32) / 2.0;
                   }
                   else
                   {
                       pointDepth = preZ[part] + d21;
                   }
              }
         }

             

         if (useSmoothZ)
         {
              if (allNotZero)
              {
                   if (keypoints2d[3*part+2] > pointThd2)
                   {
                       pointDepth = (pointDepth + preZ[part] + prepreZ[part]) / 3.0;
                   }
              }
         }

         p_3d = proba;
         z_3d = pointDepth;
         x_3d = z_3d * float(col - centerX) / focusX;
         y_3d = z_3d * float(centerY - row) / focusY;

         
         if (fcol<padW || fcol>colorWidth-padW || frow<padH || frow>colorHeight-padH)
         {
               z_3d = 2.0*preZ[part]-prepreZ[part];
               y_3d = 2.0*preY[part]-prepreY[part];
               x_3d = 2.0*preX[part]-prepreX[part];
         }
         

         keypoints3d_temp.push_back(p_3d);
         keypoints3d_temp.push_back(x_3d);
         keypoints3d_temp.push_back(y_3d);
         keypoints3d_temp.push_back(z_3d);

         if (allNotZero)
         {
              prepreprepreZ[part] = preprepreZ[part];
              preprepreZ[part] = prepreZ[part];
              prepreZ[part] = preZ[part];
              preZ[part] = z_3d;
              prepreprepreY[part] = preprepreY[part];
              preprepreY[part] = prepreY[part];
              prepreY[part] = preY[part];
              preY[part] = y_3d;
              prepreprepreX[part] = preprepreX[part];
              preprepreX[part] = prepreX[part];
              prepreX[part] = preX[part];
              preX[part] = x_3d;
         }

    } // end part


    // smooth x,y,z
    if (allNotZero && (!ifCalculating))
    {
         std::cout << lenLeftThigh << "," << lenLeftLeg << "," << lenRightThigh << "," << lenRightLeg << std::endl;
         if (useSmoothKneeAnk)
         {
              predict_invalid_3d_knee(keypoints3d_temp,keypoints2d);
              //predict_invalid_knee(keypoints3d_temp);
              predict_invalid_3d_ank(keypoints3d_temp,keypoints2d);
              //for (int kk=0; kk<keypoints3d_temp.size()/4; kk++)
              //{
              //     preZ[kk] = keypoints3d_temp[4*kk+3];
              //     preY[kk] = keypoints3d_temp[4*kk+2];
              //     preX[kk] = keypoints3d_temp[4*kk+1];
              //}
         }
         if (useSmoothLeg)
         {
              // 0-nose,1-neck,2-rsho,3-relb,4-rwri,5-lsho,6-lelb,7-lwri,8-rhip,9-rknee,10-rank,11-lhip,12-lknee,13-lank
              // predict bad ank
              //predict_invalid_ank(keypoints3d_temp);
         }
         if (useAllSmooth)
         {
              for (int kk=0; kk < keypoints3d_temp.size()/4; kk++)
              {
                  keypoints3d_temp[4*kk+1] = (keypoints3d_temp[4*kk+1] + prepreX[kk]) / 2.0;
                  keypoints3d_temp[4*kk+2] = (keypoints3d_temp[4*kk+2] + prepreY[kk]) / 2.0;
                  keypoints3d_temp[4*kk+3] = (keypoints3d_temp[4*kk+3] + prepreZ[kk]) / 2.0;
              }
         }
         // output
         keypoints3d.push_back(float(colorFrameID));
         keypoints3d.push_back(float(colorTimestamp));
         for (int kk=0; kk < keypoints3d_temp.size(); kk++)
         {
              keypoints3d.push_back(keypoints3d_temp[kk]);
         }
         keypoints3d.push_back(0);
         keypoints3d.push_back(9.8);
         keypoints3d.push_back(0);
         keypoints3d_temp.clear();
         return 1;
    }

          
    if (!allNotZero)
    {
         int countNotZero = 0;
         for (int kk=0; kk<keypoints3d_temp.size()/4; kk++)
         {
              float pointDepth = keypoints3d_temp[kk*4+3];
              if ((!(pointDepth < minDepthBound || pointDepth > maxDepthBound)))
              {
                   countNotZero++;
              }
         }
         if (15 == countNotZero)
         {
              for (int kk=0; kk<keypoints3d_temp.size()/4; kk++)
              {
                   preZ[kk] = keypoints3d_temp[kk*4+3];
                   prepreZ[kk] = preZ[kk];
                   preprepreZ[kk] = prepreZ[kk];
                   prepreprepreZ[kk] = preprepreZ[kk];
                   preY[kk] = keypoints3d_temp[kk*4+2];
                   prepreY[kk] = preY[kk];
                   preprepreY[kk] = prepreY[kk];
                   prepreprepreY[kk] = preprepreY[kk];
                   preX[kk] = keypoints3d_temp[kk*4+1];
                   prepreX[kk] = preX[kk];
                   preprepreX[kk] = prepreX[kk];
                   prepreprepreX[kk] = preprepreX[kk]; 
              }
              allNotZero = true;
         }
    }

    // calc leg & thigh length
    if (allNotZero && ifCalculating)
    {
         float x_lank = keypoints3d_temp[53]; // 4*13+1
         float y_lank = keypoints3d_temp[54];
         float z_lank = keypoints3d_temp[55];
         float x_rank = keypoints3d_temp[41]; // 4*10+1
         float y_rank = keypoints3d_temp[42];
         float z_rank = keypoints3d_temp[43];
         float x_lknee = keypoints3d_temp[49];  // 4*12+1
         float y_lknee = keypoints3d_temp[50];
         float z_lknee = keypoints3d_temp[51];
         float x_rknee = keypoints3d_temp[37];  // 4*9+1
         float y_rknee = keypoints3d_temp[38];
         float z_rknee = keypoints3d_temp[39];
         float x_lhip = keypoints3d_temp[45];  // 4*11+1
         float y_lhip = keypoints3d_temp[46];
         float z_lhip = keypoints3d_temp[47];
         float x_rhip = keypoints3d_temp[33];  // 4*8+1
         float y_rhip = keypoints3d_temp[34];
         float z_rhip = keypoints3d_temp[35];
         lenLeftLeg += sqrt((x_lank-x_lknee)*(x_lank-x_lknee)+(y_lank-y_lknee)*(y_lank-y_lknee)+(z_lank-z_lknee)*(z_lank-z_lknee));
         lenRightLeg += sqrt((x_rank-x_rknee)*(x_rank-x_rknee)+(y_rank-y_rknee)*(y_rank-y_rknee)+(z_rank-z_rknee)*(z_rank-z_rknee));
         lenLeftThigh += sqrt((x_lhip-x_lknee)*(x_lhip-x_lknee)+(y_lhip-y_lknee)*(y_lhip-y_lknee)+(z_lhip-z_lknee)*(z_lhip-z_lknee));
         lenRightThigh += sqrt((x_rhip-x_rknee)*(x_rhip-x_rknee)+(y_rhip-y_rknee)*(y_rhip-y_rknee)+(z_rhip-z_rknee)*(z_rhip-z_rknee));
         calcFrame++;
         if (calcFrame >= calcFrames)
         {
              ifCalculating = false;
              lenLeftLeg /= float(calcFrames+1);
              lenRightLeg /= float(calcFrames+1);
              lenLeftThigh /= float(calcFrames+1);
              lenRightThigh /= float(calcFrames+1);
         }
    }

    return 0;
}






int KeepPose::get_action_scores()
{
    TofApi *tof = reinterpret_cast<TofApi*>(tofApiPtr);
    if (NULL == tof)
    {
         return 0;
    }

    vector<PoseKeyPoint> key_pts(15);
    for (int q = 0; q < 15; q++) 
    {
         key_pts[q].idx = colorFrameID;
         key_pts[q].timestamp = colorTimestamp;
         key_pts[q].confidence = keypoints3d[q*4+2];
         key_pts[q].x = keypoints3d[q*4+3];
         key_pts[q].y = keypoints3d[q*4+4];
         key_pts[q].z = -keypoints3d[q*4+5];
    }
    acc[0] = 0;
    acc[1] = 9.8;
    acc[2] = 0;
    tof->detect(key_pts, acc);
    key_pts.clear();
    
    return 1;
}






