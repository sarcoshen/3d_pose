#include <vector>
#include <stdio.h>
#include <YDPeopleSensor.h>
#include <tof_api.hpp>

// image size
const int Height = 480;
const int Width = 640;
// keypoint threshold
const float pointThd = 0.05;
const float pointThd2 = 0.2; // for 2d tracking
// YD depth boundary
const int padW = 1; //30; // depth image bound
const int padH = 1; //25; // depth image bound
// point from 2d to 3d
const float centerX = 319.5;
const float centerY = 239.5;
const float focusX = 521.3766;
const float focusY = 525.5348;
// keeppose mapping YD
const int poseMapping[14] = {0, 1, 3, 5, 7, 2, 4, 6, 11, 13, 15, 10, 12, 14};
// num action keypoints
const int numKeypoints = 15;
// depth boundary
const float minDepthBound = 1.0;
const float maxDepthBound = 4.0;
const int nearestR = 25;
// if smooth
const int useSmoothZ = 1;
// predict if using 4 frames
const int useFourFramesForPredict = 1;
// if ank,knee smooth
const int useSmoothKneeAnk = 1;
// if leg smooth
const int useSmoothLeg = 1;
// frameas of calc 3d leg & thigh
const int calcFrames = 10;
const int startFrameThd = 20;
// smooth all x,y,z
const int useAllSmooth = 0;
// if using 2d for display
const int use2dForDisplay = 1;



class KeepPose
{
public:
    long poseExtractorCaffePtr;
    long tofApiPtr;

public:
    KeepPose();
    ~KeepPose();
    int initialize();
    int get_frame();
    int get_2d_pose();
    int get_3d_pose();
    int get_action_scores();
    void predict_invalid_knee(std::vector<float> &keypoints3d_temp);
    void predict_invalid_ank(std::vector<float> &keypoints3d_temp);
    void predict_invalid_3d_knee(std::vector<float> &points3d, std::vector<float> &points2d);
    void predict_invalid_3d_ank(std::vector<float> &points3d, std::vector<float> &points2d);
    void jointindex_for_draw(std::vector<float> &temp_points, std::vector<int> &pose_pairs);

    float bbox_dist(float min_x, float min_y, float max_x, float max_y, float minx, float miny, float maxx, float maxy);
    void calcMinMaxPoint(std::vector<float> &temp_points,float &minx,float &miny,float &maxx,float &maxy);
    float mMinX;
    float mMinY;
    float mMaxX;
    float mMaxY;
    std::vector<float> keypoints2d_for_display;
    std::vector<int> pose_pair_list;
    std::vector<float> keypoints2d;
    std::vector<float> keypoints3d;
    long colorFrameID;
    long long colorTimestamp;
    int colorWidth;
    int colorHeight;  
    unsigned char *frameData;  // color frame

public:
    // yd
    YDPeopleSensor::Sensor sensor;
    YDPeopleSensor::ColorFrame color;
    YDPeopleSensor::DepthFrame depth;
    YDPeopleSensor::PublishData pubData;
    // frame
    int mFrame;
    // 3d pose
    int numPeople;
    int numParts;
    bool allNotZero;
    float prepreprepreZ[15];
    float preprepreZ[15];
    float prepreZ[15];
    float preZ[15];
    float prepreprepreX[15];
    float prepreprepreY[15];
    float preprepreX[15];
    float preprepreY[15];
    float prepreX[15];
    float prepreY[15];
    float preX[15];
    float preY[15];
    // calc 3d leg & thigh
    float lenLeftLeg;
    float lenRightLeg;
    float lenLeftThigh;
    float lenRightThigh;
    int calcFrame;
    int ifCalculating;
    // 2d pose
    bool allNotZero2d;
    float preprepreX2d[14];
    float preprepreY2d[14];
    float prepreX2d[14];
    float prepreY2d[14];
    float preX2d[14];
    float preY2d[14];
    // score
    ActionType act_type;
    ActionName act_name;
    std::string count_template_folder_path;
    BodyInfo body_info;
    float acc[3];

    // temp
    std::vector<std::string> image_paths;
    int i_image = 0;
};



