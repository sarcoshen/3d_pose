#ifndef POSE_DETECTION_HPP
#define POSE_DETECTION_HPP
#include <string>
#include <vector>
#include <math.h>
#include <cmath>

//#define TRACE
//#define TRACE (printf("%s(%d)-<%s>: ",__FILE__, __LINE__, __FUNCTION__), printf)
#define TRACE printf

#define KEYPOINTSCNT 15
#define ACTIONFRAME 200
#define FRAME_RATE 30
#define KEYPOINTVERSION 001
#define INVALID_VALUE1 -1
#define DIST_TEMPLATE_ROW 48
#define STANDARD_HEIGHT 100
#define LOWEST_CONFIDENCE 0.1
#define CRUS_THIGH_RATE 0.86
#define LEG_HEIGHT_RATE 0.442
#define ACTION_TYPE 10

#define GRADE_1 0
#define GRADE_2 1
#define GRADE_3 2
#define GRADE_4 3
#define GRADE_5 4

#define HEAD 0
#define NECK 1
#define R_SHO 2
#define R_ELBOW 3
#define R_WRIST 4
#define L_SHO 5
#define L_ELBOW 6
#define L_WRIST 7
#define R_HIP 8
#define R_KNEE 9
#define R_ANKLE 10
#define L_HIP 11
#define L_KNEE 12
#define L_ANKLE 13
#define HIP_MID 14

#define C_STATUS_COUNT 50

typedef struct{
    long idx;
    long long timestamp;
    float confidence;
    float x;
    float y;
    float z;
}PoseKeyPoint, *pPoseKeyPoint;
typedef struct{
    float y_min[KEYPOINTSCNT][KEYPOINTSCNT];
    float y_max[KEYPOINTSCNT][KEYPOINTSCNT];
    float y_ideal[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_min[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_max[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_ideal[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_min[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_max[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_ideal[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_rate_min[KEYPOINTSCNT];
    float dist_rate_max[KEYPOINTSCNT];
    float dist_rate_ideal[KEYPOINTSCNT];
}KeyPointTemplate, *pKeyPointTemplate;

typedef struct{
    float y_angles[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_angles[KEYPOINTSCNT][KEYPOINTSCNT];
    float v_dists[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_rate[KEYPOINTSCNT];
}KeyPointFeature, *pKeyPointFeature;

typedef struct{
    int start_idx;
    int mid_idx1;
    int mid_idx2;
    int mid_idx3;
    int end_idx;
}ActionIdx, *pActionIdx;
typedef struct{
    ActionIdx frame;
    float score;
}ActionResult, *pActionResult;
typedef struct{
    float min;
    float max;
    float thv;
}ActionRange, *pActionRange;

typedef struct{
    float height;
    float width;
    float ground_y;
}BodyInfo, *pBodyInfo;

enum PoseStatus{
    Success = 0,
    Failure,
    Aborted,
};

enum ActionFlag{
    PreStart = 0,
    Start,
    Middle1,
    Middle2,
    Middle3,
    End,
    Idle,
};

enum ActionName{
    Squat = 0,                              //深蹲
    HalfSquat,                              //半蹲
    BasicSquatWithSideLegLift,              //半蹲侧抬腿
    SquatWithKneeLiftAndTwist,              //深蹲提膝
    QuadStretchLeft,                        //左腿前侧拉伸
    QuadStretchRight,                       //右腿前侧拉伸
    JumpingJack,                            //开合跳
    LateralJumpingJack,                     //侧向开合跳
    ScissorsHopwithClap,                    //合掌跳
    PulseJumpSquat,                         //缓冲深蹲跳
    JumpingJackSquatwithArmRaise,           //缓冲深蹲
    CossackSquat,                           //哥萨克下蹲
    ScissorHops,                            //前后交叉小跳
};

enum PoseName{
    squat_01 = 0,
    squat_02,
    halfSquat_02,
    basicSquatWithSideLegLift_03_L,
    basicSquatWithSideLegLift_03_R,
    squatWithKneeLiftAndTwist_03_L,
    squatWithKneeLiftAndTwist_03_R,
    quadStretchLeft_01,
    quadStretchRight_01,
    jumpJack_01,
    jumpJack_02,
    scissorsHopwithClap_01_L,
    scissorsHopwithClap_01_R,
    scissorsHopwithClap_02_R,
    scissorsHopwithClap_02_L,
    pulseJumpSquat_03,
    jumpingJackSquatwithArmRaise_02,
    cossackSquat_L,
    cossackSquat_R,
    scissorHops_L,
    scissorHops_R,
};

enum DeepSquatErrorType{
    score = 0,
    TooLight,               //4
    TooDeep,                //4
    TooNarrow,              //5
    KneeIn,                 //5
    KneeOut,                //5
    OverTiptoe,             //3
    TooFast,                //2
    TooSlow,                //2
};

class PoseDetection{
public:
    int m_status_count;               //add force reset based on tiem.
    int m_template_num;               //-1:idle; 0:first; 1:second;
    ActionFlag m_action_flag;
    ActionFlag m_pre_action_flag;
    ActionRange m_elapsed_time;
    ActionIdx m_act_idx;
    BodyInfo m_body_info;
    KeyPointFeature m_feature_info;
    std::vector<KeyPointFeature> m_feature_arr;
    std::vector< std::vector<KeyPointTemplate> > m_count_templates;
    float m_time_gap[13][2]; //please refer to tof_api.hpp ActionName.
    
public:
    PoseDetection();
    PoseStatus loadTemplate(const std::string template_path, KeyPointTemplate &act_template);
    void initTemplate(KeyPointTemplate &act_template);
    PoseStatus detectPose();
    void getPoseFeatures(const std::vector<PoseKeyPoint> body_pts, KeyPointFeature &feature_info);
    PoseStatus statisticCounts(std::vector<std::vector<PoseKeyPoint> > &current_data, std::vector<ActionResult> &rst);
    PoseStatus statisticTime(std::vector<std::vector<PoseKeyPoint> > &current_data, int &cnts);
    PoseStatus updateBodyInfo(std::vector<std::vector<PoseKeyPoint> > frames);
    PoseStatus preprocessingKeyPoints(std::vector<PoseKeyPoint> &key_pts, const float acc[3]);
    void calcDynamicScore(const ActionName act_name, const std::vector<std::vector<PoseKeyPoint> > &act_seq, const ActionIdx key_idx, float scores[ACTION_TYPE]);
    float getTimeScore(const int valid_cnts, const int total_cnts);
    float calculateDist(const PoseKeyPoint p1, const PoseKeyPoint p2){
        float dist = -1;
        if (p1.confidence > LOWEST_CONFIDENCE && p2.confidence > LOWEST_CONFIDENCE) {
            dist = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z));
        }
        return dist;
    }
    
private:
    int calculateAngle(const PoseKeyPoint p1, const PoseKeyPoint p2, const PoseKeyPoint p3, const PoseKeyPoint p4);
    void findOptimalIdx(const int st_idx, const int end_idx, const int key_idx, const float scale, int &optimal_idx);
    
};
#endif
#ifndef POSE_DETECTION_HPP
#define POSE_DETECTION_HPP
#include <string>
#include <vector>
#include <math.h>
#include <cmath>

//#define TRACE
//#define TRACE (printf("%s(%d)-<%s>: ",__FILE__, __LINE__, __FUNCTION__), printf)
#define TRACE printf

#define KEYPOINTSCNT 15
#define ACTIONFRAME 200
#define FRAME_RATE 30
#define KEYPOINTVERSION 001
#define INVALID_VALUE1 -1
#define DIST_TEMPLATE_ROW 48
#define STANDARD_HEIGHT 100
#define LOWEST_CONFIDENCE 0.1
#define CRUS_THIGH_RATE 0.86
#define LEG_HEIGHT_RATE 0.442
#define ACTION_TYPE 10

#define GRADE_1 1
#define GRADE_2 2
#define GRADE_3 3
#define GRADE_4 4
#define GRADE_5 5

#define HEAD 0
#define NECK 1
#define R_SHO 2
#define R_ELBOW 3
#define R_WRIST 4
#define L_SHO 5
#define L_ELBOW 6
#define L_WRIST 7
#define R_HIP 8
#define R_KNEE 9
#define R_ANKLE 10
#define L_HIP 11
#define L_KNEE 12
#define L_ANKLE 13
#define HIP_MID 14

#define C_STATUS_COUNT 50

typedef struct{
    long idx;
    long long timestamp;
    float confidence;
    float x;
    float y;
    float z;
}PoseKeyPoint, *pPoseKeyPoint;
typedef struct{
    float y_min[KEYPOINTSCNT][KEYPOINTSCNT];
    float y_max[KEYPOINTSCNT][KEYPOINTSCNT];
    float y_ideal[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_min[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_max[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_ideal[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_min[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_max[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_ideal[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_rate_min[KEYPOINTSCNT];
    float dist_rate_max[KEYPOINTSCNT];
    float dist_rate_ideal[KEYPOINTSCNT];
}KeyPointTemplate, *pKeyPointTemplate;

typedef struct{
    float y_angles[KEYPOINTSCNT][KEYPOINTSCNT];
    float z_angles[KEYPOINTSCNT][KEYPOINTSCNT];
    float v_dists[KEYPOINTSCNT][KEYPOINTSCNT];
    float dist_rate[KEYPOINTSCNT];
}KeyPointFeature, *pKeyPointFeature;

typedef struct{
    int start_idx;
    int mid_idx1;
    int mid_idx2;
    int mid_idx3;
    int end_idx;
}ActionIdx, *pActionIdx;
typedef struct{
    ActionIdx frame;
    float score;
}ActionResult, *pActionResult;
typedef struct{
    float min;
    float max;
    float thv;
}ActionRange, *pActionRange;

typedef struct{
    float height;
    float width;
    float ground_y;
}BodyInfo, *pBodyInfo;
enum PoseStatus{
    Success = 0,
    Failure,
    Aborted,
};

enum ActionFlag{
    PreStart = 0,
    Start,
    Middle1,
    Middle2,
    Middle3,
    End,
    Idle,
};

enum ActionName{
    Squat = 0,                              //深蹲
    HalfSquat,                              //半蹲
    BasicSquatWithSideLegLift,              //半蹲侧抬腿
    SquatWithKneeLiftAndTwist,              //深蹲提膝
    QuadStretchLeft,                        //左腿前侧拉伸
    QuadStretchRight,                       //右腿前侧拉伸
    JumpingJack,                            //开合跳
    LateralJumpingJack,                     //侧向开合跳
    ScissorsHopwithClap,                    //合掌跳
    PulseJumpSquat,                         //缓冲深蹲跳
    JumpingJackSquatwithArmRaise,           //缓冲深蹲
    CossackSquat,                           //哥萨克下蹲
    ScissorHops,                            //前后交叉小跳
};

enum PoseName{
    squat_01 = 0,
    squat_02,
    halfSquat_02,
    basicSquatWithSideLegLift_03_L,
    basicSquatWithSideLegLift_03_R,
    squatWithKneeLiftAndTwist_03_L,
    squatWithKneeLiftAndTwist_03_R,
    quadStretchLeft_01,
    quadStretchRight_01,
    jumpJack_01,
    jumpJack_02,
    scissorsHopwithClap_01_L,
    scissorsHopwithClap_01_R,
    scissorsHopwithClap_02_R,
    scissorsHopwithClap_02_L,
    pulseJumpSquat_03,
    jumpingJackSquatwithArmRaise_02,
    cossackSquat_L,
    cossackSquat_R,
    scissorHops_L,
    scissorHops_R,
};

enum DeepSquatErrorType{
    score = 0,
    TooLight,               //4
    TooDeep,                //4
    TooNarrow,              //5
    KneeIn,                 //5
    KneeOut,                //5
    OverTiptoe,             //3
    TooFast,                //2
    TooSlow,                //2
};

class PoseDetection{
public:
    int m_status_count;               //add force reset based on tiem.
    int m_template_num;               //-1:idle; 0:first; 1:second;
    ActionFlag m_action_flag;
    ActionFlag m_pre_action_flag;
    ActionRange m_elapsed_time;
    ActionIdx m_act_idx;
    BodyInfo m_body_info;
    KeyPointFeature m_feature_info;
    std::vector<KeyPointFeature> m_feature_arr;
    std::vector< std::vector<KeyPointTemplate> > m_count_templates;
    float m_time_gap[13][2]; //please refer to tof_api.hpp ActionName.
    
public:
    PoseDetection();
    PoseStatus loadTemplate(const std::string template_path, KeyPointTemplate &act_template);
    void initTemplate(KeyPointTemplate &act_template);
    PoseStatus detectPose();
    void getPoseFeatures(const std::vector<PoseKeyPoint> body_pts, KeyPointFeature &feature_info);
    PoseStatus statisticCounts(std::vector<std::vector<PoseKeyPoint> > &current_data, std::vector<ActionResult> &rst);
    PoseStatus statisticTime(std::vector<std::vector<PoseKeyPoint> > &current_data, int &cnts);
    PoseStatus updateBodyInfo(std::vector<std::vector<PoseKeyPoint> > frames);
    PoseStatus preprocessingKeyPoints(std::vector<PoseKeyPoint> &key_pts, const float acc[3]);
    void calcDynamicScore(const ActionName act_name, const std::vector<std::vector<PoseKeyPoint> > &act_seq, const ActionIdx key_idx, float scores[ACTION_TYPE]);
    float getTimeScore(const int valid_cnts, const int total_cnts);
    float calculateDist(const PoseKeyPoint p1, const PoseKeyPoint p2){
        float dist = -1;
        if (p1.confidence > LOWEST_CONFIDENCE && p2.confidence > LOWEST_CONFIDENCE) {
            dist = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) + (p1.z - p2.z)*(p1.z - p2.z));
        }
        return dist;
    }
    
private:
    int calculateAngle(const PoseKeyPoint p1, const PoseKeyPoint p2, const PoseKeyPoint p3, const PoseKeyPoint p4);
    void findOptimalIdx(const int st_idx, const int end_idx, const int key_idx, const float scale, int &optimal_idx);
    
};
#endif
