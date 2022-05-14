//
//  tof_api.hpp
//  ThreeDExercisesRecognition
//
//  Created by bo guo on 2018/9/20.
//  Copyright © 2018 bo guo. All rights reserved.
//
#ifndef TOF_API_HPP
#define TOF_API_HPP

#include <stdio.h>
#include "pose_detection.hpp"

//#define TRACE
//#define TRACE (printf("%s(%d)-<%s>: ",__FILE__, __LINE__, __FUNCTION__), printf)
#define TRACE printf
#define ACTION_COUNTS 100
//#define ONLINE

enum ActionType{
    StaticBodyInfo = 0,
    StatisticCount,
    StatisticTime,
};

class TofApi{
private:
    PoseDetection m_pose_det;
    std::string m_count_template_file_path;
    std::string m_score_template_file_path;
    std::string m_time_template_file_path;
    ActionType m_act_type;
    ActionName m_action_name;
    void setActionTemplatePath(const ActionName act_name, const std::string template_folder_path, std::vector<std::vector<std::string> > &count_template_path);
    
public:
    PoseStatus m_initial_status;
    int m_start_idx;
    int m_end_idx;
    int m_act_cnts;
    int m_time_cnts;
    int m_time_frame_cnts;
    long int m_frame_cnts;
    long int m_start_timestamp;
    long int m_cur_timestamp;
    float m_scores[ACTION_TYPE];
    std::vector<ActionResult> m_rst;
    std::vector<std::vector<PoseKeyPoint> > m_current_data;
    
public:
    TofApi(const ActionType act_type, const ActionName act_name, const BodyInfo body_info, const std::string count_template_folder_path);
    ~TofApi();
    void TofInit(const ActionType act_type, const ActionName act_name, const BodyInfo body_info, const std::string count_template_folder_path);
    PoseStatus getLoadedStatus(){return m_initial_status;}
    BodyInfo calcBodyInfo(const std::vector<PoseKeyPoint> body_pts, const float acc[3]);
    BodyInfo getBodyInfo(){return m_pose_det.m_body_info;}
    PoseStatus detect(const std::vector<PoseKeyPoint> body_pts, const float acc[3]);
    float* getScore(){return m_scores;}
};

#endif /* TOF_API_HPP */
