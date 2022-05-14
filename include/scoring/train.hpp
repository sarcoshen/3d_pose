#ifndef TRAIN_HPP
#define TRAIN_HPP
#include "pose_detection.hpp"

void loadUserData(const std::string file_path, const char separator, std::vector<std::vector<PoseKeyPoint> > &user_data, std::vector<std::vector<float> > &acc_data);
void statisticsAllSamplePoseInfo(const std::vector<std::vector<PoseKeyPoint> > user_data, const std::vector<std::vector<float> > acc_data, const BodyInfo body_info, const int k1, const int k2, float rate_arr[6], int angle_arr[6]);
PoseStatus updateTemplateFile(const int angle_arr[6], const float rate_arr[6], const int k1, const int k2, const std::string template_file);
void statisticsL1Dist(std::vector<std::vector<PoseKeyPoint> > &act_seq, const float height);
void saveStatisticResultToCsv(std::string path_name, const int max_angle_arr[KEYPOINTSCNT],
                              const int min_angle_arr[KEYPOINTSCNT], const float point_offset_arr[KEYPOINTSCNT],
                              const float max_dist_offset[KEYPOINTSCNT][KEYPOINTSCNT],
                              const float min_dist_offset[KEYPOINTSCNT][KEYPOINTSCNT]);
void saveRawDataToCsv(std::string path_name, const int idx, std::vector<PoseKeyPoint> frame);
#endif
