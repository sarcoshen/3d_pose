#ifndef updateEulerForAndroid_h
#define updateEulerForAndroid_h
#include "math.h"
// #define samplesCal 60*50

#ifdef __cplusplus
extern "C" {
#endif
    void updateEuler(float gx, float gy, float gz, float ax, float ay, float az, int startFlag,float sampleFreq, float *euler_1, float *euler_2, float *euler_3, float *quatern0, float *quatern1, float *quatern2, float *quatern3);
    void updateEulerbyAcc(float ax, float ay, float az,float *euler_1, float *euler_2, float *euler_3, float *quatern0, float *quatern1, float *quatern2, float *quatern3);
//    void euler2quatern_test(float roll, float pitch, float yaw, float qtest[4]);
    void quaternRotate(float v[3],float q[4], float vRotate[3]);
#ifdef __cplusplus
}
#endif

#endif
