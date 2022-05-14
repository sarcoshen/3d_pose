//////////////////////////////////////////////////////////////////////////////////////////////
///                                                                                        ///
///                                     CONFIDENTIAL                                       ///
///                                                                                        ///
///                         Beijing Yundong Technology Co., Ltd.                           ///
///                                    Copyright 2018                                      ///
///                                                                                        ///
///                                 All rights reserved.                                   ///
///                                                                                        ///
/// If you get this file, that means your company have signed Mutual NDA with              ///
/// Beijing Yundong Technology Co., Ltd., any disclosure, publication or dissemination     ///
/// of Confidential Information or anything related with the technology of Beijing Yundong ///
/// Technology Co., Ltd. to a third party no matter in what kind of format will be treated ///
/// as the breach of the NDA and will be prosecuted.                                       ///
///                                                                                        ///
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _YD_PEOPLE_SENSOR_H_
#define _YD_PEOPLE_SENSOR_H_

#undef DLLFUNCTION
#ifdef _MSC_VER

#ifdef YDPEOPLESENSORDLL
#define DLLFUNCTION __declspec(dllexport)
#else
#define DLLFUNCTION __declspec(dllimport)
#endif //YDPEOPLESENSORDLL

#else //_MSC_VER

#ifdef YDPEOPLESENSORDLL
#define DLLFUNCTION __attribute__ ((visibility ("default")))
#else
#define DLLFUNCTION
#endif //YDPEOPLESENSORDLL

#endif

#undef _CDECL_
#if defined(_MSC_VER) & defined(__cplusplus)
#define _CDECL_ __cdecl
#else
#define _CDECL_
#endif

#define YDPEOPLESENSOR_SKELETON_LIMIT (6)

typedef void* YD_PEOPLE_SENSOR;
typedef unsigned char YDBOOL;
#define YDTRUE  (1)
#define YDFALSE (0)

typedef enum _YDPEOPLESENSOR_ERRORCODE
{
    YDPEOPLESENSOR_ERRORCODE_SUCCESS = 0,
    YDPEOPLESENSOR_ERRORCODE_NO_DEVICE = 100,
    YDPEOPLESENSOR_ERRORCODE_NO_PERMISSION,
    YDPEOPLESENSOR_ERRORCODE_DEVICE_UNSUPPORTED,
    YDPEOPLESENSOR_ERRORCODE_DEVICE_NOT_OPEN,
    YDPEOPLESENSOR_ERRORCODE_STREAM_UNSUPPORTED,
    YDPEOPLESENSOR_ERRORCODE_STREAM_DISABLED,
    YDPEOPLESENSOR_ERRORCODE_TIMEOUT
} YDPEOPLESENSOR_ERRORCODE;

typedef enum _YDPEOPLESENSOR_JOINTTYPE
{
    YDPEOPLESENSOR_JOINTTYPE_HEAD = 0,
    YDPEOPLESENSOR_JOINTTYPE_NECK,
    YDPEOPLESENSOR_JOINTTYPE_LSHOULDER,
    YDPEOPLESENSOR_JOINTTYPE_RSHOULDER,
    YDPEOPLESENSOR_JOINTTYPE_LELBOW,
    YDPEOPLESENSOR_JOINTTYPE_RELBOW,
    YDPEOPLESENSOR_JOINTTYPE_LHAND,
    YDPEOPLESENSOR_JOINTTYPE_RHAND,
    YDPEOPLESENSOR_JOINTTYPE_LCHEST,
    YDPEOPLESENSOR_JOINTTYPE_RCHEST,
    YDPEOPLESENSOR_JOINTTYPE_LWAIST,
    YDPEOPLESENSOR_JOINTTYPE_RWAIST,
    YDPEOPLESENSOR_JOINTTYPE_LKNEE,
    YDPEOPLESENSOR_JOINTTYPE_RKNEE,
    YDPEOPLESENSOR_JOINTTYPE_LFOOT,
    YDPEOPLESENSOR_JOINTTYPE_RFOOT,
    YDPEOPLESENSOR_JOINTTYPE_TORSO,
    YDPEOPLESENSOR_JOINTTYPE_COUNT
} YDPEOPLESENSOR_JOINTTYPE;

typedef enum _YDPEOPLESENSOR_RESOLUTION
{
    YDPEOPLESENSOR_RESOLUTION_QVGA,
    YDPEOPLESENSOR_RESOLUTION_VGA,
    YDPEOPLESENSOR_RESOLUTION_UVGA
} YDPEOPLESENSOR_RESOLUTION;

typedef struct _YDPEOPLESENSOR_FLOAT2
{
    float x;
    float y;
} YDPEOPLESENSOR_FLOAT2;

typedef struct _YDPEOPLESENSOR_FLOAT3
{
    float x;
    float y;
    float z;
} YDPEOPLESENSOR_FLOAT3;

typedef struct _YDPEOPLESNESOR_FLOAT4
{
    float x;
    float y;
    float z;
    float w;
} YDPEOPLESENSOR_FLOAT4;

typedef struct _YDPEOPLESENSOR_JOINT
{
    YDPEOPLESENSOR_FLOAT4 Position;
    YDPEOPLESENSOR_FLOAT4 Rotation;
    YDPEOPLESENSOR_FLOAT3 Orientation[3];
} YDPEOPLESENSOR_JOINT;

typedef struct _YDPEOPLESENSOR_SKELETON
{
    unsigned short        UserID;
    YDBOOL                IsTracked;
    YDPEOPLESENSOR_FLOAT4 Position;
    YDPEOPLESENSOR_JOINT  Joints[YDPEOPLESENSOR_JOINTTYPE_COUNT];
} YDPEOPLESENSOR_SKELETON;

typedef struct _YDPEOPLESENSOR_SKELETONS
{
    unsigned int Size;
    YDPEOPLESENSOR_SKELETON* Data;
} YDPEOPLESENSOR_SKELETONS;

typedef struct _YDPEOPLESENSOR_USERMASK
{
    unsigned short* Mask;
    unsigned int    Width;
    unsigned int    Height;
} YDPEOPLESENSOR_USERMASK;

typedef struct _YDPEOPLESENSOR_PUBLISHDATA
{
    YDPEOPLESENSOR_SKELETONS Skeletons;
    YDPEOPLESENSOR_FLOAT4    GroundModel;
    YDPEOPLESENSOR_USERMASK  UserMask;
    long long                Timestamp;      // 毫秒
} YDPEOPLESENSOR_PUBLISHDATA;

typedef struct _YDPEOPLESENSOR_COLORFRAME
{
    unsigned int Width;
    unsigned int Height;
    long FrameID;
    long long Timestamp; // 毫秒

    const unsigned int* Pixels;
} YDPEOPLESENSOR_COLORFRAME;

typedef struct _YDPEOPLESENSOR_DEPTHFRAME
{
    unsigned int Width;
    unsigned int Height;
    long FrameID;
    long long Timestamp; // 毫秒

    const unsigned short* Pixels;
} YDPEOPLESENSOR_DEPTHFRAME;

typedef struct _YDPEOPLESENSOR_AUDIOFRAME
{
    unsigned int Size;
    unsigned int Duration;
    long FrameID;
    long long Timestamp;

    const unsigned short* Data;
} YDPEOPLESENSOR_AUDIOFRAME;

#ifdef __cplusplus
extern "C"
{
#endif
    DLLFUNCTION YDBOOL _CDECL_ CreatePeopleSensor(YD_PEOPLE_SENSOR* pSensor);
    DLLFUNCTION void   _CDECL_ DestroyPeopleSensor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ InitializePeopleSensor(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_RESOLUTION colorResolution, YDPEOPLESENSOR_RESOLUTION depthResolution, YDBOOL useSkeleton, int sensorIndex);
    DLLFUNCTION void   _CDECL_ UninitializePeopleSensor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ StartPeopleSensor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION void   _CDECL_ StopPeopleSensor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION YDBOOL _CDECL_ IsPeopleSensorRunning(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION void   _CDECL_ SetPeopleSensorDepthMappedToColor(YD_PEOPLE_SENSOR sensor, YDBOOL isMapped);
    DLLFUNCTION YDBOOL _CDECL_ IsPeopleSensorDepthMappedToColor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION void   _CDECL_ SetPeopleSensorSkeletonMappedToColor(YD_PEOPLE_SENSOR sensor, YDBOOL isMapped);
    DLLFUNCTION YDBOOL _CDECL_ IsPeopleSensorSkeletonMappedToColor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ TurnOnPeopleSensorInfraredEmitter(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ TurnOffPeopleSensorInfraredEmitter(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ EnablePeopleSensorColor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ DisablePeopleSensorColor(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION YDBOOL _CDECL_ IsPeopleSensorColorEnabled(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ EnablePeopleSensorAudio(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ DisablePeopleSensorAudio(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION YDBOOL _CDECL_ IsPeopleSensorAudioEnabled(YD_PEOPLE_SENSOR sensor);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorColorFrame(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_COLORFRAME* frame);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorDepthFrame(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_DEPTHFRAME* frame);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorAudioFrame(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_AUDIOFRAME* frame);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorPublishData(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_PUBLISHDATA* pub);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorColorFrameTimeout(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_COLORFRAME* frame, unsigned int timeout);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorDepthFrameTimeout(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_DEPTHFRAME* frame, unsigned int timeout);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorAudioFrameTimeout(YD_PEOPLE_SENSOR sensor, YDPEOPLESENSOR_AUDIOFRAME* frame, unsigned int timeout);
    DLLFUNCTION int    _CDECL_ GetPeopleSensorPublishDataTimeout(YD_PEOPLE_SENSOR senosr, YDPEOPLESENSOR_PUBLISHDATA* pub, unsigned int timeout);
    DLLFUNCTION int    _CDECL_ PeopleSensorDepthSpacePointToScreen(YD_PEOPLE_SENSOR sensor, const YDPEOPLESENSOR_FLOAT4* point3D, YDPEOPLESENSOR_FLOAT2* point2D);
    DLLFUNCTION int    _CDECL_ PeopleSensorColorSpacePointToScreen(YD_PEOPLE_SENSOR sensor, const YDPEOPLESENSOR_FLOAT4* point3D, YDPEOPLESENSOR_FLOAT2* point2D);
    DLLFUNCTION void   _CDECL_ YDGreenScreen(const unsigned int* image, unsigned int imageWidth, unsigned int imageHeight, const unsigned short* mask, unsigned int maskWidth, unsigned int maskHeight, unsigned int* maskedImage);
#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
namespace YDPeopleSensor
{
    typedef YDPEOPLESENSOR_FLOAT2      FLOAT2;
    typedef YDPEOPLESENSOR_FLOAT3      FLOAT3;
    typedef YDPEOPLESENSOR_FLOAT4      FLOAT4;
    typedef YDPEOPLESENSOR_JOINT       Joint;
    typedef YDPEOPLESENSOR_SKELETON    Skeleton;
    typedef YDPEOPLESENSOR_SKELETONS   Skeletons;
    typedef YDPEOPLESENSOR_USERMASK    UserMask;
    typedef YDPEOPLESENSOR_PUBLISHDATA PublishData;
    typedef YDPEOPLESENSOR_COLORFRAME  ColorFrame;
    typedef YDPEOPLESENSOR_DEPTHFRAME  DepthFrame;
    typedef YDPEOPLESENSOR_AUDIOFRAME  AudioFrame;

    const int SKELETON_LIMIT = YDPEOPLESENSOR_SKELETON_LIMIT;

    enum class ErrorCode
    {
        Success           = YDPEOPLESENSOR_ERRORCODE_SUCCESS,
        NoDevice          = YDPEOPLESENSOR_ERRORCODE_NO_DEVICE,          // 未找到设备
        NoPermission      = YDPEOPLESENSOR_ERRORCODE_NO_PERMISSION,      // 未得到设备打开权限
        DeviceUnsupported = YDPEOPLESENSOR_ERRORCODE_DEVICE_UNSUPPORTED, // 不支持的图像设备
        DeviceNotOpen     = YDPEOPLESENSOR_ERRORCODE_DEVICE_NOT_OPEN,    // 设备未开启
        StreamUnsupported = YDPEOPLESENSOR_ERRORCODE_STREAM_UNSUPPORTED, // 不支持的数据流类型
        StreamDisabled    = YDPEOPLESENSOR_ERRORCODE_STREAM_DISABLED,    // 当前类型数据流未开启
        Timeout           = YDPEOPLESENSOR_ERRORCODE_TIMEOUT             // 获取数据超时    
    };

    enum class JointType
    {
        Head      = YDPEOPLESENSOR_JOINTTYPE_HEAD,
        Neck      = YDPEOPLESENSOR_JOINTTYPE_NECK,
        LShoulder = YDPEOPLESENSOR_JOINTTYPE_LSHOULDER,
        RShoulder = YDPEOPLESENSOR_JOINTTYPE_RSHOULDER,
        LElbow    = YDPEOPLESENSOR_JOINTTYPE_LELBOW,
        RElbow    = YDPEOPLESENSOR_JOINTTYPE_RELBOW,
        LHand     = YDPEOPLESENSOR_JOINTTYPE_LHAND,
        RHand     = YDPEOPLESENSOR_JOINTTYPE_RHAND,
        LChest    = YDPEOPLESENSOR_JOINTTYPE_LCHEST,
        RChest    = YDPEOPLESENSOR_JOINTTYPE_RCHEST,
        LWaist    = YDPEOPLESENSOR_JOINTTYPE_LWAIST,
        RWaist    = YDPEOPLESENSOR_JOINTTYPE_RWAIST,
        LKnee     = YDPEOPLESENSOR_JOINTTYPE_LKNEE,
        RKnee     = YDPEOPLESENSOR_JOINTTYPE_RKNEE,
        LFoot     = YDPEOPLESENSOR_JOINTTYPE_LFOOT,
        RFoot     = YDPEOPLESENSOR_JOINTTYPE_RFOOT,
        Torso     = YDPEOPLESENSOR_JOINTTYPE_TORSO,
        Count     = YDPEOPLESENSOR_JOINTTYPE_COUNT
    };

    enum class ColorResolution
    {
        QVGA = YDPEOPLESENSOR_RESOLUTION_QVGA,   // 320  * 240
        VGA  = YDPEOPLESENSOR_RESOLUTION_VGA,    // 640  * 480
        UVGA = YDPEOPLESENSOR_RESOLUTION_UVGA    // 1280 * 960
    };

    enum class DepthResolution
    {
        QVGA = YDPEOPLESENSOR_RESOLUTION_QVGA,   // 320 * 240
        VGA  = YDPEOPLESENSOR_RESOLUTION_VGA     // 640 * 480
    };

    class Sensor
    {
    public:
        Sensor()
        {
            CreatePeopleSensor(&this->sensor);
        }
        Sensor(const Sensor& other) = delete;
       ~Sensor()
        {
            DestroyPeopleSensor(this->sensor);
        }

        int  Initialize(ColorResolution colorResolution, DepthResolution depthResolution, bool useSkeleton, int sensorIndex = 0)
        {
            return InitializePeopleSensor(this->sensor, (YDPEOPLESENSOR_RESOLUTION)colorResolution, (YDPEOPLESENSOR_RESOLUTION)depthResolution, useSkeleton, sensorIndex);
        }
        void Uninitialize()
        {
            UninitializePeopleSensor(this->sensor);
        }
        int  Start()
        {
            return StartPeopleSensor(this->sensor);
        }
        void Stop()
        {
            StopPeopleSensor(this->sensor);
        }
        bool IsRunning() const
        {
            return IsPeopleSensorRunning(this->sensor) ? true : false;
        }
        void SetDepthMappedToColor(bool isMapped)
        {
            SetPeopleSensorDepthMappedToColor(this->sensor, isMapped);
        }
        bool IsDepthMappedToColor() const
        {
            return IsPeopleSensorDepthMappedToColor(this->sensor) ? true : false;
        }
        void SetSkeletonMappedToColor(bool isMapped)
        {
            SetPeopleSensorSkeletonMappedToColor(this->sensor, isMapped ? YDTRUE : YDFALSE);
        }
        bool IsSkeletonMappedToColor() const
        {
            return IsPeopleSensorSkeletonMappedToColor(this->sensor) ? true : false;
        }
        int  TurnOnInfraredEmitter()
        {
            return TurnOnPeopleSensorInfraredEmitter(this->sensor);
        }
        int  TurnOffInfraredEmitter()
        {
            return TurnOffPeopleSensorInfraredEmitter(this->sensor);
        }
        int  EnableColor()
        {
            return EnablePeopleSensorColor(this->sensor);
        }
        int  DisableColor()
        {
            return DisablePeopleSensorColor(this->sensor);
        }
        bool IsColorEnabled() const
        {
            return IsPeopleSensorColorEnabled(this->sensor) ? true : false;
        }
        int  EnableAudio()
        {
            return EnablePeopleSensorAudio(this->sensor);
        }
        int  DisableAudio()
        {
            return DisablePeopleSensorAudio(this->sensor);
        }
        bool IsAudioEnabled() const
        {
            return IsPeopleSensorAudioEnabled(this->sensor) ? true : false;
        }
        int  GetColorFrame(ColorFrame& frame)
        {
            return GetPeopleSensorColorFrame(this->sensor, &frame);
        }
        int  GetDepthFrame(DepthFrame& frame)
        {
            return GetPeopleSensorDepthFrame(this->sensor, &frame);
        }
        int  GetAudioFrame(AudioFrame& frame)
        {
            return GetPeopleSensorAudioFrame(this->sensor, &frame);
        }
        int  GetPublishData(PublishData& data)
        {
            return GetPeopleSensorPublishData(this->sensor, &data);
        }
        int  GetColorFrame(ColorFrame& frame, unsigned int timeoutMilliseconds)
        {
            return GetPeopleSensorColorFrameTimeout(this->sensor, &frame, timeoutMilliseconds);
        }
        int  GetDepthFrame(DepthFrame& frame, unsigned int timeoutMilliseconds)
        {
            return GetPeopleSensorDepthFrameTimeout(this->sensor, &frame, timeoutMilliseconds);
        }
        int  GetAudioFrame(AudioFrame& frame, unsigned int timeoutMilliseconds)
        {
            return GetPeopleSensorAudioFrameTimeout(this->sensor, &frame, timeoutMilliseconds);
        }
        int  GetPublishData(PublishData& data, unsigned int timeoutMilliseconds)
        {
            return GetPeopleSensorPublishDataTimeout(this->sensor, &data, timeoutMilliseconds);
        }
        int  DepthSpacePointToScreen(const FLOAT4& point3D, FLOAT2& point2D)
        {
            return PeopleSensorDepthSpacePointToScreen(this->sensor, &point3D, &point2D);
        }
        int  ColorSpacePointToScreen(const FLOAT4& point3D, FLOAT2& point2D)
        {
            return PeopleSensorColorSpacePointToScreen(this->sensor, &point3D, &point2D);
        }

        Sensor& operator=(const Sensor& other) = delete;

    private:
        YD_PEOPLE_SENSOR sensor;
    };
}
#endif

#undef DLLFUNCTION
#undef _CDECL_

#endif
