cmake_minimum_required(VERSION 2.8.12)

PROJECT(KeepPoseScoreInterface_track)

include_directories(/usr/local/cuda/include /usr/include/hdf5/serial /workspace/KeepPoseScoreInterface_track/3rdparty/caffe/distribute/include /workspace/KeepPoseScoreInterface_track/include /workspace/KeepPoseScoreInterface_track /workspace/KeepPoseScoreInterface_track/include/scoring)

link_directories(/usr/lib /usr/local/lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /workspace/KeepPoseScoreInterface_track/3rdparty/caffe/build/lib /workspace/KeepPoseScoreInterface_track/lib)

set(NEED_SO libkeeppose.so libscoring.so libYDPeopleSensor.so libopenpose.so libcaffe.so libglog.so libgflags.so libopencv_imgproc.so libopencv_core.so libopencv_highgui.so)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

ADD_EXECUTABLE(main main.cpp)

TARGET_LINK_LIBRARIES(main ${NEED_SO})

