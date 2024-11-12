cmake_minimum_required(VERSION 3.10)

project(nanotrack)

# 指定交叉编译工具链
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# 设置 OpenCV 路径
set(OpenCV_DIR "/home/xcc/workcode/NanoTrack/opencv/build/install/lib/cmake/opencv4")

# 查找 OpenCV
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
endif()

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(rknpu_tracker_file nanotrack.cpp)

if (TARGET_SOC STREQUAL "rv1106" OR TARGET_SOC STREQUAL "rv1103")
    add_definitions(-DRV1106_1103)
endif()

if(TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    add_definitions(-DRKNPU1)
endif()

# 设置RKNN包含路径和库路径
set(RKNN_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rknpu2/include")
set(RKNN_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rknpu2/Linux/aarch64")

# 设置RGA包含路径
set(RGA_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/librga/include")

# 设置CMake的安装路径
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# 查找所有的源文件
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 添加可执行文件
add_executable(${PROJECT_NAME}
    main.cpp
    RKNNModel.cpp
    ${rknpu_tracker_file}
)

# 链接必要的库文件
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${RKNN_LIB_PATH}/librknnrt.so
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
        log
    )
endif()

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# 设置包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${RKNN_INCLUDE_PATH}
    ${RGA_INCLUDE_PATH}  # 添加 RGA 头文件目录
    ${OpenCV_INCLUDE_DIRS}
)

