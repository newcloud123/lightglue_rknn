cmake_minimum_required(VERSION 3.10)
project(lightclue_rknn)

# ====================== 核心路径配置（已匹配你的实际目录） ======================
# OpenCV 根路径（确认无误）
set(OPENCV_ROOT_DIR "/userdata/luoshiyong/goodcode/jly_algorithm/3rdparty/opencv4")
# 关键：指向包含 opencv2 的上层目录（即 include/opencv4）
set(OPENCV_INCLUDE_DIR "${OPENCV_ROOT_DIR}/include/opencv4")
set(OPENCV_LIB_DIR "${OPENCV_ROOT_DIR}/lib")
# OpenCV 配置文件路径（share/OpenCV 或 share/opencv4，根据实际调整）
set(OpenCV_DIR "${OPENCV_ROOT_DIR}/share/OpenCV")

# RKNN 库路径（保持原有配置）
set(RKNN_API_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/deps/runtime/Linux/librknn_api")
set(RKNN_INCLUDE_DIR "${RKNN_API_ROOT}/include")
set(RKNN_LIB_DIR "${RKNN_API_ROOT}/aarch64")
# Eigen 库路径（头文件库，无需链接，仅需包含路径）
set(EIGEN_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen")
set(EIGEN_INCLUDE_DIR "${EIGEN_ROOT_DIR}")  # Eigen核心头文件直接在deps/eigen下（Eigen/Dense）

# ====================== 基础编译配置 ======================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fPIC")
# 动态库运行路径（确保运行时能找到 OpenCV/RKNN 库）
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=${OPENCV_LIB_DIR} -Wl,-rpath=${RKNN_LIB_DIR}")

# ====================== 严格路径检查（提前规避错误） ======================
# 检查 OpenCV 核心头文件是否存在
if(NOT EXISTS "${OPENCV_INCLUDE_DIR}/opencv2/opencv.hpp")
    message(FATAL_ERROR "❌ OpenCV 核心头文件缺失！
    期望路径：${OPENCV_INCLUDE_DIR}/opencv2/opencv.hpp
    实际检测：该文件不存在，请核对 OpenCV 安装路径！")
endif()
# 检查 Eigen 核心头文件是否存在（Eigen是头文件库，核心文件为Eigen/Dense）
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Dense")
    message(FATAL_ERROR "❌ Eigen 核心头文件缺失！
    期望路径：${EIGEN_INCLUDE_DIR}/Eigen/Dense
    实际检测：该文件不存在，请核对 Eigen 路径（deps/eigen）！")
else()
    message(STATUS "✅ Eigen 头文件路径验证成功：${EIGEN_INCLUDE_DIR}")
endif()
# 检查 OpenCV 库目录
if(NOT EXISTS "${OPENCV_LIB_DIR}")
    message(FATAL_ERROR "❌ OpenCV 库目录不存在：${OPENCV_LIB_DIR}")
endif()

# 检查 RKNN 库目录
if(NOT EXISTS "${RKNN_LIB_DIR}")
    message(FATAL_ERROR "❌ RKNN 库目录不存在：${RKNN_LIB_DIR}")
endif()

# ====================== 查找依赖库 ======================
# 1. 查找 OpenCV（指定自定义路径，避免系统默认）
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui
             PATHS ${OPENCV_ROOT_DIR} NO_DEFAULT_PATH)
if(OpenCV_FOUND)
    message(STATUS "✅ OpenCV 匹配成功！")
    message(STATUS "  - 版本：${OpenCV_VERSION}")
    message(STATUS "  - 头文件：${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  - 库文件：${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "❌ OpenCV 查找失败，请检查 OPENCV_ROOT_DIR！")
endif()

# 2. 查找 RKNN 运行时库
include_directories(SYSTEM ${RKNN_INCLUDE_DIR})
link_directories(${RKNN_LIB_DIR})
find_library(RKNNRT_LIB rknnrt PATHS ${RKNN_LIB_DIR} NO_DEFAULT_PATH)
if(NOT RKNNRT_LIB)
    message(FATAL_ERROR "❌ RKNN 库（rknnrt）未找到：${RKNN_LIB_DIR}")
else()
    message(STATUS "✅ RKNN 匹配成功：${RKNNRT_LIB}")
endif()

# ====================== 构建可执行文件 ======================
set(SOURCE_FILES infer.cpp)
add_executable(lightclue_infer ${SOURCE_FILES})

# 仅为目标程序添加头文件（现代 CMake 推荐）
target_include_directories(lightclue_infer SYSTEM PRIVATE
    ${OPENCV_INCLUDE_DIR}  # 核心：指向 include/opencv4
    ${RKNN_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}   # Eigen头文件目录（新增）
)

# 链接所有依赖库
target_link_libraries(lightclue_infer
    ${OpenCV_LIBS}          # OpenCV 自动匹配的库
    ${RKNNRT_LIB}           # RKNN 库
    pthread dl              # 系统依赖
    
)

# ====================== 安装（可选） ======================
install(TARGETS lightclue_infer 
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# ====================== 编译信息汇总 ======================
message(STATUS "====================== 编译配置汇总 ======================")
message(STATUS "OpenCV 根路径：${OPENCV_ROOT_DIR}")
message(STATUS "OpenCV 头文件：${OPENCV_INCLUDE_DIR}")
message(STATUS "OpenCV 库路径：${OPENCV_LIB_DIR}")
message(STATUS "RKNN 头文件：${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN 库路径：${RKNN_LIB_DIR}")
message(STATUS "==========================================================")