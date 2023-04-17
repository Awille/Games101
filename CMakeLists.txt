cmake_minimum_required(VERSION 3.24)
project(Games101)

set(CMAKE_CXX_STANDARD 17)
include_directories(E:/eigen/eigen-3.4.0)

add_executable(Games101 main.cpp Triangle.hpp Triangle.cpp rasterizer.hpp rasterizer.cpp global.hpp)

# 设置 OpenCV 安装路径
set(OpenCV_DIR "E:/opencv/opencv/mingw_build")  # 指向 OpenCV 安装目录下的 build/install 文件夹

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 添加 OpenCV 头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 打印 OpenCV 包含目录
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")


# 添加 OpenCV 库路径
link_directories(${OpenCV_LIBRARY_DIRS})
# 打印 OpenCV 包含目录
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_LIBRARY_DIRS}")

#add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

# 添加 OpenCV 库
target_link_libraries(Games101 ${OpenCV_LIBS})




