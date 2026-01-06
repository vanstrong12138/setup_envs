# BundleSDF 环境配置教程（适用于 NVIDIA 50 系列显卡）

[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-blue.svg?logo=ubuntu)](https://ubuntu.com/)

|[中文](https://github.com/vanstrong12138/setup_envs/blob/main/BundleSDF/DepolyBundleSDF_zh.md)|[English](https://github.com/vanstrong12138/setup_envs/blob/main/BundleSDF/DepolyBundleSDF.md)|

[BundleSDF](https://github.com/NVlabs/BundleSDF)是一个用于实时6D未知物体姿态估计与跟踪的模型。

## 安装anaconda/miniconda

从[官方网站](https://www.anaconda.com/download/success)下载anaconda/miniconda

1. 选择你需要的版本下载，或直接在命令行中使用 wget 下载：
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2025.12-1-Linux-x86_64.sh
```
![conda](imgs/image01.png)
网站页面是这样的
![condas_sh](imgs/image02.png)

1. 修改此 **`.sh`** 脚本的权限并运行它：

```bash
sudo chmod +x ./Anaconda3-2025.12-1-Linux-x86_64.sh

bash ./Anaconda3-2025.12-1-Linux-x86_64.sh
```

然后同意 Anaconda 服务条款并设置默认安装目录。

3. 配置你的 conda 环境，使其不会干扰原始系统环境。

我们将设置 conda 默认自动激活，但是又需要避免每次打开终端都需要手动退出 conda 环境

```bash
echo "conda deactivate" >> ~/.bashrc
```

## 配置你的 CUDA 环境

1. 下载并安装 NVIDIA 驱动程序

- 对于 NVIDIA 50 系列 GPU，为了兼容类似 FoundationPose 等较旧的项目，我们建议使用较旧版本的驱动，本教程使用**驱动版本：570.195.03** 和 **CUDA 版本：12.8**

![drivers](imgs/image03.png)

- 重启

2. 下载并安装 CUDA

- 我们可以在[CUDA归档网站]((https://developer.nvidia.com/cuda-toolkit-archive)) 找到所有版本，这里我们选择CUDA 12.8

![cuda](imgs/image04.png)

- 设置环境变量

```bash
vim ~/.bashrc
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"
```

3. 下载并安装 cuDNN

- 下载 Tarball 文件：

```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.17.1.4_cuda12-archive.tar.xz
```

![libcudnn](imgs/image05.png)

- 解压并安装：

```bash
tar -xvf cudnn-linux-x86_64-9.17.1.4_cuda12-archive.tar.xz
cd ./cudnn-linux-x86_64-9.16.0.29_cuda12-archive
sudo cp ./lib/libcudnn* /usr/local/cuda/lib64
sudo cp ./include/cudnn*.h /usr/local/cuda/include
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

4. 下载并安装 TensorRT

- 我们可以在 [TensorRT归档网站](https://developer.nvidia.com/tensorrt/download/10x) 找到所有版本，这里我们安装与 CUDA 12 兼容的最新版本

![tensorrt](imgs/image06.png)

- 解压并移动文件：

```bash
tar -xvf TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9.tar.gz
cd TensorRT-10.14.1.48.Linux.x86_64-gnu.cuda-12.9/
sudo mv TensorRT-10.14.1.48/ /usr/local/
```

- 设置环境变量：

```bash
echo "export LD_LIBRARY_PATH="/usr/local/TensorRT-10.14.1.48/lib:$LD_LIBRARY_PATH"">>~/.bashrc
```

## 配置 BundleSDF 环境

```bash
# 创建新的 conda 环境
conda create -n bundlesdf python=3.10

# 激活环境
conda activate bundlesdf

pip install pyyaml typeguard resolver

# 安装 PyTorch (与 CUDA 12.8 兼容)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 安装依赖的 Python 库
pip install albumentations ray einops kornia loguru yacs tqdm autopep8 jupyterlab matplotlib pytorch-lightning joblib h5py trimesh wandb matplotlib imageio tqdm open3d ruamel.yaml sacred kornia pymongo pyrender jupyterlab scipy scikit-learn yacs einops transformations xatlas pymeshlab cython dearpygui

# 确保 numpy 版本兼容性
pip install --upgrade numpy scipy

# 卸载opencv-python
pip uninstall opencv-python

```

接下来通过conda配置虚拟cpp环境（尽管大部分用户使用conda都用来管理python虚拟环境，但是conda确实也能有效管理cpp环境,后续部分将会详细介绍这一点）

在虚拟环境安装需要的cpp库
```bash
conda install -c conda-forge \
    gcc=14 gxx=14 boost=1.74 eigen=5.0.1 cmake ninja make gxx_linux-64\
    sysroot_linux-64 libdc1394 \
    yaml-cpp pybind11 zeromq cppzmq jsoncpp \
    freeglut glew mesa-libgl-devel-cos7-x86_64 \
    libblas libcblas liblapacke cudnn mesa-libgl-cos7-x86_64 \ 
    mesa-dri-drivers-cos7-x86_64 ffmpeg gstreamer gst-plugins-base \
    gst-plugins-good gst-plugins-bad gst-plugins-ugly jpeg libtiff \
    libpng openjpeg webp libopenjp2 openh264 glib flann -y

conda install protobuf
```

在虚拟环境中设置环境变量，使虚拟环境使用主机环境的cuda（conda中使用cuda cpp的方法）

```bash
conda activate bundlesdf

export CUDA_HOME=/usr/local/cuda-12.8
export FORCE_CUDA=1

```

- 然后编译安装kaolin到虚拟环境
```bash
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
# 只执行编译
python setup.py build_ext --inplace
# 只安装不再编译
pip install . --no-build-isolation
```

如果直接通过`pip install .`来安装，则会出现下面的情况:

![error in ZH](imgs/image07_EN.png)

所以我的解决办法是上面的分别编译与安装的方法

如果我用上面分别编译与安装的方法安装kaolin，则正常安装

- 然后编译安装PyTorch3D到虚拟环境

```bash
git clone -b stable https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
# 只执行编译
python setup.py build_ext --inplace
# 只安装不再编译
pip install . --no-build-isolation
```

- 编译安装opencv到虚拟环境

下载opencv-4.11以及相应版本的opencv-contrib

![opencv](imgs/image08.png)

以及[opencv-contirb](https://github.com/opencv/opencv_contrib/tree/4.11.0)
![opencv-contrib](imgs/image09.png)

```bash
cd opencv
# 注意你的opencv-contrib的路径
git clone -b 4.11.0 https://github.com/opencv/opencv_contrib.git
```

设置环境变量使虚拟环境能够使用主机环境的cuda以及虚拟环境的gcc和g++

```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export CUDA_HOME=/usr/local/cuda-12.8
export CC=$(which x86_64-conda-linux-gnu-gcc)
export CXX=$(which x86_64-conda-linux-gnu-g++)

cmake ..  \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_CUDA_STUBS=OFF \
-DBUILD_DOCS=OFF \
-DWITH_MATLAB=OFF \
-Dopencv_dnn_BUILD_TORCH_IMPORTE=OFF \
-DCUDA_FAST_MATH=ON \
-DMKL_WITH_OPENMP=ON \
-DOPENCV_ENABLE_NONFREE=ON \
-DWITH_OPENMP=ON \
-DWITH_QT=ON -DWITH_OPENEXR=ON \
-DENABLE_PRECOMPILED_HEADERS=OFF \
-DBUILD_opencv_cudacodec=OFF \
-DINSTALL_PYTHON_EXAMPLES=OFF \
-DWITH_TIFF=OFF \
-DWITH_WEBP=OFF \
-DWITH_FFMPEG=ON \
-DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
-DCMAKE_CXX_FLAGS=-std=c++17 \
-DENABLE_CXX11=OFF \
-DBUILD_opencv_xfeatures2d=OFF \
-DOPENCV_DNN_OPENCL=OFF \
-DWITH_CUDA=ON \
-DWITH_OPENCL=OFF \
-DBUILD_opencv_wechat_qrcode=OFF \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_STANDARD_REQUIRED=ON \
-DOPENCV_CUDA_OPTIONS_opencv_test_cudev=-std=c++17 \
-DCUDA_ARCH_BIN="8.9" \
-DINSTALL_PKGCONFIG=ON \
-DOPENCV_GENERATE_PKGCONFIG=ON \
-DINSTALL_PYTHON_EXAMPLES=OFF \
-DINSTALL_C_EXAMPLES=OFF \
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
-DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
-DCMAKE_INCLUDE_PATH=$CONDA_PREFIX/include \
-DCMAKE_LIBRARY_PATH=$CONDA_PREFIX/lib  

make -j32

make install 
```

- 编译安装pcl1.10.0到虚拟环境

```bash
git clone -b pcl-1.10.0 https://github.com/PointCloudLibrary/pcl.git
```

```bash
cd pcl
mkdir build 
cd build

cmake .. \
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
-DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_apps=OFF \
-DBUILD_GPU=OFF \
-DBUILD_CUDA=OFF \
-DBUILD_examples=OFF \
-DBUILD_global_tests=OFF \
-DBUILD_simulation=OFF \
-DCUDA_BUILD_EMULATION=OFF \
-DCMAKE_CXX_FLAGS="-std=c++14 -include cassert" \
-DPCL_ENABLE_SSE=ON \
-DPCL_SHARED_LIBS=ON \
-DWITH_VTK=OFF \
-DPCL_ONLY_CORE_POINT_TYPES=ON \
-DPCL_COMMON_WARNINGS=OFF \
-DBOOST_ROOT=$CONDA_PREFIX \
-DBoost_NO_SYSTEM_PATHS=ON \
-DBoost_NO_BOOST_CMAKE=ON \
-DBoost_DEBUG=ON \
-DCMAKE_C_COMPILER=$(which x86_64-conda-linux-gnu-gcc) \
-DCMAKE_CXX_COMPILER=$(which x86_64-conda-linux-gnu-g++) \
-DFLANN_ROOT=$CONDA_PREFIX \
-DEIGEN_ROOT=$CONDA_PREFIX \
-DWITH_OPENNI=OFF \
-DWITH_LIBUSB=OFF \
-DWITH_PNG=OFF \
-DWITH_QHULL=OFF \
-DWITH_PCAP=OFF

make -j32

make install

```

- 构建`mycuda`

```bash
cd your_BundleSDF_path
cd BundleSDF/
ROOT=$(pwd)

# Set PyTorch library path
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"
export TORCH_LIBRARIES="/usr/local/lib/python3.10/dist-packages/torch/lib"

# Additional PyTorch environment variables
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"

# Ensure PyTorch can be found
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"

# Print debug info
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Testing PyTorch import..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

```bash
cd mycuda

python setup.py build_ext --inplace

pip install . --no-build-isolation
```

- 构建`BundleTrack`

修改`CmakeLists.txt`为：

<details>

<summary>展开CMakeLists.txt</summary>

```bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
cmake_minimum_required(VERSION 3.5)
project(BundleTrack LANGUAGES CUDA CXX C)

set(Python3_ROOT_DIR $ENV{CONDA_PREFIX})
set(Python_ROOT_DIR $ENV{CONDA_PREFIX})
set(Python3_FIND_REGISTRY NEVER)
set(Python3_FIND_STRATEGY LOCATION)
set(Python_FIND_REGISTRY NEVER)
set(Python_FIND_STRATEGY LOCATION)
include_directories(BEFORE $ENV{CONDA_PREFIX}/include)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set output directories
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Enable CUDA support
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

# Set C++17 standard once
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Base flags with warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --default-stream per-thread")

# Check CUDA version for architecture compatibility
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "11.0")
    message(WARNING "CUDA version < 11.0 may not support all specified architectures")
endif()

# Set CUDA architectures based on CUDA version
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.8")
    set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90")
else()
    set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89")
endif()

# Feature flags
add_definitions(-DG2O=0)
add_definitions(-DTIMER=0)
add_definitions(-DPRINT_RESIDUALS_DENSE=0)
add_definitions(-DPRINT_RESIDUALS_SPARSE=0)
add_definitions(-DCUDA_RANSAC=1)
add_definitions(-DCUDA_MATCHING=1)

# RPATH settings
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Find required packages with version specifications where appropriate
find_package(Boost 1.65.0 REQUIRED COMPONENTS system program_options serialization)
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(JSONCPP REQUIRED jsoncpp)

find_package(PCL 1.10 REQUIRED COMPONENTS common io filters registration features segmentation kdtree search visualization)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.11 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp 0.6 REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA 12.8 REQUIRED)
find_package(GLUT REQUIRED)
find_package(LAPACK REQUIRED)
# find_package(pybind11 2.6 REQUIRED)
# find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 验证Python是否来自Conda环境
message(STATUS "Python interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Python includes: ${Python3_INCLUDE_DIRS}")

set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARIES ${Python3_LIBRARIES})

# 然后查找pybind11（它会使用已找到的Python）
find_package(pybind11 2.6 REQUIRED)
find_package(ZeroMQ QUIET)
if(NOT ZeroMQ_FOUND)
    find_library(ZMQ_LIB zmq)
    if(NOT ZMQ_LIB)
        message(FATAL_ERROR "ZeroMQ library not found")
    endif()
endif()

set(PYBIND11_CPP_STANDARD -std=c++17)

# Include directories
include_directories(
    src
    ${Boost_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${CSPARSE_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/cuda/
    ${PROJECT_SOURCE_DIR}/src/cuda/Solver/
    ${PROJECT_SOURCE_DIR}/src/Thirdparty
    ${PROJECT_SOURCE_DIR}/src/Thirdparty/g2o
)

# Optional Gurobi support
if(DEFINED GUROBI)
    message("Using Gurobi")
    add_definitions(-DGUROBI=1)
    if(NOT DEFINED GUROBI_HOME)
        message(FATAL_ERROR "GUROBI_HOME must be defined when using Gurobi")
    endif()
    include_directories("${GUROBI_HOME}/include")
    find_library(GUROBI_LIBRARY
        NAMES gurobi90 gurobi95
        PATHS "${GUROBI_HOME}/lib"
        REQUIRED
    )
    find_library(GUROBI_CXX_LIBRARY
        NAMES gurobi_c++
        PATHS "${GUROBI_HOME}/lib"
        REQUIRED
    )
endif()

# Source files
file(GLOB MY_SRC 
    ${PROJECT_SOURCE_DIR}/src/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/cuda/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cpp
)

file(GLOB G2O_LIBS ${PROJECT_SOURCE_DIR}/src/Thirdparty/g2o/lib/libg2o*)

# Remove problematic PCL components if present
list(REMOVE_ITEM PCL_LIBRARIES pcl_simulation)

# CUDA files
file(GLOB CUDA_FILES
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/src/cuda/*.cu"
    "${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cu"
)

# Build CUDA library
add_library(MY_CUDA_LIB SHARED ${CUDA_FILES})
set_target_properties(MY_CUDA_LIB PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
target_link_libraries(MY_CUDA_LIB PUBLIC
    yaml-cpp
    ${YAML_CPP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Main library
add_library(${PROJECT_NAME} SHARED ${MY_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${YAML_CPP_LIBRARIES}
    MY_CUDA_LIB
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${G2O_LIBS}
    zmq
)

if(DEFINED GUROBI)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${GUROBI_LIBRARY} ${GUROBI_CXX_LIBRARY})
endif()

# Python bindings
pybind11_add_module(my_cpp 
    pybind_interface/pybind_api.cpp
    src/Frame.cpp
)
set_target_properties(my_cpp PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(my_cpp PRIVATE 
    ${PROJECT_NAME}
    MY_CUDA_LIB
    yaml-cpp
    ${YAML_CPP_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${PYTHON_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    zmq
)

```
</details>

然后编译

```bash
cd BundleTrack/
rm -rf build && mkdir build && cd build && cmake .. && make -j32
```

## 运行

- 准备示例数据

下载[example milk data](https://drive.google.com/file/d/1akutk_Vay5zJRMr3hVzZ7s69GT4gxuWN/view)

- 下载欲训练权重

下载[Xmem权重](https://drive.google.com/file/d/1MEZvjbBdNAOF7pXcq6XPQduHeXB50VTc/view)并放到`./BundleTrack/XMem/saves/XMem-s012.pth`

下载[LoFTR_out_door_ds.ckpt](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp)并放到`./BundleTrack/LoFTR/weights/outdoor_ds.ckpt`

- 执行指令

```bash
# 注意文件路径
python run_custom.py --mode run_video --video_dir /home/agilex/non_ros_workspace/BundleSDF/dataset/2022-11-18-15-10-24_milk --out_folder /home/agilex/non_ros_workspace/BundleSDF/result/bundlesdf_2022-11-18-15-10-24_milk --use_segmenter 1 --use_gui 1 --debug_level 2
```

![result](imgs/image10.png)
