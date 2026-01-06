# FoundationPose 环境配置教程（适用于 NVIDIA 50 系列显卡）
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04-blue.svg?logo=ubuntu)](https://ubuntu.com/)

|[中文](https://github.com/vanstrong12138/setup_envs/blob/main/FoundationPose/DeployFoundationPose_zh.md)|[English](https://github.com/vanstrong12138/setup_envs/blob/main/FoundationPose/DeployFoundationPose.md)|

此配置文档针对 NVIDIA 50 系列 GPU，包含一些特殊修改

[FoundationPose](https://github.com/NVlabs/FoundationPose)是一个用于 6D 物体姿态估计与跟踪的模型。

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

## 配置 FoundationPose 环境

```bash
# 创建新的 conda 环境
conda create -n foundationpose python=3.12

# 激活环境
conda activate foundationpose

# 安装基础依赖
pip install pyyaml typeguard resolver

# 安装 PyTorch (与 CUDA 12.8 兼容)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 安装依赖的 Python 库
pip install jupyterlab ipywidgets scipy numpy scikit-learn scikit-image ruamel.yaml ninja h5py numba pybind11 imageio opencv-contrib-python plotly open3d pyglet pysdf trimesh xatlas rtree pyrender pyOpenGL pyOpenGL_accelerate meshcat webdataset omegaconf pypng Panda3D simplejson bokeh roma seaborn pin openpyxl torchnet wandb colorama GPUtil imgaug xlsxwriter timm albumentations xatlas nodejs jupyterlab objaverse g4f ultralytics pycocotools py-spy pybullet videoio kornia einops transformations joblib warp-lang fvcore cython

# 确保 numpy 版本兼容性
pip install "numpy<2" --upgrade

# 通过 conda 安装 Eigen
conda install conda-forge::eigen=3.4.0
```

本地安装 nvdiffrast

```bash
conda activate foundationpose
cd FoundationPose
git clone https://github.com/NVlabs/nvdiffrast.git

cd nvdiffrast/
python setup.py build_ext --inplace
pip install . --no-build-isolation
```

本地安装 Kaolin
```bash
conda activate foundationpose
cd FoundationPose
git clone https://github.com/NVIDIAGameWorks/kaolin.git

cd kaolin

python setup.py build_ext --inplace

pip install . --no-build-isolation
```

本地安装 PyTorch3D
```bash
conda activate foundationpose
cd FoundationPose
git clone https://github.com/facebookresearch/pytorch3d.git

cd pytorch3d

python setup.py build_ext --inplace

pip install . --no-build-isolation
```

修改部分源代码以适配 Python 3.12

1. 修改`Utils.py`

```bash
cd FoundationPose
# 定位到第 46 行
# 原内容：import mycpp
# 修改为：
# import mycpp.build.mycpp as mycpp
# 可以使用 sed 命令快速修改
sed -i '46s/import mycpp/import mycpp.build.mycpp as mycpp/' Utils.py
```
2. 修改`bundsdf/mycuda/common.cu`

```bash
cd FoundationPose/bundsdf/mycuda/
# 定位到第 120, 162, 268 行
# 将 `.type()` 替换为 `.scalar_type()`
# 可以使用 sed 命令批量修改
sed -i '120s/\.type()/\.scalar_type()/' common.cu
sed -i '162s/\.type()/\.scalar_type()/' common.cu
sed -i '268s/\.type()/\.scalar_type()/' common.cu
```

编译mycpp

```bash
cd FoundationPose/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)
```

编译并安装 mycuda

```bash
cd FoundationPose/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python setup.py build_ext --inplace && \
pip install . --no-build-isolation
```

