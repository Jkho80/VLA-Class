# VLA-Class
模式识别课程 具身智能 操作文档

# 项目介绍
大家需要在我们提供的服务器上完成以下任务：
- 下载 RoboOrchardLab 项目并配置 SEM 训练与推理环境 （25分）
- 下载 RoboTwin 仿真器项目并部署环境 （25分）
- 使用 RoboTwin 来生成数据 （10分）
- 将 RoboTwin 生成的数据转换为 SEM 格式 （10分）
- 训练 SEM 模型 （CPU: 5 分，CUDA：10 分，多个GPU：15分）
- 将 SEM 模型部署到 RoboTwin 中并进行验证 （10分）

附加分：
- SEM 在 RoboTwin 仿真器中的分数 （10 分）
- 在 RoboTwin 仿真器验证中实现 RTC （10 分）

## 推荐配置
- 推荐使用 conda 创建 python=3.10 的虚拟环境
- 推荐使用 Cuda 11.8 版本，其次为 Cuda 12.1
- 推荐使用 `v0.2.0-release` 版本的 RoboOrchardLab 项目，因为 `v0.2.0-release` 版本是比较稳定的版本
- 推荐使用 `Challenge-Cup-2025` 版本的 RoboTwin 仿真器项目，或者 根据 SEM 推荐的 `e71140e9734e69686daa420a9be8b75a20ff4587` 版本

## 须知
> 下面大部分流程都可在官网查看，但存在些需要大家自行查找的文件并复制到对应路径或者额外配置的文件
> 本次课程使用的两个项目分别是 [RoboOrchardLab](https://github.com/HorizonRobotics/RoboOrchardLab) 和 [RoboTwin](https://github.com/robotwin-Platform/RoboTwin)

## 1. SEM 项目部分
### 1.1 下载 `RoboOrchardLab`
首先需要将 `RoboOrchardLab` 的项目下载到本地，具体可查看 [SEM 项目](https://github.com/HorizonRobotics/RoboOrchardLab/tree/v0.2.0-release/projects/sem/robotwin)
```bash
git clone https://github.com/HorizonRobotics/RoboOrchardLab.git
```

### 1.2 下载 GroundingDino 预训练模型

```bash
cd /path/to/RoboOrchardLab/projects/sem/robotwin

mkdir ckpt

# 这是用来下载 GroundingDino 预训练模型的脚本
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth -O ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth
# 这是用来重命名的程序
python tools/ckpt_rename.py ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth --output ./ckpt
```

### 1.3 下载 `RoboTwin` 后下载下面的额外python依赖库
该部分最好是在能够通过 `RoboTwin` 仿真器生成数据后再进行额外下载
```bash
# It is recommended to use CUDA 11.8.
torch==2.4.1
torchmetrics==1.6.1 
torchvision==0.19.1
transformers==4.49.0
lmdb==1.6.2 
safetensors==0.5.3 
accelerate==1.4.0 
diffusers==0.32.2 
timeout-decorator==0.5.0
requests==2.32.3 
h5py==3.13.0
```
### 1.4 数据转换
下面的脚本是用来将 `RoboTwin` 生成的数据格式转换为 SEM 模型能够读取的格式
```bash
cd path/to/robo_orchard_lab

# for the robotwin2.0 master branch before commit e71140e9734e69686daa420a9be8b75a20ff4587 or the Challenge-Cup-2025 branch
python robo_orchard_lab/dataset/robotwin/robotwin_packer.py \
    --input_path path/to/robotwin_data \
    --output_path "projects/sem/robotwin/data/lmdb" \
    --task_names ${task_names} \
    --config_name demo_clean
```

### 1.5 训练 SEM 模型
```bash
cd projects/sem/robotwin
CONFIG=config_sem_robotwin.py

# train with single-gpu
python3 train.py --config ${CONFIG}

# train with multi-gpu multi-machine
# example: 2 machines × 8 gpus
accelerate launch  \
    --num_machines 2 \
    --num-processes 16  \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7  \
    --machine_rank ${current_rank} \
    --main_process_ip ${main_process_ip} \
    --main_process_port 1227 \
    train.py \
    --workspace /job_data \
    --config ${CONFIG}
    
# RoboTwin 模型验证可以查看 ./sem_policy/README.md
```

## 2. RoboTwin 仿真器项目部分
### 2.1 下载 `RoboTwin`
```bash
git clone https://github.com/robotwin-Platform/RoboTwin.git
```
使用文档可参考：[RoboTwin Usage And Installation](https://robotwin-platform.github.io/doc/usage/index.html)

### 2.2 创建并配置 `RoboTwin` 虚拟环境
这是 RoboTwin 官方提供并推荐的python版本
```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
```

在下载 `RoboTwin` 项目后，需要额外下载下面的python依赖库

`RoboTwin` 官方为此提供了一个一键下载的依赖库的脚本
```bash
bash script/_install.sh
```

