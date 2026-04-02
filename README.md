# 项目介绍
你们需要完成以下任务（95 分）：
- 下载 RoboOrchardLab 项目并配置 SEM 训练与推理环境 （25 分）
- 下载 RoboTwin 仿真器项目并部署环境 （25 分）
- 使用 RoboTwin 来生成数据 （10 分）
- 将 RoboTwin 生成的数据转换为 SEM 格式 （10 分）
- 训练 SEM 模型 （CPU: 5 分; CUDA：10 分）
- 将 SEM 模型部署到 RoboTwin 中并进行验证 （15 分）

附加题：
- 训练出的 SEM 策略在 RoboTwin 仿真器中的分数接近甚至超过官方
- 能够在 RoboTwin 仿真器中配置新的任务
- 使用多卡来训练 SEM 模型
- 在 RoboTwin 仿真器验证中实现 RTC

## 推荐配置
- 推荐使用 conda 创建 python=3.10 的虚拟环境
- 推荐使用 Cuda 11.8 版本，其次为 Cuda 12.1
- 推荐使用 `v0.2.0-release` 版本的 RoboOrchardLab 项目，因为 `v0.2.0-release` 版本是比较稳定的版本
- 推荐使用 `Challenge-Cup-2025` 版本的 RoboTwin 仿真器项目，或者 根据 SEM 推荐的 `e71140e9734e69686daa420a9be8b75a20ff4587` 版本

## 须知
> 下面大部分流程都可在官网查看，但存在些需要大家自行查找的文件并复制到对应路径或者额外配置的文件
> 本次课程使用的两个项目分别是 [RoboOrchardLab](https://github.com/HorizonRobotics/RoboOrchardLab) 和 [RoboTwin](https://github.com/robotwin-Platform/RoboTwin)


## 1. RoboTwin 仿真器项目部分
### 1.1 下载 `RoboTwin`
```bash
git clone https://github.com/robotwin-Platform/RoboTwin.git
```
使用文档可参考：[RoboTwin Usage And Installation](https://robotwin-platform.github.io/doc/usage/index.html)

### 1.2 创建并配置 `RoboTwin` 虚拟环境
#### 1.2.1 下载 RoboTwin 官方推荐的 python 版本
```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
```

#### 1.2.2 在下载 `RoboTwin` 项目后，需要额外下载下面的python依赖库

`RoboTwin` 官方为此提供了一个一键下载的依赖库的脚本,先进入 `RoboTwin` 项目根目录然后执行下面的脚本
```bash
bash script/_install.sh
```

#### 1.2.3 下载 `RoboTwin` 资源（纹理库，3D模型等）
```bash
bash script/_download_assets.sh
```

### 1.3 使用 `RoboTwin` 生成数据
这个部分是 RoboTwin 的主要功能，它可以通过一些我们设置的简单任务奖励函数来快速生成机器人的轨迹数据
```bash
bash collect_data.sh clean_bathroom demo_randomized 0 
```

如果你想知道RoboTwin是如何生成合成数据可能需要去读取他们的论文 [RoboTwin论文](arxiv.org/pdf/2506.18088)

## 2. SEM 项目部分
### 2.1 下载 `RoboOrchardLab`
首先需要将 `RoboOrchardLab` 的项目下载到本地，具体可查看 [SEM 项目](https://github.com/HorizonRobotics/RoboOrchardLab/tree/v0.2.0-release/projects/sem/robotwin)
```bash
git clone https://github.com/HorizonRobotics/RoboOrchardLab.git
```

### 2.2 下载 GroundingDino 预训练模型

```bash
cd {RoboOrchardLab}/projects/sem/robotwin

mkdir ckpt

# 这是用来下载 GroundingDino 预训练模型的脚本
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth -O ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth
# 这是用来重命名的程序
python tools/ckpt_rename.py ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth --output ./ckpt
```

### 2.3 下载 `RoboTwin` 后下载下面的额外python依赖库
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
### 2.4 数据转换
下面的脚本是用来将 `RoboTwin` 生成的数据格式转换为 SEM 模型能够读取的格式
```bash
cd {RoboOrchardLab}/robo_orchard_lab

# for the robotwin2.0 master branch before commit e71140e9734e69686daa420a9be8b75a20ff4587 or the Challenge-Cup-2025 branch
python robo_orchard_lab/dataset/robotwin/robotwin_packer.py \
    --input_path path/to/robotwin_data \
    --output_path "projects/sem/robotwin/data/lmdb" \
    --task_names ${task_names} \
    --config_name demo_clean
```

### 2.5 训练 SEM 模型
```bash
cd projects/sem/robotwin
CONFIG=config_sem_robotwin.py

# 单卡训练
python3 train.py --config ${CONFIG}

# 多卡训练
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

### 2.6 部署 SEM 模型到 RoboTwin 仿真器中
具体的文档可在 sem_policy/ 目录查看`cd sem_policy/`

#### 2.6.1 复制你的训练策略配置文件
```bash
cd {RoboOrchardLab}/projects/sem/robotwin/sem_policy
cp ../config_sem_robotwin.py ./
```

#### 2.6.2 修改配置文件
```bash
# 先复制 urdf 文件
cp ${ROBOTWIN_DIR}/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf ./
```

再修改到你的具体路径
```bash
urdf="/workspace/policy/custom_policy/arx5_description_isaac.urdf",
```
#### 2.6.3 准备验证模型
```bash
CHECKPOINT_DIR=checkpoints/place_empty_cup
mkdir -p ${CHECKPOINT_DIR}
cp {PATH_TO_YOUR_SAFETENSOR_CKPT} ${CHECKPOINT_DIR}
```
如果还未能够完成训练，可以先分工，使用官方提供的[PlaceEmptyCup] (https://huggingface.co/HorizonRobotics/SEM-RoboTwin-Tiny)模型 在 RoboTwin 中对 `place_empty_cup` 任务进行模型验证

#### 2.6.4 将配置好的 sem_policy 部署到 RoboTwin 中
```bash
cd ..
cp -r sem_policy {ROBOTWIN2_PATH}/policy/
```

#### 2.6.5 运行 RoboTwin 并验证 SEM

首先拷贝下机器人类型的配置文件
```bash
cp {ROBOTWIN2_PATH}/task_config/_embodiment_config.yml {ROBOTWIN2_PATH}/task_config/agent_config.yml 
```

再执行验证脚本
```bash
cd {ROBOTWIN2_PATH}/policy/sem_policy
sh eval.sh
```
