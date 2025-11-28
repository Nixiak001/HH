# HH-HumanPlus 分层控制集成指南

本指南说明如何将 HH (HARL) 作为上层策略输出目标位姿，让 HumanPlus HST 作为下层控制器执行。

## 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                    上层: HH (HARL框架)                          │
│  输入: 84维观测 (姿态、速度、关节状态等)                         │
│  输出: 19维目标关节位置                                          │
│  算法: HAPPO/HATRPO/HASAC等                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ target_jt (19维)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              下层: HST (Humanoid Shadowing Transformer)         │
│  输入: obs_history (8帧×84维) + target_jt (19维)                │
│  输出: 19维关节动作 → PD控制 → 力矩                              │
│  架构: Transformer (4层, 128隐藏维度)                            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ torques
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   IsaacGym 物理仿真                              │
│  H1 人形机器人 (19自由度) + 地形                                 │
│  返回: obs, reward, done                                         │
└─────────────────────────────────────────────────────────────────┘
```

## 训练流程

### 阶段1: 预训练HST (可与阶段2并行)

在 humanplus 目录下训练 HST:

```bash
cd /path/to/humanplus/HST/legged_gym

# 训练HST (使用预录制的人体动作数据)
python scripts/train.py --run_name hst_pretrain --headless --sim_device cuda:0 --rl_device cuda:0

# 训练完成后，模型保存在 logs/h1/hst_pretrain/ 目录下
```

### 阶段2: 训练上层HH策略 (可与阶段1并行)

在 HH 目录下训练上层策略:

```bash
cd /path/to/HH/examples

# 使用HAPPO算法训练
python train.py --algo happo --env humanplus --exp_name hh_upper_layer \
    --humanplus_path /path/to/humanplus \
    --headless true \
    --use_pretrained_hst false \
    --training_phase 2

# 或使用其他算法
python train.py --algo hasac --env humanplus --exp_name hh_upper_hasac
```

### 阶段3: 联合微调

当HST和上层HH都训练好后，进行联合微调:

```bash
cd /path/to/HH/examples

python train.py --algo happo --env humanplus --exp_name joint_finetuning \
    --humanplus_path /path/to/humanplus \
    --hst_checkpoint /path/to/humanplus/HST/logs/h1/hst_pretrain/model.pt \
    --use_pretrained_hst true \
    --freeze_hst false \
    --training_phase 3
```

## 配置说明

### 环境配置 (`harl/configs/envs_cfgs/humanplus.yaml`)

```yaml
# 任务配置
task: h1_walking

# humanplus安装路径
humanplus_path: /path/to/humanplus

# 仿真配置
headless: true  # 训练时设为true，渲染时设为false
device: "cuda:0"

# Episode配置
episode_length: 1000

# HST配置
use_pretrained_hst: true
hst_checkpoint: /path/to/hst/model.pt
freeze_hst: true  # 阶段2设为true，阶段3设为false

# 训练阶段
training_phase: 2  # 1=仅HST, 2=仅HH, 3=联合训练
```

### 算法配置

推荐使用 HAPPO 或 HASAC:

```bash
# HAPPO (on-policy)
python train.py --algo happo --env humanplus --exp_name test

# HASAC (off-policy, 样本效率更高)
python train.py --algo hasac --env humanplus --exp_name test
```

## 渲染与视频保存

训练完成后，生成可视化视频:

```bash
cd /path/to/HH/examples

# 设置渲染模式
python train.py --algo happo --env humanplus --exp_name render_test \
    --use_render true \
    --model_dir /path/to/trained/models \
    --headless false \
    --render_episodes 5
```

## 观测空间说明

上层HH策略接收的观测 (84维):

| 分量 | 维度 | 说明 |
|------|------|------|
| `base_orn_rp` | 2 | 身体姿态 roll/pitch |
| `base_ang_vel` | 3 | 角速度 |
| `commands` | 3 | 速度命令 (vx, vy, ω) |
| `dof_pos - default` | 19 | 当前关节位置偏差 |
| `dof_vel` | 19 | 关节速度 |
| `actions` | 19 | 上一步动作 |
| `target_jt` | 19 | 目标关节位置 (来自上层) |

## 动作空间说明

上层HH策略输出 (19维目标关节位置):

| 关节组 | 关节名称 | 索引 |
|--------|----------|------|
| 左腿 | hip_yaw, hip_roll, hip_pitch, knee, ankle | 0-4 |
| 右腿 | hip_yaw, hip_roll, hip_pitch, knee, ankle | 5-9 |
| 躯干 | torso | 10 |
| 左臂 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 11-14 |
| 右臂 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 15-18 |

## 奖励函数

使用HST环境中定义的任务奖励:

- `tracking_lin_vel`: 线速度跟踪奖励
- `tracking_ang_vel`: 角速度跟踪奖励
- `target_jt`: 目标关节位置跟踪奖励
- `feet_air_time`: 步态奖励
- 各种惩罚项: 力矩、碰撞、关节限位等

## 文件结构

```
HH/
├── harl/
│   ├── envs/
│   │   └── humanplus/
│   │       ├── __init__.py
│   │       ├── humanplus_env.py      # 环境wrapper
│   │       └── humanplus_logger.py   # 日志记录器
│   ├── configs/
│   │   └── envs_cfgs/
│   │       └── humanplus.yaml        # 环境配置
│   └── utils/
│       ├── envs_tools.py             # 已更新，支持humanplus
│       └── configs_tools.py          # 已更新，支持humanplus
├── examples/
│   └── train.py                      # 已更新，支持humanplus
└── docs/
    └── HUMANPLUS_INTEGRATION.md      # 本文档
```

## 常见问题

### Q: IsaacGym导入错误
A: 确保在PyTorch之前导入isaacgym。代码已自动处理此问题。

### Q: 找不到humanplus模块
A: 设置正确的 `humanplus_path` 参数指向humanplus安装目录。

### Q: 训练不稳定
A: 建议先单独预训练HST至收敛，再训练上层HH策略。

### Q: 如何使用我已经训练好的17维扭矩输出策略？
A: 需要修改动作空间维度并调整输出映射。由于HST期望19维目标关节位置，建议重新训练。

## 参考

- [HARL论文](https://jmlr.org/papers/v25/23-0488.html)
- [HumanPlus项目](https://humanoid-ai.github.io/)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
