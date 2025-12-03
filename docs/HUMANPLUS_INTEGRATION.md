# HH-HumanPlus 分层控制集成指南

本指南说明如何将 HH (HARL) 作为上层策略，直接控制 H1 人形机器人行走。

## 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                    上层: HH (HARL框架)                          │
│  输入: 84维观测 (姿态、速度、关节状态等)                         │
│  输出: 19维关节位置偏移量 (相对于默认站立姿态)                    │
│  动作范围: [-0.5, 0.5] (可配置)                                  │
│  输出0 = 保持默认站立姿态                                        │
│  算法: HAPPO/HATRPO/HASAC等                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ action_offsets (19维)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│      位置计算: target_joint = default_pose + offset             │
│                                                                  │
│  default_dof_pos ─────┬───→ target_joint_positions             │
│  (站立姿态)            │                                         │
│                        │                                         │
│  HH输出 ─────→ offset ─┘                                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ target_joint_positions (19维)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   IsaacGym 物理仿真                              │
│  H1 人形机器人 (19自由度) + PD控制器                             │
│  返回: obs, reward, done                                         │
└─────────────────────────────────────────────────────────────────┘
```

## 重要说明：为什么不使用预训练的HST策略

在阶段2训练中，我们**不使用**预训练的HST策略网络，原因如下：

1. **奖励冲突**: HST是用npy轨迹文件训练的，其奖励函数包含`target_jt`奖励，
   惩罚机器人偏离这些预设轨迹。
2. **目标矛盾**: 当HH学习自己的行走策略时，机器人运动必然偏离npy轨迹，
   导致`target_jt`奖励持续下降，即使任务表现在改善。
3. **简化训练**: 直接让HH学习输出关节目标位置，由PD控制器跟踪，
   避免了复杂的策略级联问题。

## 训练流程

### 阶段1: 预训练HST (可选)

如果需要使用HST策略，在 humanplus 目录下训练:

```bash
cd /path/to/humanplus/HST/legged_gym

# 训练HST (使用预录制的人体动作数据)
python scripts/train.py --run_name hst_pretrain --headless --sim_device cuda:0 --rl_device cuda:0

# 训练完成后，模型保存在 logs/rough_h1/hst_pretrain/ 目录下
```

### 阶段2: 训练上层HH策略 (推荐方式)

在 HH 目录下直接训练上层策略，**无需使用预训练HST**:

```bash
cd /path/to/HH/examples

# 使用HAPPO算法训练（推荐配置）
python train.py --algo happo --env humanplus --exp_name hh_walking \
    --humanplus_path /path/to/humanplus \
    --headless true \
    --use_pretrained_hst false \
    --training_phase 2

# 可选：调整动作范围（更小的范围=更保守的动作）
python train.py --algo happo --env humanplus --exp_name hh_walking \
    --humanplus_path /path/to/humanplus \
    --action_scale 0.3
```

**关键配置**:
- `use_pretrained_hst: false` - 不使用HST策略网络
- `disable_target_jt_reward: true` - 禁用轨迹跟踪奖励（默认启用）

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

# HST配置 - 阶段2推荐设置
use_pretrained_hst: false  # 不使用HST策略
freeze_hst: true

# 动作空间配置
action_scale: 0.5  # 偏移量范围 [-0.5, 0.5]

# 奖励配置 - 关键！
disable_target_jt_reward: true  # 必须禁用，否则奖励会下降
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
| `target_jt` | 19 | 目标关节位置 |

## 动作空间说明

HH策略输出19维关节位置**偏移量** (相对于默认站立姿态):

| 关节组 | 关节名称 | 索引 |
|--------|----------|------|
| 左腿 | hip_yaw, hip_roll, hip_pitch, knee, ankle | 0-4 |
| 右腿 | hip_yaw, hip_roll, hip_pitch, knee, ankle | 5-9 |
| 躯干 | torso | 10 |
| 左臂 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 11-14 |
| 右臂 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 15-18 |

**动作解释**:
- 输出 0 = 保持默认站立姿态
- 输出范围 [-0.5, 0.5] (可通过 action_scale 调整)
- 实际目标位置 = default_dof_pos + offset

## 奖励函数

使用HST环境中定义的任务奖励 (**`target_jt`已禁用**):

- ✅ `tracking_lin_vel`: 线速度跟踪奖励 (主要优化目标)
- ✅ `tracking_ang_vel`: 角速度跟踪奖励 (主要优化目标)
- ✅ `feet_air_time`: 步态奖励
- ✅ 各种惩罚项: 力矩、碰撞、关节限位等
- ❌ `target_jt`: 轨迹跟踪奖励 (**已禁用！**)

## 常见问题

### Q: 为什么奖励一直下降？

A: 最可能的原因：
1. `target_jt`奖励未禁用 - 检查日志是否显示 "Disabled target_jt reward"
2. 使用了预训练HST - 设置 `use_pretrained_hst: false`

### Q: 如何调整学习难度？

A: 调整 `action_scale`:
- 更小的值 (如 0.2): 更保守的动作，训练更稳定
- 更大的值 (如 0.8): 更大的动作幅度，学习更快但可能不稳定

### Q: 可以在阶段3使用HST吗？

A: 理论上可以，但需要确保HST是用随机目标位置训练的（而非npy轨迹）。
当前预训练的HST使用npy轨迹，不适合作为下层控制器。

## 参考

- [HARL论文](https://jmlr.org/papers/v25/23-0488.html)
- [HumanPlus项目](https://humanoid-ai.github.io/)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
