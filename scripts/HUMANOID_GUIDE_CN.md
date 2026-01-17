# Humanoid-v2-17x1 对比实验快速指南

本指南专门针对 Humanoid-v2-17x1 环境的 HAPPO、MAPPO 和 HATD3 算法对比实验。

## 环境特点

**Humanoid-v2-17x1** 是一个高维度的人形机器人控制任务：
- **17个智能体**: 每个控制身体的一个部位
- **异构智能体**: 不同部位有不同的观察和动作空间
- **连续动作空间**: 所有三种算法都支持
- **高计算成本**: 每个实验运行需要 24-48 小时（GPU）

## 快速开始

### 方法 1: 使用一键脚本

```bash
cd /home/runner/work/HH/HH
bash scripts/example_humanoid_comparison.sh
```

这个脚本会：
1. 生成对比配置
2. 询问确认后运行实验（3个种子 × 3个算法 = 9次运行）
3. 自动分析结果并生成报告

### 方法 2: 分步执行

#### 第1步: 生成配置

```bash
cd /home/runner/work/HH/HH
python scripts/generate_comparison_configs.py \
    --env mamujoco \
    --scenario Humanoid-v2 \
    --agent_conf 17x1 \
    --num_env_steps 10000000 \
    --seeds 1 2 3
```

**输出**: 配置文件保存在 `comparison_configs/mamujoco_Humanoid-v2_17x1/`

#### 第2步: 运行实验

```bash
bash scripts/run_comparison_experiment.sh mamujoco Humanoid-v2 17x1 3
```

**注意事项**:
- 每个实验需要 24-48 小时（取决于硬件）
- 建议使用 `screen` 或 `tmux` 保持会话
- 确保有足够的磁盘空间（每个运行约 5-10GB）

**运行单个算法**（如果资源有限）:
```bash
cd /home/runner/work/HH/HH/examples
python train.py \
    --load_config ../comparison_configs/mamujoco_Humanoid-v2_17x1/happo_comparison.json \
    --exp_name comparison_happo_seed1 \
    --seed 1
```

#### 第3步: 分析结果

```bash
python scripts/analyze_comparison_results.py \
    --exp_dir comparison_results/mamujoco_Humanoid-v2_17x1
```

**输出**:
- `analysis/learning_curves.png` - 学习曲线对比图
- `analysis/comparison_report.md` - 详细统计报告
- `analysis/statistics.json` - 原始统计数据

## 配置说明

### 已生成的配置差异

| 参数 | HAPPO | MAPPO | HATD3 |
|------|-------|-------|-------|
| **算法类型** | On-policy | On-policy | Off-policy |
| **参数共享** | No (`share_param: false`) | Yes (`share_param: true`) | No (`share_param: false`) |
| **更新顺序** | Sequential (`fixed_order: false`) | Fixed (`fixed_order: true`) | Sequential (`fixed_order: false`) |
| **网络结构** | [256, 256] | [128, 128, 128] | [256, 256] |
| **学习率** | 0.0003 | 0.0005 | 0.0005 |
| **使用RNN** | Yes (`use_recurrent_policy: true`) | No | N/A (off-policy) |
| **Episode长度** | 2000 | 100 | N/A (continuous) |

### 统一的参数

- **环境**: Humanoid-v2-17x1
- **总训练步数**: 10,000,000
- **评估设置**: 40 episodes, 20 threads
- **CUDA**: 启用
- **随机种子**: 1, 2, 3

## 预期结果

根据论文 (JMLR 2024)，在 MAMuJoCo 环境上：
- **HAPPO** 通常表现最好，特别是在异构智能体场景
- **MAPPO** 在同构场景下表现良好，但在异构场景可能稍弱
- **HATD3** 作为 off-policy 算法，样本效率可能更高

### Humanoid-v2-17x1 特点
- **高维度**: 17个智能体协作控制
- **稀疏奖励**: 需要长时间训练才能看到明显进步
- **训练曲线**: 预计在前 5M 步会有明显提升

## 常见问题

### Q: 可以减少训练时间吗？

A: 可以，但会影响结果质量：

```bash
# 减少到 5M 步
python scripts/generate_comparison_configs.py \
    --env mamujoco \
    --scenario Humanoid-v2 \
    --agent_conf 17x1 \
    --num_env_steps 5000000
```

### Q: 只想运行 1 个随机种子可以吗？

A: 可以，但统计可信度会降低：

```bash
python scripts/generate_comparison_configs.py \
    --env mamujoco \
    --scenario Humanoid-v2 \
    --agent_conf 17x1 \
    --seeds 1

bash scripts/run_comparison_experiment.sh mamujoco Humanoid-v2 17x1 1
```

### Q: 内存不足怎么办？

A: 在配置文件中减少并行环境数量：

打开 `comparison_configs/mamujoco_Humanoid-v2_17x1/happo_comparison.json`，修改：
```json
"n_rollout_threads": 10  // 从 20 减少到 10
```

### Q: 如何查看训练进度？

A: 实时日志保存在：
```bash
tail -f comparison_results/mamujoco_Humanoid-v2_17x1/happo_seed1_log.txt
```

或使用 TensorBoard:
```bash
tensorboard --logdir comparison_results/mamujoco_Humanoid-v2_17x1
```

### Q: 实验中断了怎么办？

A: 如果使用了 checkpoint（在配置中设置 `model_dir`），可以从断点继续。否则需要重新运行。

## 性能优化建议

### 硬件要求
- **最低**: GPU with 6GB memory, 32GB RAM
- **推荐**: GPU with 16GB+ memory, 64GB+ RAM
- **最佳**: Multi-GPU setup

### 加速技巧

1. **使用多GPU**（需要修改代码）
2. **减少评估频率**: 修改 `eval_interval` 为更大的值
3. **使用更少的评估回合**: 修改 `eval_episodes` 为 20
4. **禁用评估**: 设置 `use_eval: false`（仅用于快速测试）

## 分析技巧

### 查看特定指标

```bash
python scripts/analyze_comparison_results.py \
    --exp_dir comparison_results/mamujoco_Humanoid-v2_17x1 \
    --metric train/average_episode_rewards
```

### 导出数据用于论文

结果已保存为 JSON:
```bash
cat comparison_results/mamujoco_Humanoid-v2_17x1/analysis/statistics.json
```

### 自定义绘图

使用 Python 读取数据并自定义绘图:
```python
import json
with open('comparison_results/mamujoco_Humanoid-v2_17x1/analysis/statistics.json', 'r') as f:
    stats = json.load(f)
    print(stats)
```

## 引用

如果使用本框架进行研究，请引用 HARL 论文:

```bibtex
@article{JMLR:v25:23-0488,
  author  = {Yifan Zhong and Jakub Grudzien Kuba and Xidong Feng and Siyi Hu and Jiaming Ji and Yaodong Yang},
  title   = {Heterogeneous-Agent Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {32},
  pages   = {1--67},
  url     = {http://jmlr.org/papers/v25/23-0488.html}
}
```

## 联系和反馈

如有问题或建议，请在 GitHub Issues 中提出。
