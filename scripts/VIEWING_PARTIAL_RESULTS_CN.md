# 查看部分实验结果指南

## 问题

实验还在运行中，但我想看看当前的实验效果和学习曲线，应该怎么做？

## 解决方案

使用 `plot_partial_results.py` 脚本可以在实验进行中随时查看部分结果。

## 快速使用

### 基本用法

```bash
cd /home/runner/work/HH/HH
python scripts/plot_partial_results.py \
    --results_dir comparison_results/mamujoco_Humanoid-v2_17x1
```

### 输出说明

脚本会在 `comparison_results/mamujoco_Humanoid-v2_17x1/partial_analysis/` 目录下生成：

1. **`partial_comparison.png`** - 算法对比图
   - 显示三个算法（HAPPO、MAPPO、HATD3）的当前学习曲线
   - 包含均值和标准差（如果有多个种子）
   - 标题显示"In Progress"提醒这是部分结果

2. **`partial_individual_runs.png`** - 独立运行曲线
   - 三个子图分别显示每个算法的所有种子
   - 可以看到每个种子的具体表现
   - 方便识别异常的运行

3. **`progress_report.md`** - 进度报告
   - 每个算法有多少个种子已开始
   - 当前训练到多少步
   - 最新的奖励值
   - 最佳奖励值

## 示例场景

### 场景1: 查看整体进度

```bash
python scripts/plot_partial_results.py \
    --results_dir comparison_results/mamujoco_Humanoid-v2_17x1
```

输出示例：
```
============================================================
Visualizing Partial Experiment Results
============================================================
Results Directory: comparison_results/mamujoco_Humanoid-v2_17x1
Output Directory: comparison_results/mamujoco_Humanoid-v2_17x1/partial_analysis
Metric: eval/average_episode_rewards
============================================================

Aggregating available results...
Found 5 TensorBoard event files
  ✓ happo seed 1: 1 metrics
  ✓ happo seed 2: 1 metrics
  ✓ mappo seed 1: 1 metrics
  ✓ mappo seed 2: 1 metrics
  ✓ hatd3 seed 1: 1 metrics

Generating visualizations...
✓ Saved individual runs plot to: .../partial_individual_runs.png
✓ Saved comparison plot to: .../partial_comparison.png

Generating progress report...
✓ Saved progress report to: .../progress_report.md

============================================================
Partial Analysis Complete!
============================================================

Results saved to: .../partial_analysis
  - partial_comparison.png
  - partial_individual_runs.png
  - progress_report.md

You can run this script again as experiments progress to see updated results.
============================================================
```

### 场景2: 查看训练曲线而非评估曲线

```bash
python scripts/plot_partial_results.py \
    --results_dir comparison_results/mamujoco_Humanoid-v2_17x1 \
    --metric train/average_episode_rewards
```

### 场景3: 自定义输出目录

```bash
python scripts/plot_partial_results.py \
    --results_dir comparison_results/mamujoco_Humanoid-v2_17x1 \
    --output_dir my_analysis
```

## 何时使用

### ✅ 适合使用的情况

- **实验刚开始**：确认实验正在正常运行
- **训练中期**：检查算法表现趋势，判断是否需要调整
- **发现问题**：某个算法训练异常，想快速确认
- **定期检查**：每隔几小时查看一次进度
- **论文截图**：需要中期结果作为分析素材

### ❌ 不适合的情况

- 实验已全部完成（此时应使用 `analyze_comparison_results.py`）
- 还没有任何数据生成

## 与最终分析的区别

| 特性 | plot_partial_results.py | analyze_comparison_results.py |
|------|------------------------|-------------------------------|
| **使用时机** | 实验进行中 | 实验完成后 |
| **数据完整性** | 处理不完整数据 | 需要完整数据 |
| **统计分析** | 基本统计 | 完整统计分析 |
| **输出目录** | partial_analysis/ | analysis/ |
| **可重复运行** | ✅ 建议多次运行 | ✅ 一次即可 |

## 进度报告示例

生成的 `progress_report.md` 内容示例：

```markdown
# Partial Experiment Results (In Progress)

**Generated at**: 2026-01-18 13:30:45

## Progress Overview

| Algorithm | Seeds Available | Latest Step | Current Reward |
|-----------|-----------------|-------------|----------------|
| HAPPO | 2 | 1,500,000 | 4,523.45 |
| MAPPO | 2 | 1,200,000 | 4,102.31 |
| HATD3 | 1 | 800,000 | 3,876.22 |

## Detailed Progress

### HAPPO

**Seed 1**:
- Current Step: 1,500,000
- Current Reward: 4,523.45
- Data Points: 60
- Best Reward: 4,678.90

**Seed 2**:
- Current Step: 1,450,000
- Current Reward: 4,401.23
- Data Points: 58
- Best Reward: 4,512.34

### MAPPO
...

```

## 实用技巧

### 技巧1: 设置定时任务查看进度

```bash
# 每小时自动生成一次部分结果
watch -n 3600 "python scripts/plot_partial_results.py --results_dir comparison_results/mamujoco_Humanoid-v2_17x1"
```

### 技巧2: 比较不同时间点的进度

```bash
# 第一次运行
python scripts/plot_partial_results.py \
    --results_dir comparison_results/mamujoco_Humanoid-v2_17x1 \
    --output_dir partial_analysis_t1

# 几小时后再次运行
python scripts/plot_partial_results.py \
    --results_dir comparison_results/mamujoco_Humanoid-v2_17x1 \
    --output_dir partial_analysis_t2

# 对比两次结果
```

### 技巧3: 结合 TensorBoard 使用

```bash
# 终端1: 运行 TensorBoard
tensorboard --logdir comparison_results/mamujoco_Humanoid-v2_17x1

# 终端2: 定期生成静态图
python scripts/plot_partial_results.py \
    --results_dir comparison_results/mamujoco_Humanoid-v2_17x1
```

## 故障排除

### 问题1: 找不到数据

```
ERROR: No data found to visualize
```

**解决方法**:
1. 确认实验已经开始运行
2. 检查结果目录路径是否正确
3. 确认 TensorBoard 日志已生成（在 `results/` 目录下查找 `events.out.tfevents.*` 文件）

### 问题2: 指标不存在

```
WARNING: Metric 'eval/average_episode_rewards' not found in data.
```

**解决方法**:
- 运行脚本会自动列出所有可用指标
- 使用 `--metric` 参数指定其他指标
- 常见指标名称：
  - `eval/average_episode_rewards` (评估奖励)
  - `train/average_episode_rewards` (训练奖励)
  - `train/episode_rewards` (单回合奖励)

### 问题3: 缺少依赖包

```
ERROR: matplotlib/seaborn required. Install with: pip install matplotlib seaborn
```

**解决方法**:
```bash
pip install matplotlib seaborn tensorboard pandas
```

## 总结

使用 `plot_partial_results.py` 可以：
- ✅ 实时监控训练进度
- ✅ 提前发现问题
- ✅ 不需要等待实验完成
- ✅ 随时生成可视化结果
- ✅ 支持不完整数据

建议在实验运行期间定期使用此工具检查进度！
