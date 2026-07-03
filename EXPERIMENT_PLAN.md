# SR4IR 小规模复现实验计划(det / VOC / EDSR)

> 目标:在 RTX 4070 Laptop (8GB) 上以 1/3 训练量复现 SR4IR 论文检测主表的趋势,
> 为后续自研方法(ROI 选择性超分)提供可信的基线与代码基础。
> 对应《实验清单与记录规范》中的阶段 0(S1/S2)。

## 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop, 8GB |
| PyTorch / CUDA | 2.7.1+cu118 / 12.8 driver |
| Python | 3.13.5 |
| 数据 | Pascal VOC2012 → COCO 格式(train 5717 / val 5823) |
| 预训练 SR | DIV2K EDSR-baseline ×4 / ×8(官方 Google Drive) |
| P0 官方权重 | det ×8 全部 5 设定 + H2T(官方 Google Drive) |

## 公共训练配置(P1–P3 所有训练 run)

相对官方配置的统一修改,其余保持官方值:

| 项 | 官方 | 本实验 | 说明 |
|---|---|---|---|
| batch_size | 16 | **2** | 8GB 显存限制 |
| grad_accum | —(无此功能) | **4** | 新增功能,等效 batch 8 |
| AMP | —(无此功能) | **on** | 新增功能,`train.amp: true` |
| epoch | 30 | **10** | 官方 1/3 |
| manual_seed | 100 | **42** | |
| num_threads | 16 | 4 | Windows DataLoader spawn 开销 |
| eval_freq / save_freq | 5 / 10 | 5 / 5 | epoch=10 下保证有中间 ckpt |
| 学习率 | 官方值 | 官方值不变 | 等效 batch 8 vs 16,靠 warmup 稳住 |

> AMP 与梯度累积由本次对 `src/models/det/*.py` 的修改提供(`train.amp`、`train.grad_accum`,
> 默认 false/1,不影响官方配置的行为)。
> 可视化默认只输出验证集前 10 张(SequentialSampler,样本固定),由 `test.visualize_first_n` 控制。

## P0 · 管线验证(0 次训练)

官方预训练权重(det, EDSR, ×8)`--test_only --visualize`,指标对齐 README 结果表:

| 设定 | 官方 exp 名 | README mAP | README LPIPS |
|---|---|---|---|
| H→T | 000_H2T | 36.7 | 0.000 |
| L→T | 002_L2T_x8 | 18.9 | 0.559 |
| S→T | 021_S2T_edsr_x8 | 21.9 | 0.494 |
| T→S | 022_T2S_edsr_x8 | 15.5 | 0.476 |
| S+T | 023_SwT_edsr_x8 | 20.3 | 0.506 |
| SR4IR | 024_SR4IR_edsr_x8 | 25.5 | 0.416 |

配置副本在 `options/det/p0/`(仅改 num_threads=4,`name` 与官方一致以复用权重目录)。
注:README 的 mAP 为 COCO `AP@[0.5:0.95]`;逐位对齐以该列为准。

## P1 · ×4 主线(5 run,核心产出)

| run | 名称 | model_type | 说明 |
|---|---|---|---|
| 1 | det_voc_x4_h2t_e10_s42 | hr_det | 上限,只训检测器 |
| 2 | det_voc_x4_l2t_e10_s42 | lr_det | 下限,bilinear 上采样 LR 直接检测 |
| 3 | det_voc_x4_s2t_e10_s42 | sr_det(SR 冻结) | SR = DIV2K 预训练 EDSR ×4 直接用 |
| 4 | det_voc_x4_swt_e10_s42 | sr_det(联合训练) | 朴素联合 S+T:pixel + det loss |
| 5 | det_voc_x4_sr4ir_e10_s42 | sr4ir_det | 完整方法:TDP + CQMix + 交替训练 |

- 配置在 `options/det/p1/`,均派生自官方 001/011/013/014。
- run3 偏离说明:官方 S→T 的 SR 先在 VOC 上训 30 epoch(010_SR),本实验按计划直接用
  DIV2K 预训练权重,省一次训练;解读 run3 时注意此差异。
- run5 先跑 smoke(`det_voc_x4_sr4ir_smoke_e1_s42.yml`,1 epoch + warmup_epoch=0 以便
  一开始就走 TDP/CQMix 全路径),确认不 OOM、loss 下降、能出 ckpt,再跑全程。
- 调度器按 1/3 缩放:cosine T_max/periods 30→10;SR4IR 的 warmup_epoch 3→1,periods [2,27]→[1,9]。
- 产出:五柱对比图(mAP@50 / mAP@[.5:.95]),趋势应为 H2T > SR4IR > S2T ≈ S+T > L2T。

## P2 · 消融(3 run,基于 run5 配置各改一处)

| run | 名称 | 改动 |
|---|---|---|
| 6 | det_voc_x4_sr4ir_notdp_e10_s42 | 去 TDP loss(删除 tdp_opt) |
| 7 | det_voc_x4_sr4ir_nocqmix_e10_s42 | 去 CQMix(删除 det_cqmix_opt,det_sr/det_hr 权重 0.5/0.5) |
| 8 | det_voc_x4_sr4ir_noalt_e10_s42 | 去交替训练(phase1/phase2 同步更新,需代码开关) |

分支:`exp/p2-ablation`。

## P3 · ×8 趋势验证(3 run)

| run | 名称 | 派生自 |
|---|---|---|
| 9 | det_voc_x8_l2t_e10_s42 | 官方 002 |
| 10 | det_voc_x8_swt_e10_s42 | 官方 023(SR 用 edsr_baseline_x8.pt) |
| 11 | det_voc_x8_sr4ir_e10_s42 | 官方 024(同上) |

验证"倍率越大 SR4IR 优势越明显"。分支:`exp/p3-x8`。

## 每个 run 的执行协议

1. smoke 先行(新模板首次使用时):1 epoch + 1 次验证,确认不 OOM、loss 下降、出 ckpt。
2. 训练:`python src/main.py -opt options/det/p1/<run>.yml`
3. 终测:`python src/main.py -opt options/det/p1/<run>.yml --test_only --visualize`
   (test_only 自动加载 latest 权重并计算 LPIPS)
4. 收集:`python collect_results.py --task det --run <run名>` → 追加/更新 `results.csv`
5. 可视化:`python tools/make_viz.py --task det --run <run名>` → `viz/<run名>/` 固定 10 张缩略图
6. Git:提交该 run 的 yml + results.csv + viz 缩略图;`datasets/`、`experiments/`、`*.pth` 已在 .gitignore。

分支:P1 → `exp/p1-mainline`,P2 → `exp/p2-ablation`,P3 → `exp/p3-x8`。

## results.csv 字段

`run, date, git_commit, task, scale, setting, epochs, seed, map_50, map_5095, ap_small, psnr, lpips, source_log, notes`
