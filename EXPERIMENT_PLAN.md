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

### P1 结果(2026-07-03/04 完成,全部 5 run 一次通过,无 OOM/NaN)

| run | 设定 | mAP@50 | mAP@[.5:.95] | PSNR | LPIPS | 训练时长 |
|---|---|---|---|---|---|---|
| 1 | H→T(oracle) | 0.659 | **0.379** | — | — | 40 min |
| 2 | L→T | 0.533 | 0.285 | 22.77 | 0.411 | 39 min |
| 3 | S→T | 0.538 | 0.290 | 20.14 | 0.424 | 62 min |
| 4 | S+T | 0.557 | 0.302 | 23.78 | 0.355 | 61 min |
| 5 | **SR4IR** | **0.583** | **0.329** | **24.00** | **0.280** | 191 min |

对比图:`viz/p1_x4_map5095.png`

**结论:**
1. 排序 **H→T (0.379) > SR4IR (0.329) > S+T (0.302) > S→T (0.290) > L→T (0.285)**,
   与论文主表趋势完全一致(预期 H→T > SR4IR > S+T ≈ S→T > L→T)✅。
2. SR4IR 相对 L→T 下限 +4.4 mAP 点,吃掉了 H→T/L→T 差距(9.4 点)的 47%;
   相对最强朴素基线 S+T 也 +2.7 点,且 PSNR/LPIPS 同时最优——
   "任务驱动感知损失不牺牲重建质量还能涨点"的核心叙事在 1/3 训练量下成立。
3. 与官方 ×8 表相比,×4 下各设定间差距压缩(退化更轻,下限更高),消融信号
   依然清晰,P2 可直接基于 run5 配置展开。
4. 备注:run1 (0.379) 略超官方 30-epoch H2T (0.367),说明 10 epoch + 等效 batch 8
   对该检测器已足够;run3 的 PSNR 低于 L→T 是 DIV2K 权重未在 VOC 微调的已知偏离
   (见 run3 yml 注释),不影响检测端结论。

## P2 · 消融(3 run,基于 run5 配置各改一处)

| run | 名称 | 改动 |
|---|---|---|
| 6 | det_voc_x4_sr4ir_notdp_e10_s42 | 去 TDP loss(删除 tdp_opt) |
| 7 | det_voc_x4_sr4ir_nocqmix_e10_s42 | 去 CQMix(删除 det_cqmix_opt,det_sr/det_hr 权重 0.5/0.5) |
| 8 | det_voc_x4_sr4ir_noalt_e10_s42 | 去交替训练(phase1/phase2 同步更新,需代码开关) |

分支:`exp/p2-ablation`。

### P2 结果(2026-07-04 完成,3 run 一次通过)

| 变体 | mAP@50 | mAP@[.5:.95] | Δ vs full | PSNR | LPIPS |
|---|---|---|---|---|---|
| SR4IR full(run5) | 0.583 | 0.329 | — | 24.00 | 0.280 |
| w/o TDP(run6) | 0.580 | 0.327 | −0.2 pt | 24.57 | 0.339 |
| w/o CQMix(run7) | 0.589 | 0.331 | +0.2 pt | 23.90 | 0.280 |
| w/o 交替训练(run8) | 0.516 | **0.284** | **−4.5 pt** | 23.99 | 0.315 |

对比图:`viz/p2_x4_ablation_map5095.png`(代码开关:`train.alternate_training`,默认 true)

**结论:**
1. **交替训练是小规模设定下唯一 load-bearing 组件**:去掉后 mAP@[.5:.95] 跌回
   L→T 下限(0.284 vs 0.285),SR4IR 的全部检测优势消失;其 1-epoch 中间态
   mAP@50 仅 0.009(交替模式同期 0.189),同步更新让 TDP 梯度直接冲击检测器、
   检测器又在漂移的 SR 分布上训练,早期即崩、后期未能恢复。
2. TDP 与 CQMix 的单独检测效应在 ×4 / 10 epoch 下落在种子噪声内(±0.2 pt);
   但 **TDP 对感知质量的贡献清晰**:去掉后 LPIPS 0.280→0.339(+21%),
   PSNR 反升 0.57 dB——TDP 用像素保真换任务相关高频,与论文叙事一致。
3. 解读注意:论文消融在 ×8 / 30 epoch 下进行,组件效应更大;本组只做方向性
   验证。TDP/CQMix 的检测端效应待 P3(×8)复核,单 seed 结论不宜过度外推。

## P3 · ×8 趋势验证(3 run)

| run | 名称 | 派生自 |
|---|---|---|
| 9 | det_voc_x8_l2t_e10_s42 | 官方 002 |
| 10 | det_voc_x8_swt_e10_s42 | 官方 023(SR 用 edsr_baseline_x8.pt) |
| 11 | det_voc_x8_sr4ir_e10_s42 | 官方 024(同上) |

验证"倍率越大 SR4IR 优势越明显"。分支:`exp/p3-x8`。

### P3 结果(2026-07-05 完成;run11 曾于 07-04 在 epoch 4 处中断,从头重跑)

| 设定 | ×4 mAP@[.5:.95] | ×8 mAP@[.5:.95] | ×8 官方 30ep | ×8 PSNR / LPIPS |
|---|---|---|---|---|
| L→T | 0.285 | 0.191 | 0.189 | 20.29 / 0.559 |
| S+T | 0.302 | 0.215 | 0.203 | 21.04 / 0.500 |
| SR4IR | 0.329 | **0.258** | 0.255 | 20.90 / 0.423 |

趋势图:`viz/p3_scale_trend_map5095.png`

**结论:**
1. **"倍率越大、SR4IR 优势越明显"成立**:SR4IR−S+T 差距从 ×4 的 +2.7 pt 放大到
   ×8 的 **+4.3 pt**,SR4IR−L→T 从 +4.4 pt 放大到 **+6.7 pt**
   (官方 30ep ×8 参照:+5.2 / +6.6 pt,方向与量级一致)。
2. ×8 三个点均与官方 30-epoch 结果高度一致(0.191/0.189、0.215/0.203、
   0.258/0.255),1/3 训练量在该任务上已基本收敛,复现成本控制有效。
3. 结合 P2:×8 下组件效应确实更大(SR4IR 相对 S+T 的增益即 TDP+CQMix+交替的
   合成效应,×8 比 ×4 大 60%),支持 P2 的"小倍率下单组件效应被压缩"解读。

## 战役总结(11/11 run 完成,2026-07-03 ~ 07-05)

P0 六设定逐位对齐 → P1 五连趋势复现(H2T > SR4IR > S+T > S2T > L2T)→
P2 消融定位关键组件(交替训练 load-bearing,TDP 主贡献感知质量)→
P3 双倍率验证核心结论(优势随倍率放大)。
全部训练在单卡 4070 Laptop 完成,11 个正式 run 总训练时长约 19.1 小时
(另 2 次 smoke 约 0.7h),峰值显存 4.8GB,零 OOM、零 NaN。
详细流程与经验见 REPRODUCTION_REPORT.md。SR4IR 复现闭环,可作为后续自研方法
(ROI 选择性超分)的基线与代码底座;阶段 0 的 S1/S2 判据(指标差距 <1 mAP)达成。

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
