# Git 规则（Claude Code 全权代理 commit + push）
- 每完成一个实验 run 自动 commit 并 push 到当前分支
- commit 内容仅限：yml 配置、results.csv、EXPERIMENT_PLAN.md、viz/ 下缩略图（单文件 <2MB）
- commit message 格式：exp: {run名} 完成, mAP50={x}, PSNR={x}
- 提交前必须 git status 自查；严禁 add 任何 .pth / .tar / datasets/ / experiments/ 内容
- push 被拒绝时执行 git pull --rebase 后重推；出现冲突则停下报告，不要自行解决
- 阶段完成后的 merge 回 master 由用户手动执行，Claude Code 不操作 master 分支
