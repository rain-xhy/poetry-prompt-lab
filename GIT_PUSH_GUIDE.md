# Git 推送准备指南

## 当前状态
- `.git` 已存在（68.6MB）
- `.venv/` 已被误加入暂存区（167.2MB，3308个文件）
- remote 尚未配置

## ⚠️ 必须先清理

### 1. 移除暂存区中的 .venv
```bash
cd "E:\desktop\北大软微\课程\大语言模型应用与实践\作业\11\代码\poetry-prompt-lab"
git rm -r --cached .venv
```

### 2. 验证 .gitignore 生效
```bash
git status --short | head -20
```

### 3. 如果还有其他不需要的文件，继续清理
```bash
# 查看当前暂存的文件
git status

# 如果需要重置所有暂存
git reset
```

## 推送步骤

### 1. 配置 remote
```bash
git remote add origin https://github.com/rain-xhy/poetry-prompt-lab.git
```

### 2. 查看当前分支
```bash
git branch
```

### 3. 添加需要的文件
```bash
# 添加所有符合 .gitignore 规则的文件
git add .

# 查看将要提交什么
git status
```

### 4. 提交
```bash
git commit -m "Initial commit: poetry prompt strategy research

- 9 prompt strategies comparison
- 4 types of semantic traps
- 1534 generations across 4 stages
- Complete analysis and figures
"
```

### 5. 推送
```bash
# 首次推送
git push -u origin main

# 或者如果分支叫 master
git push -u origin master

# 如果远程已有内容需要合并
git pull origin main --rebase
git push -u origin main
```

## 推送前检查清单

- [ ] `.venv/` 已从暂存区移除
- [ ] `.gitignore` 已更新
- [ ] `git status` 显示的文件合理（无 .venv、无 __pycache__）
- [ ] README.md 已更新为正确的 GitHub 链接
- [ ] 没有包含密钥或敏感信息
- [ ] 仓库大小合理（应该 < 50MB）

## 如果需要完全重新开始

```bash
# 备份当前代码
cd "E:\desktop\北大软微\课程\大语言模型应用与实践\作业\11\代码"
cp -r poetry-prompt-lab poetry-prompt-lab-backup

# 删除 .git
cd poetry-prompt-lab
rm -rf .git

# 重新初始化
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/rain-xhy/poetry-prompt-lab.git
git branch -M main
git push -u origin main
```

## 预期推送内容

应该包含：
- `README.md` (~11KB)
- `.gitignore`
- `data/` (诗词数据和提示模板)
- `scripts/` (Python脚本)
- `src/` (源代码)
- `results/figures/` (7张图)
- `results/stage2_main/analysis/` (汇总分析)
- 论文文档

不应包含：
- `.venv/` (167MB)
- `__pycache__/`
- `.log` 文件
- JSON 生成结果（太大）
