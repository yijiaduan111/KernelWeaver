# KernelWeaver 服务器迁移 Checklist

这份文档是换服务器时的实际操作清单。目标是让新服务器尽快恢复到当前 `feature/model-deliberation-v0` 实验主线可运行的状态。

## 1. 当前主线

- GitHub 仓库：`https://github.com/yijiaduan111/KernelWeaver.git`
- 主线分支：`feature/model-deliberation-v0`
- 迁移时以远端 `origin/feature/model-deliberation-v0` 的最新提交为准。
- `main` 不是当前实验主线，不要直接在新服务器基于 `main` 跑实验。

## 2. 已经进 Git 的关键内容

这些内容 clone 后会直接存在：

- 核心代码：`src/`、`stark/`、`stark_cli.py`
- 常用配置：`configs/`
- 实验脚本：`scripts/run_single.sh`、`scripts/run_batch.sh`、`scripts/run_sequential_tasks.sh`
- 结果接纳脚本：`scripts/accept_run.py`、`scripts/summarize_accepted.py`
- 环境重建脚本：`scripts/bootstrap_env.sh`
- 环境检查脚本：`scripts/check_env.sh`
- 环境锁定文件：`environment.lock.yml`、`requirements-lock.txt`
- 最终结果快照：`runs_final/kernelbench_k10/`
- 迁移文档：`docs/migration.md`、`docs/env_snapshot.md`、本文件

## 3. 不在 Git 里的内容

这些需要手动迁移或重新生成：

- 私有 API 配置：`.env` 或服务器实际使用的 env 文件
- 大型 conda-pack 环境包：如果后面打包，单独传输，不进 Git
- 临时原始运行目录：除 `runs_final/` 外的历史 `runs/` 可按需打包，不建议全部进 Git
- 本地备份文件：例如 `src/providers/http_utils.py.bak_20260621_` 不需要迁移

## 4. 新服务器准备

新服务器至少需要：

- Linux + Git
- Conda / Miniconda / Mambaforge
- NVIDIA driver 能支持 CUDA 12.8 运行时；当前旧服务器参考见 `docs/env_snapshot.md`
- 能正常访问模型中转站/API
- 可选但推荐：`tmux`
- 可选：`nvcc`，如果 CUDA extension 编译需要系统 CUDA toolkit
- 可选：`ncu`，如果要启用 Nsight Compute 诊断

## 5. Clone 与切换分支

```bash
git clone https://github.com/yijiaduan111/KernelWeaver.git
cd KernelWeaver
git checkout feature/model-deliberation-v0
git pull origin feature/model-deliberation-v0
```

确认分支：

```bash
git branch --show-current
git log -1 --oneline --decorate
```

## 6. 重建 Python 环境

默认创建 `kernelweaver` 环境：

```bash
bash scripts/bootstrap_env.sh
```

如果想指定环境名：

```bash
KW_ENV_NAME=kernelweaver bash scripts/bootstrap_env.sh
```

如果新服务器的 `conda` 不在 PATH：

```bash
CONDA_BIN=/path/to/conda KW_ENV_NAME=kernelweaver bash scripts/bootstrap_env.sh
```

脚本会优先使用 `environment.lock.yml`，然后执行 `pip install -e .`。

## 7. 迁移私有 env 配置

不要把私有 env 文件提交进 Git。建议在新服务器仓库根目录放 `.env`，或者放到固定路径后通过 `KERNELWEAVER_ENV_FILE` 指定。

需要重点确认的变量类型：

- OpenAI/GPT compatible provider：key、base URL、model、wire API
- Claude compatible provider：key、base URL、model、API version
- Gemini compatible provider：key、base URL、model
- 任何当前实验脚本依赖的 provider route 配置

可以参考：

```bash
cp /old/server/path/.env .env
chmod 600 .env
```

或：

```bash
export KERNELWEAVER_ENV_FILE=/absolute/path/to/private.env
```

## 8. 环境体检

```bash
KW_ENV_NAME=kernelweaver bash scripts/check_env.sh
```

最低要求：

- `torch.cuda.is_available()` 为 true
- 能 import `src` 和 `stark`
- `nvidia-smi` 能看到 GPU
- provider env 至少把本次实验要用的模型配置好

`nvcc` / `ncu` 如果缺失，不一定阻塞所有实验，但会影响 CUDA 编译或 NCU 诊断能力。是否必须安装取决于当前实验配置。

## 9. 最小 smoke 验证

先不要直接跑大规模实验。建议先跑单题、低 attempt，确认链路通。

示例：

```bash
export KERNELWEAVER_ENV_FILE=$PWD/.env
bash scripts/run_single.sh \
  --experiment main \
  --level 1 \
  --problem-id 40 \
  --backend cuda \
  --python "$(conda run -n kernelweaver python -c 'import sys; print(sys.executable)')" \
  --route-config codeagent_claude \
  --deliberation-config main
```

如果要后台跑：

```bash
export KERNELWEAVER_ENV_FILE=$PWD/.env
bash scripts/run_single.sh \
  --experiment main \
  --level 1 \
  --problem-id 40 \
  --backend cuda \
  --python "$(conda run -n kernelweaver python -c 'import sys; print(sys.executable)')" \
  --route-config codeagent_claude \
  --deliberation-config main \
  --detach
```

验证点：

- API 调用成功
- candidate 生成成功
- CUDA 编译成功
- correctness/evaluation 正常写入 `run.json`
- 输出目录写到 `runs/`

## 10. 正式实验前检查

正式跑批量实验前，建议确认：

```bash
git status --short
KW_ENV_NAME=kernelweaver bash scripts/check_env.sh
nvidia-smi
```

确认没有误在 `main` 分支、没有遗漏 `.env`、没有 GPU 被其他进程占满。

## 11. 常见问题

### PyTorch CUDA 不可用

先看：

```bash
nvidia-smi
conda run -n kernelweaver python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如果新服务器 driver 太旧，优先升级 driver；如果 driver 不方便升级，再换与服务器 CUDA/driver 匹配的 PyTorch wheel。

### API 在本机能通但服务器不通

优先排查：

- `.env` 是否复制到位
- `KERNELWEAVER_ENV_FILE` 是否指向正确文件
- base URL 是否和当前中转站一致
- 服务器是否能访问对应域名
- 中转站余额/限流是否正常

### NCU 不可用

`ncu` 不是 pip 包，来自 NVIDIA Nsight Compute。缺失时可以先关闭相关诊断开关跑主流程，后续再安装系统工具。

### 环境重建太慢或失败

后续可以走完整环境打包方案：

```bash
conda install -n base conda-pack
conda pack -n stark -o kernelweaver-stark-env.tar.gz
```

这个包比较大，不进 Git，适合临搬服务器前单独传输。
