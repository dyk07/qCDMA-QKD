# qCDMA-QKD

q-CDMA-CV-QKD 中**码分发闭环**的研究实现：把"CDMA 码如何安全分发与更新"
这一信任空白变为可计算、可归约、可预算的对象。

**本仓库不声称提升 CV-QKD 的量子安全下界。** 详见
[`docs/00_summary.md`](docs/00_summary.md) 的"可声称/不可声称"清单。

## 快速开始

```bash
pip install numpy scipy matplotlib pytest

python -m pytest                 # 48 项：物理极限 + 统计一致性
python experiments/run_all.py    # E0–E8，约 6 秒
```

产出：
- `results/figures/*.png` — E1–E8 八张图
- `results/data/*.json` — 全部数值（文档数字的唯一来源）

## 目录结构

```
qCDMA-QKD/
├── docs/                       研究文档（README.md 是索引）
│   ├── initial.md              原始论文提纲（需求基线，不改动）
│   ├── 00_summary.md           核心结论与关键数字
│   ├── 01_model_and_assumptions.md   威胁模型、通信假设、符号表
│   ├── 02_protocol.md          闭环协议
│   ├── 03_cvqkd_model.md       CV-QKD 速率模型与验证
│   ├── 04_mui_analysis.md      动态 MUI 统计（核心物理修正）
│   ├── 05_security.md          安全归约、跨轮次泄漏、epsilon 合并
│   ├── 06_key_budget.md        密钥预算与 K_0
│   ├── 07_results.md           E0–E8 结果与解读
│   └── 08_open_problems.md     空白、局限、后续工作
├── src/qcdma_qkd/              核心库
│   ├── config.py               所有场景参数（唯一入口）
│   ├── cvqkd.py                GG02 速率、Holevo、有限码长、PE
│   ├── codes.py                扩频码生成与互相关
│   ├── mui.py                  MUI 统计、尾部、块平均 xi
│   ├── budget.py               认证/种子/应用三路预算，K_0
│   ├── security.py             epsilon 合并、LHL、PRF
│   ├── attack.py               MitM、盲估计、跨轮泄漏、态制备泄漏
│   ├── loop.py                 闭环协议仿真（KDF + PRF 扩码 + 滚动认证）
│   └── plotting.py             统一绘图风格
├── experiments/                E0–E8
├── tests/                      48 项校验
└── results/                    图与数据
```

## 四个量化发现

| # | 发现 | 数字 |
|---|---|---|
| 1 | 块平均 MUI 涨落**不随块长衰减** | 相对 std $=\sqrt{2/K}$，8 用户时 **53%** |
| 2 | 认证成本**与消息长度无关** | 667 消息/轮 → **0.13 bit**/轮 |
| 3 | 刷新约束基于"**已知**芯片数" | $f>0.75$（而非直觉的 $0.25$） |
| 4 | 码保密的**量子层**价值可忽略 | $L{=}4096$ 时速率损失 **<1%** |

## 环境

Python ≥ 3.9，`numpy`、`scipy`、`matplotlib`、`pytest`。无网络依赖。