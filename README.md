# qCDMA-QKD

q-CDMA-CV-QKD 中**码分发闭环**的研究实现：把"CDMA 码如何安全分发与更新"
这一信任空白变为可计算、可归约、可预算的对象。

**本仓库不声称提升 CV-QKD 的量子安全下界。** 详见
[`docs/00_summary.md`](docs/00_summary.md) 的"可声称/不可声称"清单。

## 快速开始

```bash
pip install numpy scipy matplotlib pytest

python experiments/run_all.py    # E0–E9，约 6 秒（先跑这个，产出 JSON）
python -m pytest                 # 187 项：物理极限 + 统计一致性 + 单位约定 + 文档一致性
python tools/doc_check.py        # 单独检查 docs/ 的数字是否仍等于 results/data/
```

产出：
- `results/figures/*.png` — E1–E9 九张图
- `results/data/*.json` — 全部数值（文档数字的**唯一来源**；改代码后重跑 E0（即 `run_all.py`））

### 为什么有 `tools/doc_check.py`

本项目曾三次出现"代码改对了、文档里的数字没跟上、但所有测试仍然全绿"：
认证成本停在摊销值 `0.132 bit`（实际每轮换钥，应为 `207.7 bit`）、
闭环停在渐近速率（实际已切到有限码长，`4.31` → `3.33` Mbit）、
$\epsilon$ 组合仍在引用旧的轮数上限。
**单元测试看不见散文。**

该脚本把 `results/data/*.json` 当作唯一真值，断言：
1. 文献中每个关键数字都能从 JSON 重新算出来（`CLAIMS`）；
2. 已废弃的数值不得再出现（`RETIRED`），除非该行显式在讨论历史。

`tests/test_docs_consistency.py` 把这两条接进 `pytest`，因此文档漂移会直接
**让 CI 变红**，而不是等读者发现。

## 目录结构

```
qCDMA-QKD/
├── tools/
│   └── doc_check.py            文档数字 ↔ results/data 一致性校验
├── docs/                       研究文档（README.md 是索引）
│   ├── initial.md              原始论文提纲（需求基线，不改动）
│   ├── 00_summary.md           核心结论与关键数字
│   ├── 01_model_and_assumptions.md   威胁模型、通信假设、符号/单位表
│   ├── 02_protocol.md          闭环协议
│   ├── 03_cvqkd_model.md       CV-QKD 速率模型与验证
│   ├── 04_mui_analysis.md      动态 MUI 统计（核心物理修正）
│   ├── 05_security.md          安全归约、跨轮次泄漏、刷新约束推导、epsilon 合并
│   ├── 06_key_budget.md        密钥预算与 K_0
│   ├── 07_results.md           E0–E9 结果与解读
│   └── 08_open_problems.md     空白、局限、后续工作
├── src/qcdma_qkd/              核心库
│   ├── config.py               所有场景参数（chip_rate / L / 种子，唯一入口）
│   ├── cvqkd.py                GG02 速率、Holevo、有限码长、PE
│   ├── codes.py                扩频码生成与互相关
│   ├── mui.py                  MUI 统计、尾部、块平均 xi
│   ├── budget.py               认证/种子/应用三路预算，K_0
│   ├── security.py             epsilon 合并、LHL、PRF
│   ├── attack.py               MitM、盲估计、跨轮泄漏、刷新约束（含精确尾部版）
│   ├── loop.py                 闭环协议仿真（KDF + PRF 扩码 + 滚动认证）
│   └── plotting.py             统一绘图风格
├── experiments/                E0–E9
├── tests/                      79 项校验（5 个文件）
└── results/                    图与数据
```

## 四个量化发现

| # | 发现 | 数字 |
|---|---|---|
| 1 | 块平均 MUI 涨落**不随块长衰减** | 相对 std $=\sqrt{2/K}$，8 用户时 **53%** |
| 2 | 认证成本**与消息长度无关** | 667 消息/轮 → **0.13 bit**/轮 |
| 3 | 刷新约束基于"**已知**芯片数"，判据是 $\epsilon_{\text{code}}$ 目标；**仅提取器分支** | $f\ge0.7886$（而非直觉的 $0.25$） |
| 4 | 码保密的**量子层**价值可忽略 | $L{=}4096$ 时速率损失 **<1%** |

## 码层安全模型：PRF + 全刷新（**本项目最终方案**）

```python
CDMAConfig(code_model="prf", refresh_fraction=1.0)   # 默认
```

| 项 | PRF + $f=1$（采用） | 信息论提取器（备选） |
|---|---|---|
| 码层安全 | **计算**（PRF 假设） | 信息论 |
| 刷新要求 | 无（$f_{\min}=0$） | $f\ge0.7886$ |
| 跨轮次相关 | 无 | 每轮保留 $n_{\text{reused}}$ 个 chip |
| 额外假设 | PRF | 无 |

PRF 在**本地**由种子展开，所以"每轮全新码"的通信代价为 0（只有计算代价），
且 $f=1$ 把跨轮次相关确定性清成 0（实测 $\langle c_n,c_{n-1}\rangle/L=0.0115
\approx O(1/\sqrt L)$）。

**代价必须写明**：PRF 只给**计算**安全。若要求码层信息论安全，256 bit 种子
装不下 $L=1024$ 个已知 chip 的线性方程，就**必须**降到 $f\ge0.7886$——
`docs/05_security.md` §5 的全部推导就是在把这个代价算清楚。
切换：`CDMAConfig(code_model="extractor")`，`check_refresh()` 会在违约时抛错。

## 速率与单位（易错点）

每用户速率**不是**自由参数，由硬件码片率与扩频因子共同决定：

$$R_s=\frac{R_c}{L}=\frac{10^9}{1024}=0.977\ \text{Msymbol/s}$$

1 Gchip/s + $L=1024$ 时，25 km、8 用户、含 MUI 的每用户速率约
**42 kbit/s**（不是 Mbit/s）。$L$ 在速率上要再付一次因子 $L$。

$L$ 同时出现在三处、方向不同，因此存在**最优扩频因子** $L^\ast$：

| 效应 | 随 $L$ | 方向 |
|---|---|---|
| MUI 过噪声 $\xi\propto 1/L$ | ↓ | 越大越好 |
| 符号率 $R_s=R_c/L$ | ↓ | 越大越差 |
| 刷新约束 $f_{\min}$（**仅提取器分支**） | ↑ | 越大越差 |

前两项竞争给出 $L^\ast\approx1024$–2048（见 E9）。

## 常见陷阱

本题材有几处极易踩反的地方，`tests/` 全部锁定：

0. **码层安全类型必须显式声明** — PRF（计算安全，无刷新约束）vs
   信息论提取器（有 $f_{\min}$ 约束）。样例模型选前者；把两者的结论混着说
   是最容易犯的错。
1. **认证成本按消息条数计，不是按比特数** — MAC 密钥不随消息长度增长。
2. **刷新约束看 Eve *已知*的芯片数** — $f\ge1-(\text{seed}-k_{\text{req}})/(Lp)$，
   不是"新芯片数 > 种子长度"（那会给出 $f>0.25$，方向反了）；
   判据也不能写成 $H_{\min}>0$（那允许 $\epsilon_{\text{code}}=0.5$），
   必须显式给出 $\epsilon_{\text{code}}^{*}$，$k_{\text{req}}=\log_2(1/\epsilon^{*})$。
   $\text{seed}>Lp+k_{\text{req}}$ 时约束自动消失（$f_{\min}=0$），**不是不可行**。
   唯一真正的不可行区间是 $\text{seed}\le k_{\text{req}}=\log_2(1/\epsilon_{code})$。
3. **`symbol_rate` 必须由 `chip_rate/L` 派生** — 自由设定会隐含 Tchip/s 级
   的不可能硬件，并把速率高估 $L$ 倍。

## 环境

Python ≥ 3.9，`numpy`、`scipy`、`matplotlib`、`pytest`。无网络依赖，全部闭式计算。