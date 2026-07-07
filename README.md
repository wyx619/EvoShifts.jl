# EvoShifts

`EvoShifts` 是从 `EvoTraits.jl` 中拆出的 Ornstein-Uhlenbeck（奥恩斯坦-乌伦贝克）最适值变化检测子系统。

当前阶段目标：

- 把 `src/continuous/shifts` 及其最小基建闭包独立出来
- 保留现有 `CompactTree + shift cache + tree pruning kernel + exact refit/IC`
- 先让子系统结构和 tests 独立成形
- `validation/` 暂不迁移

## 当前目录

```text
src/
  EvoShifts.jl
  core/                  # EngineConfig / CompactTree / tree I/O / criteria
  tree/                  # phylomap 与轻量 tree pruning
  simulate/              # shifts tests 需要的最小模拟能力
  continuous/            # shifts 依赖的连续模型底层 OU helper
  shifts/                # 主子系统
  shifts.jl              # shifts include 入口
test/
  continuous/shifts/     # 从 EvoTraits 迁移来的 shifts 配套测试
```

## 当前状态

- `continuous/shifts/*` 五个子集测试已迁移并通过
- `test/runtests.jl` 已可在本地通过
- 目前测试入口采用 `include("../src/EvoShifts.jl")` 方式加载源码

## 待继续处理

1. 修好独立 `Pkg` 加载路径，达到直接 `using EvoShifts`
2. 继续清理从 `EvoTraits` 复制过来的命名和注释
3. 再决定是否迁移 `validation/shift_detection/*`
