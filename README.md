# EvoShifts

`EvoShifts` is a standalone Ornstein-Uhlenbeck shift-detection package extracted
from `EvoTraits.jl`.

`EvoShifts` 是从 `EvoTraits.jl` 中拆出的独立 Ornstein-Uhlenbeck 最适值变化检测包。

## Overview / 概览

The package focuses on a compact shifts workflow:

- tree representation and Newick I/O
- proposal generation
- exact refit and information-criterion scoring
- multivariate missing-data support
- convergent regime merging

当前包聚焦于一套独立的 shifts 工作流：

- 树结构与 Newick 读写
- proposal 生成
- exact refit 与信息准则评分
- 多变量缺失数据支持
- convergent regime merge

## Source Layout / 源码结构

Current first-level `src/` directories:

- `base`: shared types and generic criteria helpers
- `core`: tree utilities, phylomap, pruning kernel, shift cache and edge identities
- `io`: Newick and tree conversion helpers
- `simulate`: minimal simulation support used by tests
- `model`: Gaussian likelihood core and OU model helpers
- `proposal`: candidate proposal builders
- `refit`: fixed-configuration refit and IC helpers
- `search`: configuration scoring, pruning, and multivariate search logic
- `convergence`: convergent regime merge logic
- `api`: public alignment, detection, and summary entrypoints

当前 `src/` 第一层目录为：

- `base`
- `core`
- `io`
- `simulate`
- `model`
- `proposal`
- `refit`
- `search`
- `convergence`
- `api`

## Public API / 主要公开接口

Main exported interfaces include:

- tree and I/O: `load_newick_tree`, `save_newick_tree`, `to_compact_tree`
- simulation: `simulate_yule_simtree`, `simulate_yule_tree`, `simulate_mvbm1`
- shifts: `detect_ou_shifts`, `fit_ou_shifts`, `configuration_ic`
- alignment and summaries: `align_traits_to_tree`, `shift_detection_summary`, `shift_detection_summary_table`
- result/profile helpers: `profile_configurations`, `get_shift_configuration`, `best_shift_configuration`
- shift utilities: `build_shift_tree_cache`, `filter_candidate_edges`, `shift_edges_to_edge_segments`
- convergence: `merge_convergent_regimes`

## Tests / 测试

Top-level test files mirror the current package layout:

- `test/core.jl`
- `test/proposal.jl`
- `test/refit.jl`
- `test/api.jl`
- `test/convergence.jl`

Run the package and tests with:

```powershell
julia --project=. -e "using EvoShifts"
julia --project=. test/runtests.jl
julia --project=. test/run_subset.jl api
```

Each top-level test file is also directly runnable, for example:

```powershell
julia --project=. test/proposal.jl
```

## Validation Status / Validation 状态

`validation/` is not treated as part of the core package surface.

- validation-related dependencies remain because validation scripts still exist in the repo
- validation scripts are not the primary supported interface of the package
- the main supported workflows are `using EvoShifts` and the `test/` suite

`validation/` 当前不作为主包交付边界的一部分：

- 仓库中仍保留部分 validation 脚本，因此相关依赖仍存在
- validation 脚本不是当前主支持接口
- 当前主支持路径是 `using EvoShifts` 与 `test/` 测试套件
