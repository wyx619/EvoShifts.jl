# EvoShifts.jl

`EvoShifts.jl` is a Julia package for detecting evolutionary shifts in trait optima under Ornstein-Uhlenbeck (OU) models on ultrametric phylogenetic trees.

The package implements a tree-native workflow for OU shift analysis: construct a compact tree representation, align traits to tip labels, generate candidate shift configurations, refit promising configurations with the full OU likelihood, select a model with an information criterion, and map the selected shifts back to reproducible branch identities.

It supports univariate traits, multivariate traits with shifts shared across lineages, and partially missing multivariate observations. The public API is deliberately centered on analysis rather than on the internal proposal, search, and likelihood machinery.

## Contents

- [What The Package Does](#what-the-package-does)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input Requirements](#input-requirements)
- [Univariate Workflow](#univariate-workflow)
- [Multivariate Workflow](#multivariate-workflow)
- [Fixed Configurations And Model Scoring](#fixed-configurations-and-model-scoring)
- [Interpreting Results](#interpreting-results)
- [Branch Identity And R Compatibility](#branch-identity-and-r-compatibility)
- [Convergent Regimes](#convergent-regimes)
- [Performance And Reproducibility](#performance-and-reproducibility)
- [API Overview](#api-overview)
- [Package Layout](#package-layout)
- [Testing](#testing)
- [Validation](#validation)
- [Reference](#reference)

## What The Package Does

An OU shift model allows the expected trait optimum to change on selected branches of a phylogeny. A shift configuration is a set of branch ids; each selected branch starts a new adaptive regime for all descendant lineages unless a later shift supersedes it.

`detect_ou_shifts` performs automatic configuration selection. The search combines candidate-edge filtering, proposal generation, exact refits, and information-criterion ranking. `fit_ou_shifts` evaluates a configuration supplied by the caller. Both return structured result objects rather than loose vectors or tables.

The core workflow supports:

- univariate OU shift detection;
- multivariate detection with a common shift configuration across traits;
- `:mBIC`, `:BIC`, and `:AICc` scoring;
- fixed-root and random-root OU models;
- partially missing multivariate trait matrices;
- branch mapping to R/ape-style postorder edge ids;
- profile inspection, reporting tables, and convergent-regime merging.

## Installation

Use the repository as a Julia project. Start Julia with an explicit thread count:

```powershell
julia --project=. -t auto -e "using Pkg; Pkg.instantiate(); using EvoShifts"
```

Run Julia examples with automatic thread detection: `julia --project=. -t auto example.jl`. Julia selects a useful thread count from the CPU resources available on the machine.

From the repository root, verify that package loading works:

```powershell
julia --project=. -t auto -e "using EvoShifts; println(nameof(EvoShifts))"
```

The project declares compatibility with Julia `1.10`, `1.11`, and `1.12`.

## Quick Start

The following example reads a Newick tree and a CSV table containing one row per tip. The `taxon_col` argument makes the alignment explicit, so the table may be in any row order.

```julia
# Run with: julia --project=. -t auto quick_start.jl
using EvoShifts
using CSV
using DataFrames

tree = to_compact_tree(load_newick_tree("tree.tre"))
data = CSV.read("traits.csv", DataFrame)

trait = align_traits_to_tree(
    tree,
    data;
    taxon_col = :taxon,
    trait_cols = [:trait],
)

result = detect_ou_shifts(
    tree,
    trait;
    criterion = :mBIC,
    root_model = :OUrandomRoot,
)

shift_detection_summary(result)
shift_edge_table(tree; edges = result.shift_edges)
```

The expected CSV schema is:

```text
taxon,trait
species_a,1.23
species_b,0.87
species_c,1.95
```

## Input Requirements

### Trees

- Detection requires an ultrametric tree with branch lengths.
- `load_newick_tree` reads a Newick file; `to_compact_tree` converts it to the package's `CompactTree` representation.
- Tip labels must be unique.
- `save_newick_tree`, `from_compact_tree`, and `to_newick` provide the corresponding output and conversion utilities.

### Traits

- Trait rows correspond to tips and trait columns correspond to traits.
- `align_traits_to_tree` should be preferred for tabular data because it validates labels, detects duplicate taxa, and reorders rows to match the tree.
- A raw vector or matrix is accepted only when its rows are already in tree-tip order.
- Infinite values are not permitted.
- A univariate trait must be complete.
- A multivariate trait matrix may contain `NaN` values, but every tip needs at least one observed trait and every trait needs at least one observed value.

### Shift Edges

Shift configurations use internal edge ids in `1:tree.nedges`. Treat these ids as package coordinates, not as stable external identifiers. For reporting, map them through `shift_edge_table`, `phylomap_edge_table`, or `R_edge_table`.

## Univariate Workflow

For a single trait, `align_traits_to_tree` returns a vector. The automatic detector uses candidate filtering, screening and path proposals, exact OU refits, and final criterion ranking.

```julia
# Run with: julia --project=. -t auto univariate.jl
tree = to_compact_tree(load_newick_tree("tree.tre"))
data = CSV.read("trait.csv", DataFrame)

trait = align_traits_to_tree(
    tree,
    data;
    taxon_col = :species,
    trait_cols = [:leaf_area],
)

result = detect_ou_shifts(
    tree,
    trait;
    max_shifts = 20,
    criterion = :mBIC,
    edge_length_threshold = eps(Float64),
    min_descendant_tips = 1,
    root_model = :OUrandomRoot,
)

@show result.n_shifts result.score result.shift_edges
```

Use `candidate_edges` when the analysis should be restricted to a pre-defined subset of branches. The same candidate filters should be supplied to `fit_ou_shifts` when comparing a fixed configuration under `:mBIC`, because the candidate universe affects the penalty.

## Multivariate Workflow

For a matrix with two or more trait columns, shifts are selected jointly: all traits share the selected shift branches, while trait-specific OU parameters and likelihood contributions are retained.

```julia
# Run with: julia --project=. -t auto multivariate.jl
traits = align_traits_to_tree(
    tree,
    data;
    taxon_col = :species,
    trait_cols = [:trait_1, :trait_2, :trait_3],
)

result = detect_ou_shifts(
    tree,
    traits;
    max_shifts = 60,
    criterion = :mBIC,
    root_model = :OUrandomRoot,
)

summary = shift_detection_summary(result)
```

Partially observed rows are supported in this workflow. Use `NaN` in a numeric matrix, or `missing` / recognized missing strings in a `DataFrame`; the alignment layer converts supported missing values to the internal representation.

## Fixed Configurations And Model Scoring

Use `fit_ou_shifts` when the configuration is known in advance, originates from another tool, or needs independent verification after automatic detection.

```julia
# Run with: julia --project=. -t auto fixed_configuration.jl
fixed = fit_ou_shifts(
    tree,
    traits,
    result.shift_edges;
    criterion = :mBIC,
    root_model = :OUrandomRoot,
)

score = configuration_ic(
    tree,
    traits,
    result.shift_edges;
    criterion = :mBIC,
    root_model = :OUrandomRoot,
)
```

`fit_ou_shifts` returns an `OUShiftFitResult` with fitted OU parameters, likelihood, score, branch regimes, fitted means, residuals, and diagnostics. `configuration_ic` returns the corresponding criterion value for a configuration. Lower scores are preferred.

Automatic detection retains a profile of considered configurations. Inspect it with:

```julia
# Run with: julia --project=. -t auto profile.jl
profile = profile_configurations(result)
best = best_shift_configuration(result)
candidate = get_shift_configuration(result, 3)
```

## Interpreting Results

`detect_ou_shifts` returns an `OUShiftDetectionResult` with these principal fields:

| Field | Meaning |
|---|---|
| `success` | Whether the analysis completed with a selected result. |
| `shift_edges` | Selected internal branch ids. |
| `n_shifts` | Number of selected shift branches. |
| `score` | Final information-criterion score; lower is better. |
| `loglik` | Trait-wise likelihood contributions. |
| `alpha` | Trait-wise OU attraction parameters. |
| `sigma2` | Trait-wise diffusion variances. |
| `theta` | Fitted regime optima. |
| `shift_values` | Estimated changes associated with selected shifts. |
| `edge_regimes` | Regime assignment for each tree edge. |
| `profile` | Scored candidate configurations retained by the search. |
| `diagnostics` | Search settings, timing, candidate filters, source data, and warnings. |

Use `shift_detection_summary(result)` for a compact named summary and `shift_detection_summary_table(result)` for a one-row `DataFrame` suitable for export.

## Branch Identity And R Compatibility

The package separates its internal edge ids from externally reproducible branch identities. Use a `PhyloMap` to translate edges to node pairs, tip anchors, and R/ape-compatible ranks.

```julia
# Run with: julia --project=. -t auto branch_mapping.jl
map = build_phylomap(tree)

internal_table = shift_edge_table(tree, result.shift_edges; map = map)
r_edges = R_edge_table(tree; order = :postorder, map = map)
all_edges = phylomap_edge_table(tree; map = map)
```

For a saved R/ape postorder configuration, convert ranks into internal ids before refitting:

```julia
# Run with: julia --project=. -t auto r_compatibility.jl
r_postorder_edges = [12, 48, 97]
internal_edges = EvoShifts.evotraits_edge_ids_from_R_postorder(
    tree,
    r_postorder_edges;
    map = map,
)

fit = fit_ou_shifts(tree, traits, internal_edges; criterion = :mBIC)
```

The conversion helpers are intentionally precise about ordering. Do not assume that an internal edge id is equal to an R edge row number.

## Convergent Regimes

`merge_convergent_regimes` is a post-detection analysis. It does not search for additional shift branches and it does not remove the shift branches selected by `detect_ou_shifts`. Instead, it tests whether separate shift-originated regimes can share the same fitted OU optimum.

For example, an initial model may assign distinct optima to shifts on edges `10`, `27`, and `55`. A convergent fit can retain all three shift locations while assigning edges `10` and `27` to one shared optimum. This represents independent lineage changes toward the same adaptive regime rather than the absence of either shift.

This is the counterpart of the default backward-search workflow in `l1ou::estimate_convergent_regimes`. It repeatedly evaluates candidate regime merges under an information criterion. The current API does not expose l1ou's separate single-trait `rr` method.

```julia
# Run with: julia --project=. -t auto convergence.jl
merged = merge_convergent_regimes(
    result;
    criterion = :BIC,
)
```

The returned object is a new `OUShiftDetectionResult` with `model == :OUShiftsConvergent`. The original detection result is not modified. Inspect `merged.diagnostics.merge_map`, `merged.edge_regimes`, and `merged.edge_optima` to recover the accepted regime grouping and its fitted branch-level interpretation.

## Performance And Reproducibility

EvoShifts is at least 50x faster than R `l1ou` on comparable OU shift-detection workloads. This advantage comes from the compact tree representation, cached likelihood calculations, candidate pruning, and threaded multivariate scoring. The exact ratio depends on tree size, trait dimension, search settings, and hardware; benchmark both implementations with the same data and configuration.

The package uses Julia threads for portions of multivariate scoring. Set the thread count when starting Julia, not after the process is running:

```powershell
julia --project=. -t auto script.jl
```

For stable runtime measurements, control BLAS threading separately:

```julia
# Run with: julia --project=. -t auto blas.jl
using EvoShifts
set_engine_blas_threads!(1)
```

Record `result.diagnostics`, the Julia thread count, BLAS thread count, the tree file, trait column names, candidate filters, criterion, and root model alongside any published analysis. These choices affect the search space or numerical execution and are needed for reproducible comparisons.

## API Overview

| Task | Functions |
|---|---|
| Tree input and conversion | `load_newick_tree`, `save_newick_tree`, `to_compact_tree`, `from_compact_tree`, `to_newick` |
| Trait alignment | `align_traits_to_tree` |
| Automatic detection | `detect_ou_shifts` |
| Fixed refit and scoring | `fit_ou_shifts`, `configuration_ic` |
| Configuration inspection | `profile_configurations`, `get_shift_configuration`, `best_shift_configuration` |
| Result summaries | `shift_detection_summary`, `shift_detection_summary_table` |
| Branch reporting | `shift_edge_table`, `R_edge_table`, `phylomap_edge_table`, `build_phylomap` |
| Shift utilities | `build_shift_tree_cache`, `filter_candidate_edges`, `shift_edges_to_edge_segments` |
| Convergence | `merge_convergent_regimes` |
| Tree utilities | `keep_tip`, `drop_tip` |
| Simulation helpers | `simulate_yule_simtree`, `simulate_yule_tree`, `simulate_mvbm1` |

## Package Layout

The first-level modules separate stable responsibilities:

- `base`: shared types and information-criterion helpers.
- `core`: tree structures, phylomap, shift identities, caches, and pruning kernels.
- `io`: Newick parsing, writing, and tree conversion.
- `simulate`: compact simulation support.
- `model`: Gaussian message passing and OU likelihood machinery.
- `proposal`: candidate generation.
- `refit`: fixed-configuration fitting and configuration queries.
- `search`: scoring, pruning, and path search.
- `convergence`: convergent-regime merging.
- `api`: user-facing alignment, detection, and summaries.

## Testing

Run the complete suite from the repository root:

```powershell
julia --project=. -t auto test/runtests.jl
```

Run one subsystem:

```powershell
julia --project=. -t auto test/run_subset.jl api
julia --project=. -t auto test/proposal.jl
```

The top-level test files are `test/core.jl`, `test/proposal.jl`, `test/refit.jl`, `test/api.jl`, and `test/convergence.jl`. Each is directly runnable.

## Validation

The `validation/` directory contains optional empirical alignment scripts and is not required for normal package use. These scripts may require an R installation and the `ape` and `l1ou` R packages.

The repository contains multivariate checks that compare fixed configurations and automatic detection against saved `l1ou` fits. The checked datasets agree on selected shift count, information-criterion score to numerical tolerance, and R/ape postorder branch identities.

## Reference

The methodology follows the OU shift-detection framework described in:

> Khabbazian, M., Kriebel, R., Rohe, K., and Ane, C. (2016). Fast and accurate detection of evolutionary shifts in Ornstein-Uhlenbeck models. *Methods in Ecology and Evolution*, 7(7), 811-824.
