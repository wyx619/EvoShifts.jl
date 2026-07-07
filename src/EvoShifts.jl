module EvoShifts

using MKL
using LinearAlgebra
using Statistics
using Random
using NewickTree
using Optim
using DataFrames

include("core/types.jl")

Base.@kwdef struct SimmapSegment
    state::Int32
    length::Float64
end

include("core/io.jl")
include("core/criteria.jl")

include("tree/phylomap/phylomap.jl")
include("tree/prune.jl")

include("simulate/common.jl")
include("simulate/tree/io.jl")
include("simulate/tree/yule.jl")
include("simulate/traits/bm.jl")

include("continuous/checks.jl")
include("continuous/reconstruction/core/messages.jl")
include("continuous/reconstruction/core/posterior.jl")
include("continuous/univariate/shared_optimize.jl")
include("continuous/univariate/ou_common/specs.jl")
include("continuous/univariate/ou_common/params.jl")
include("continuous/univariate/ou_common/edges.jl")
include("continuous/univariate/ou_common/likelihood.jl")
include("continuous/univariate/ou_common/optimize.jl")
include("continuous/univariate/ou_common/fit.jl")

include("shifts.jl")

export EngineConfig
export CompactTree
export TipData
export SimmapSegment
export SimulatedTree

export load_newick_tree
export save_newick_tree
export to_compact_tree
export to_newick
export from_compact_tree
export to_real_tree

export compute_aic
export compute_aicc
export compute_bic

export PhyloMap
export build_phylomap
export R_node_table
export R_edge_table
export phylomap_node_table
export phylomap_edge_table
export keep_tip
export drop_tip

export simulate_yule_simtree
export simulate_yule_tree
export simulate_mvbm1

export OUShiftDetectionResult
export OUShiftConfiguration
export OUShiftFitResult
export detect_ou_shifts
export fit_ou_shifts
export configuration_ic
export align_traits_to_tree
export shift_detection_summary
export shift_detection_summary_table
export profile_configurations
export get_shift_configuration
export best_shift_configuration
export build_shift_tree_cache
export filter_candidate_edges
export shift_edges_to_edge_segments
export shift_edge_signature
export shift_edge_signatures
export shift_edges_from_signatures
export shift_edge_table
export merge_convergent_regimes

function set_engine_blas_threads!(n::Integer)
    BLAS.set_num_threads(n)
    return BLAS.get_num_threads()
end

export set_engine_blas_threads!

end # module
