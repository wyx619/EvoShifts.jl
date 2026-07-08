"""
    EngineConfig

Runtime configuration for the EvoTraits engine, including BLAS thread count,
Julia thread count, and whether branch operators should be cached.
"""
Base.@kwdef struct EngineConfig
    blas_threads::Int = 1
    julia_threads::Int = Threads.nthreads()
    cache_branch_operators::Bool = true
    use_mkl::Bool = true
end

"""
    CompactTree

Internal array-based tree cache used by all likelihood, mapping, and ancestral
state reconstruction kernels. `CompactTree` stores topology, branch lengths,
tree traversals, tip labels, and node-label metadata in a form optimized for
large-tree computation.
"""
struct CompactTree
    ntips::Int
    nnodes::Int
    nedges::Int
    parent_of_edge::Vector{Int32}
    child_of_edge::Vector{Int32}
    edge_length::Vector{Float64}
    root::Int32
    parent_of_node::Vector{Int32}
    dist_from_root::Vector{Float64}
    is_tip::BitVector
    tip_ids::Vector{Int32}
    postorder::Vector{Int32}
    postorder_internal::Vector{Int32}
    preorder::Vector{Int32}
    children::Vector{Vector{Int32}}
    first_child_edge::Vector{Int32}
    last_child_edge::Vector{Int32}
    tipname_to_id::Dict{String, Int32}
    tip_labels::Vector{String}
    node_labels::Vector{String}
end

"""
    TipData

Minimal container for tip-level discrete and continuous observations passed into
higher-level workflows.
"""
struct TipData
    discrete_states::Union{Nothing, Vector{Int32}}
    continuous_trait::Union{Nothing, Vector{Float64}}
end

