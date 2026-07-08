Base.@kwdef mutable struct OUShiftConfiguration
    shift_edges::Vector{Int} = Int[]
    score::Float64 = Inf
    criterion::Symbol = :mBIC
    alpha::Vector{Float64} = Float64[]
    sigma2::Vector{Float64} = Float64[]
    loglik::Vector{Float64} = Float64[]
    theta::Any = Float64[]
    shift_values::Any = Float64[]
    shift_means::Any = Float64[]
    fitted_means::Any = Float64[]
    residuals::Any = Float64[]
    edge_optima::Any = Float64[]
    n_shifts::Int = 0
    source::Symbol = :unspecified
end

function _normalize_ou_root_model(root_model::Symbol)
    root_model === :OUfixedRoot && return :OUfixedRoot
    root_model === :OUrandomRoot && return :OUrandomRoot
    throw(ArgumentError("root_model must be :OUfixedRoot or :OUrandomRoot"))
end

Base.@kwdef struct OUShiftDetectionResult
    success::Bool = false
    model::Symbol = :OUShifts
    ntraits::Int = 1
    shift_edges::Vector{Int} = Int[]
    n_shifts::Int = 0
    alpha::Vector{Float64} = Float64[]
    sigma2::Vector{Float64} = Float64[]
    loglik::Vector{Float64} = Float64[]
    theta::Any = Float64[]
    shift_values::Any = Float64[]
    shift_means::Any = Float64[]
    fitted_means::Any = Float64[]
    residuals::Any = Float64[]
    edge_optima::Any = Float64[]
    score::Float64 = Inf
    criterion::Symbol = :mBIC
    edge_regimes::Vector{Int} = Int[]
    edge_segments::Vector{Vector{SimmapSegment}} = Vector{Vector{SimmapSegment}}()
    profile::Vector{OUShiftConfiguration} = OUShiftConfiguration[]
    diagnostics::NamedTuple = (;)
end

Base.@kwdef struct OUShiftFitResult
    success::Bool = false
    model::Symbol = :OUShiftsFit
    shift_edges::Vector{Int} = Int[]
    n_shifts::Int = 0
    alpha::Any = NaN
    sigma2::Any = NaN
    theta::Any = Float64[]
    shift_values::Any = Float64[]
    shift_means::Any = Float64[]
    fitted_means::Any = Float64[]
    residuals::Any = Float64[]
    edge_optima::Any = Float64[]
    loglik::Any = NaN
    score::Float64 = Inf
    criterion::Symbol = :mBIC
    nparams::Int = 0
    nregimes::Int = 0
    edge_regimes::Vector{Int} = Int[]
    edge_segments::Vector{Vector{SimmapSegment}} = Vector{Vector{SimmapSegment}}()
    fit::Any = nothing
    diagnostics::NamedTuple = (;)
end

Base.@kwdef struct OUShiftTreeCache
    ntips::Int = 0
    nedges::Int = 0
    root::Int32 = Int32(0)
    tree_height::Float64 = 0.0
    edge_parent::Vector{Int32} = Int32[]
    edge_child::Vector{Int32} = Int32[]
    edge_length::Vector{Float64} = Float64[]
    dist_from_root::Vector{Float64} = Float64[]
    first_tip::Vector{Int} = Int[]
    last_tip::Vector{Int} = Int[]
    tip_order::Vector{Int} = Int[]
    descendant_tip_positions::Vector{Vector{Int}} = Vector{Int}[]
    postorder::Vector{Int32} = Int32[]
    preorder::Vector{Int32} = Int32[]
    r_postorder_edge_rank::Vector{Int} = Int[]
end

