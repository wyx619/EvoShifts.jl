function _run_multivariate_screening_round(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    round_label::Symbol = :multivariate_screening_alpha0,
    max_shifts::Integer = typemax(Int),
    n_lambda::Integer = 100,
    proposal_max_iterations::Integer = 1000,
    proposal_tol::Float64 = 1e-6,
    vote_threshold::Float64 = 0.0,
    intercept_mode::Symbol = :phylogenetic_intercept,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    column_cache::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
    keep_path::Bool = false,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    proposal = _propose_shift_configs_multivariate_screening(
        tree, cache, trait_mat, candidates, alpha_vec;
        max_shifts = max_shifts,
        n_lambda = n_lambda,
        max_iterations = proposal_max_iterations,
        tol = proposal_tol,
        intercept_mode = intercept_mode,
        vote_threshold = vote_threshold,
        missing_context = missing_context,
        column_cache = column_cache,
        keep_path = keep_path,
        root_model = root_model,
    )
    configs = OUShiftConfiguration[]
    for edges in proposal.configs
        push!(configs, OUShiftConfiguration(
            shift_edges = edges,
            n_shifts = length(edges),
            source = round_label,
        ))
    end
    return configs
end

function _run_multivariate_screening_prefix_anchor_round(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    round_label::Symbol = :multivariate_screening_prefix_anchor_alpha0,
    max_shifts::Integer = typemax(Int),
    max_prefix_edges::Integer = typemax(Int),
    min_shared_score::Union{Nothing, Float64} = nothing,
    vote_threshold::Float64 = 0.0,
    intercept_mode::Symbol = :phylogenetic_intercept,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    proposal = _propose_shift_configs_multivariate_screening_prefix_anchor(
        tree, cache, trait_mat, candidates, alpha_vec;
        max_shifts = max_shifts,
        max_prefix_edges = max_prefix_edges,
        min_shared_score = min_shared_score,
        intercept_mode = intercept_mode,
        vote_threshold = vote_threshold,
        missing_context = missing_context,
        root_model = root_model,
    )
    configs = OUShiftConfiguration[]
    for edges in proposal.configs
        push!(configs, OUShiftConfiguration(
            shift_edges = edges,
            n_shifts = length(edges),
            source = round_label,
        ))
    end
    return configs
end

function _run_multivariate_path_round(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    round_label::Symbol = :multivariate_path_alpha0,
    max_shifts::Integer = typemax(Int),
    n_lambda::Integer = 100,
    proposal_max_iterations::Integer = 1000,
    proposal_tol::Float64 = 1e-6,
    vote_threshold::Float64 = 0.0,
    intercept_mode::Symbol = :phylogenetic_intercept,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    column_cache::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
    keep_path::Bool = false,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    proposal = _propose_shift_configs_multivariate_group_path(
        tree, cache, trait_mat, candidates, alpha_vec;
        max_shifts = max_shifts,
        n_lambda = n_lambda,
        max_iterations = proposal_max_iterations,
        tol = proposal_tol,
        intercept_mode = intercept_mode,
        vote_threshold = vote_threshold,
        missing_context = missing_context,
        column_cache = column_cache,
        keep_path = keep_path,
        root_model = root_model,
    )
    configs = OUShiftConfiguration[]
    for edges in proposal.configs
        push!(configs, OUShiftConfiguration(
            shift_edges = edges,
            n_shifts = length(edges),
            source = round_label,
        ))
    end
    return configs
end

