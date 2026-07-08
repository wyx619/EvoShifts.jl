function configuration_ic(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer};
    criterion::Symbol = :mBIC,
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    max_descendant_tips::Union{Nothing, Integer} = nothing,
    max_candidate_edges::Union{Nothing, Integer} = nothing,
    candidate_sort::Symbol = :descendant_length,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    root_model::Symbol = :OUfixedRoot,
)
    fit = fit_ou_shifts(
        tree,
        trait,
        shift_edges;
        criterion = criterion,
        candidate_edges = candidate_edges,
        edge_length_threshold = edge_length_threshold,
        min_descendant_tips = min_descendant_tips,
        max_descendant_tips = max_descendant_tips,
        max_candidate_edges = max_candidate_edges,
        candidate_sort = candidate_sort,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        root_model = root_model,
    )
    return fit.score
end

function configuration_ic(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    shift_edges::AbstractVector{<:Integer};
    criterion::Symbol = :mBIC,
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    max_descendant_tips::Union{Nothing, Integer} = nothing,
    max_candidate_edges::Union{Nothing, Integer} = nothing,
    candidate_sort::Symbol = :descendant_length,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    root_model::Symbol = :OUfixedRoot,
)
    fit = fit_ou_shifts(
        tree,
        trait,
        shift_edges;
        criterion = criterion,
        candidate_edges = candidate_edges,
        edge_length_threshold = edge_length_threshold,
        min_descendant_tips = min_descendant_tips,
        max_descendant_tips = max_descendant_tips,
        max_candidate_edges = max_candidate_edges,
        candidate_sort = candidate_sort,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        root_model = root_model,
    )
    return fit.score
end

