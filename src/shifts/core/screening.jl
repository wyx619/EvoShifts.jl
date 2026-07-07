function _shift_screening_response(
    tree::CompactTree,
    values::AbstractVector{<:Real},
    alpha::Float64,
    sigma2::Float64;
    intercept_mode::Symbol = :phylogenetic_intercept,
)
    edge_a, edge_v = _shift_screening_edges(tree, alpha, sigma2)
    y_w = _tree_whiten_vector(tree, values, edge_a, edge_v)
    intercept_w, intercept_norm2 =
        if intercept_mode === :phylogenetic_intercept
            _intercept_projection_vector(_tree_whiten_vector(tree, ones(Float64, tree.ntips), edge_a, edge_v))
        elseif intercept_mode === :none
            (Float64[], 1.0)
        else
            throw(ArgumentError("Unsupported intercept_mode: $intercept_mode"))
        end
    if intercept_mode === :phylogenetic_intercept
        _remove_intercept_component!(y_w, intercept_w, intercept_norm2)
    end
    return (
        y = y_w,
        edge_a = edge_a,
        edge_v = edge_v,
        intercept = intercept_w,
        intercept_norm2 = intercept_norm2,
        intercept_mode = intercept_mode,
    )
end

function _shift_screening_column_scores(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    candidates::AbstractVector{<:Integer},
    weights::AbstractVector{<:Real},
    response::NamedTuple,
)
    length(weights) == length(candidates) || throw(ArgumentError("weights and candidates must have the same length"))
    scores = Vector{Float64}(undef, length(candidates))
    norms = Vector{Float64}(undef, length(candidates))
    workspace = _tree_whiten_column_workspace(tree, response.edge_a, response.edge_v)
    xw = Vector{Float64}(undef, tree.ntips - 1)
    @inbounds for (j, edge) in enumerate(candidates)
        _tree_whiten_shift_column!(
            xw,
            tree,
            cache,
            edge,
            Float64(weights[j]),
            response.edge_a,
            response.edge_v,
            workspace,
        )
        if response.intercept_mode === :phylogenetic_intercept
            _remove_intercept_component!(xw, response.intercept, response.intercept_norm2)
        end
        scores[j] = LinearAlgebra.dot(xw, response.y)
        norms[j] = max(LinearAlgebra.dot(xw, xw), eps(Float64))
    end
    return (scores = scores, norms = norms)
end

function _rank_shift_screening_edges(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait::AbstractVector{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha::Float64;
    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
    sigma2::Union{Nothing, Real} = nothing,
    intercept_mode::Symbol = :phylogenetic_intercept,
)
    tr = Float64.(trait)
    sig2 = sigma2 === nothing ? max(var(tr), 1e-8) : Float64(sigma2)
    w = weights === nothing ? _precompute_design_weights(tree, candidates, alpha) : weights
    response = _shift_screening_response(tree, tr, alpha, sig2; intercept_mode = intercept_mode)
    parts = _shift_screening_column_scores(tree, cache, candidates, w, response)
    ranking = sortperm(1:length(candidates); by = j -> (-abs(parts.scores[j]) / sqrt(parts.norms[j]), Int(candidates[j])))
    return (
        edges = Int[Int(candidates[j]) for j in ranking],
        scores = Float64[parts.scores[j] for j in ranking],
        norms = Float64[parts.norms[j] for j in ranking],
        standardized_scores = Float64[parts.scores[j] / sqrt(parts.norms[j]) for j in ranking],
    )
end

