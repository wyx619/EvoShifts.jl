function _multivariate_screening_trait_scores(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    intercept_mode::Symbol = :phylogenetic_intercept,
)
    if intercept_mode !== :phylogenetic_intercept && intercept_mode !== :none
        throw(ArgumentError("Unsupported intercept_mode: $intercept_mode"))
    end
    p = length(candidates)
    m = size(trait_mat, 2)
    length(alpha_vec) == m || throw(ArgumentError("alpha_vec must have $m entries"))

    scores = zeros(Float64, p, m)
    norms = zeros(Float64, p, m)
    standardized = zeros(Float64, p, m)
    visible = falses(p, m)

    @inbounds for i in 1:m
        alpha_i = _l1ou_proposal_alpha(alpha_vec[i])
        local_tree =
            missing_context !== nothing && missing_context.has_missing ?
            missing_context.pruned_trees[i] :
            tree
        local_cache =
            missing_context !== nothing && missing_context.has_missing ?
            missing_context.pruned_caches[i] :
            cache
        local_trait =
            missing_context !== nothing && missing_context.has_missing ?
            missing_context.observed_traits[i] :
            Float64.(trait_mat[:, i])

        sigma2 = max(var(local_trait), 1e-8)
        response = _shift_screening_response(
            local_tree,
            local_trait,
            alpha_i,
            sigma2;
            intercept_mode = intercept_mode,
        )
        local_weights = _precompute_design_weights(local_tree, 1:local_tree.nedges, alpha_i)
        workspace = _tree_whiten_column_workspace(local_tree, response.edge_a, response.edge_v)
        xw = Vector{Float64}(undef, local_tree.ntips - 1)

        for j in 1:p
            full_edge = Int(candidates[j])
            local_edge =
                missing_context !== nothing && missing_context.has_missing ?
                missing_context.edge_map[full_edge, i] :
                full_edge
            if local_edge == 0 || local_tree.ntips < 2
                continue
            end
            _tree_whiten_shift_column!(
                xw,
                local_tree,
                local_cache,
                local_edge,
                local_weights[local_edge],
                response.edge_a,
                response.edge_v,
                workspace,
            )
            if response.intercept_mode === :phylogenetic_intercept
                _remove_intercept_component!(xw, response.intercept, response.intercept_norm2)
            end
            cn = LinearAlgebra.dot(xw, xw)
            cn <= eps(Float64) && continue
            score = LinearAlgebra.dot(xw, response.y)
            scores[j, i] = score
            norms[j, i] = cn
            standardized[j, i] = score / sqrt(cn)
            visible[j, i] = true
        end
    end

    return (
        scores = scores,
        norms = norms,
        standardized_scores = standardized,
        visible = visible,
    )
end

