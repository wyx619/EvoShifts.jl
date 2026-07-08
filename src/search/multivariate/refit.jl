function _l1ou_missing_trait_mbic_df2(
    full_cache::OUShiftTreeCache,
    missing_context::MVShiftMissingContext,
    trait_index::Integer,
    shift_edges::AbstractVector{<:Integer},
)
    j = Int(trait_index)
    pcache = missing_context.pruned_caches[j]
    nobs = pcache.ntips
    nobs > 0 || return Inf
    df2 = 3.0 * log(nobs)
    visible_edges = _mv_shift_visible_edges(missing_context, j, shift_edges)
    isempty(visible_edges) && return df2
    sort_edges_l1ou!(pcache, visible_edges)

    covered = falses(full_cache.ntips)
    @inbounds for local_edge in visible_edges
        local_rank = pcache.r_postorder_edge_rank[local_edge]
        1 <= local_rank <= length(missing_context.full_rank_to_edge) || return Inf
        full_edge = missing_context.full_rank_to_edge[local_rank]
        full_edge != 0 || return Inf
        nnew = 0
        for tip_pos in full_cache.descendant_tip_positions[full_edge]
            if !covered[tip_pos]
                nnew += 1
            end
        end
        nnew > 0 || return Inf
        df2 += log(nnew)
        for tip_pos in full_cache.descendant_tip_positions[full_edge]
            covered[tip_pos] = true
        end
    end

    nuncovered = 0
    @inbounds for tip_pos in 1:nobs
        !covered[tip_pos] && (nuncovered += 1)
    end
    nuncovered > 0 || return Inf
    return df2 + log(nuncovered)
end

function _score_configuration_full_mv_impl(
    cache::OUShiftTreeCache,
    loglik_vec::Vector{Float64},
    n_shifts::Integer,
    shift_edges::AbstractVector{<:Integer},
    n::Integer;
    criterion::Symbol = :mBIC,
    merge_map::Dict{Int,Int} = Dict{Int,Int}(),
    has_missing::Bool = false,
    observed_count = i -> n,
    visible_edge_count = i -> length(shift_edges),
    missing_mbic_df2 = i -> Inf,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
)
    total_loglik = sum(loglik_vec)
    if criterion === :AICc
        m = length(loglik_vec)
        total_k =
            if isempty(merge_map)
                Int(n_shifts) + _ou_shift_parameter_df(n_shifts; ntraits = m)
            else
                _convergent_parameter_df(length(shift_edges), _convergent_nshiftvals(merge_map, shift_edges); ntraits = m)
            end
        return _compute_aicc(total_loglik, total_k, m * n)
    elseif criterion === :BIC
        m = length(loglik_vec)
        if isempty(merge_map)
            shift_penalty = Int(n_shifts) * log(n)
            trait_penalty = 0.0
            if has_missing
                @inbounds for i in eachindex(loglik_vec)
                    nobs = observed_count(i)
                    nlocal = visible_edge_count(i)
                    trait_penalty += (nlocal + 3) * log(nobs)
                end
            else
                trait_penalty = m * (Int(n_shifts) + 3) * log(n)
            end
            return -2.0 * total_loglik + shift_penalty + trait_penalty
        else
            n_shift_values = _convergent_nshiftvals(merge_map, shift_edges)
            total_k = _convergent_parameter_df(length(shift_edges), n_shift_values; ntraits = m)
            return -2.0 * total_loglik + total_k * log(n)
        end
    elseif criterion === :mBIC
        if has_missing
            parts = try
                _l1ou_mbic_penalty_parts(cache, shift_edges)
            catch err
                err isa ArgumentError || rethrow()
                return Inf
            end
            penalty = parts.df1
            @inbounds for i in eachindex(loglik_vec)
                trait_df2 = missing_mbic_df2(i)
                isfinite(trait_df2) || return Inf
                penalty += trait_df2
            end
            return -2.0 * total_loglik + penalty
        end
        parts = try
            _l1ou_mbic_penalty_parts(cache, shift_edges, mbic_covered_workspace, mbic_edges_workspace)
        catch err
            err isa ArgumentError || rethrow()
            return Inf
        end
        return parts.df1 - 2.0 * total_loglik + length(loglik_vec) * parts.df2
    end
    throw(ArgumentError("Unsupported criterion: $criterion"))
end

function _score_configuration_full_mv(
    cache::OUShiftTreeCache,
    loglik_vec::Vector{Float64},
    n_shifts::Integer,
    shift_edges::AbstractVector{<:Integer},
    n::Integer,
;
    criterion::Symbol = :mBIC,
    merge_map::Dict{Int,Int} = Dict{Int,Int}(),
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
)
    has_missing = missing_context !== nothing && missing_context.has_missing
    return _score_configuration_full_mv_impl(
        cache,
        loglik_vec,
        n_shifts,
        shift_edges,
        n;
        criterion = criterion,
        merge_map = merge_map,
        has_missing = has_missing,
        observed_count = i -> missing_context.observed_counts[i],
        visible_edge_count = i -> length(_mv_shift_visible_edges(missing_context, i, shift_edges)),
        missing_mbic_df2 = i -> _l1ou_missing_trait_mbic_df2(cache, missing_context, i, shift_edges),
        mbic_covered_workspace = mbic_covered_workspace,
        mbic_edges_workspace = mbic_edges_workspace,
    )
end

@inline function _mv_refit_cache_key(trait_index::Integer, shift_edges::AbstractVector{<:Integer})
    return (Int(trait_index), _shift_edges_key(shift_edges))
end

function _mv_profile_workspace_caches(ntraits::Integer)
    return [Dict{Int, _ShiftCrossproductWorkspace}() for _ in 1:Int(ntraits)]
end

function _mv_profile_workspace_from_cache!(
    workspace_cache::Union{Nothing, Dict{Int, _ShiftCrossproductWorkspace}},
    tree::CompactTree,
    n_shift_edges::Integer,
)
    workspace_cache === nothing && return nothing
    k = Int(n_shift_edges) + 2
    return get!(workspace_cache, k) do
        _shift_crossproduct_workspace(tree, k)
    end
end

mutable struct _MVExactScoringContext
    tree::CompactTree
    cache::OUShiftTreeCache
    trait_mat::AbstractMatrix
    criterion::Symbol
    optimization::Symbol
    max_iterations::Int
    rel_tol::Float64
    missing_context::Union{Nothing, MVShiftMissingContext}
    refit_cache::Dict{Tuple, NamedTuple}
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}}
    profile_workspace_caches::Union{Nothing, Vector{Dict{Int, _ShiftCrossproductWorkspace}}}
    root_model::Symbol
end

function _mv_exact_scoring_context(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real};
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    profile_workspace_caches::Union{Nothing, Vector{Dict{Int, _ShiftCrossproductWorkspace}}} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    return _MVExactScoringContext(
        tree,
        cache,
        trait_mat,
        criterion,
        optimization,
        Int(max_iterations),
        rel_tol,
        missing_context,
        refit_cache,
        warm_start_cache,
        profile_workspace_caches,
        _normalize_ou_root_model(root_model),
    )
end

@inline function _mv_trait_profile_workspace_cache(
    ctx::_MVExactScoringContext,
    trait_index::Integer,
)
    caches = ctx.profile_workspace_caches
    caches === nothing && return nothing
    i = Int(trait_index)
    return i > length(caches) ? nothing : caches[i]
end

@inline function _mv_context_has_missing(ctx::_MVExactScoringContext)
    return ctx.missing_context !== nothing && ctx.missing_context.has_missing
end

@inline function _mv_trait_tree(ctx::_MVExactScoringContext, trait_index::Integer)
    return _mv_context_has_missing(ctx) ? ctx.missing_context.pruned_trees[Int(trait_index)] : ctx.tree
end

@inline function _mv_trait_vector(ctx::_MVExactScoringContext, trait_index::Integer)
    return _mv_context_has_missing(ctx) ?
        _mv_shift_observed_trait(ctx.missing_context, ctx.trait_mat, trait_index) :
        @view(ctx.trait_mat[:, Int(trait_index)])
end

@inline function _mv_trait_visible_edges(
    ctx::_MVExactScoringContext,
    trait_index::Integer,
    shift_edges::AbstractVector{<:Integer},
)
    return _mv_context_has_missing(ctx) ?
        _mv_shift_visible_edges(ctx.missing_context, trait_index, shift_edges) :
        Int.(shift_edges)
end

@inline function _mv_trait_visible_full_edges(
    ctx::_MVExactScoringContext,
    trait_index::Integer,
    shift_edges::AbstractVector{<:Integer},
)
    return _mv_context_has_missing(ctx) ?
        _mv_shift_visible_full_edges(ctx.missing_context, trait_index, shift_edges) :
        Int.(shift_edges)
end

@inline function _mv_trait_observed_count(ctx::_MVExactScoringContext, trait_index::Integer)
    return _mv_context_has_missing(ctx) ? ctx.missing_context.observed_counts[Int(trait_index)] : ctx.tree.ntips
end

@inline function _mv_trait_profile_workspace(
    ctx::_MVExactScoringContext,
    trait_index::Integer,
    n_shift_edges::Integer,
)
    return _mv_profile_workspace_from_cache!(
        _mv_trait_profile_workspace_cache(ctx, trait_index),
        _mv_trait_tree(ctx, trait_index),
        n_shift_edges,
    )
end

function _l1ou_missing_trait_mbic_df2(
    ctx::_MVExactScoringContext,
    trait_index::Integer,
    shift_edges::AbstractVector{<:Integer},
)
    _mv_context_has_missing(ctx) || return Inf
    return _l1ou_missing_trait_mbic_df2(ctx.cache, ctx.missing_context, trait_index, shift_edges)
end

function _score_configuration_full_mv(
    ctx::_MVExactScoringContext,
    loglik_vec::Vector{Float64},
    n_shifts::Integer,
    shift_edges::AbstractVector{<:Integer};
    merge_map::Dict{Int,Int} = Dict{Int,Int}(),
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
)
    return _score_configuration_full_mv_impl(
        ctx.cache,
        loglik_vec,
        n_shifts,
        shift_edges,
        ctx.tree.ntips;
        criterion = ctx.criterion,
        merge_map = merge_map,
        has_missing = _mv_context_has_missing(ctx),
        observed_count = i -> _mv_trait_observed_count(ctx, i),
        visible_edge_count = i -> length(_mv_trait_visible_edges(ctx, i, shift_edges)),
        missing_mbic_df2 = i -> _l1ou_missing_trait_mbic_df2(ctx, i, shift_edges),
        mbic_covered_workspace = mbic_covered_workspace,
        mbic_edges_workspace = mbic_edges_workspace,
    )
end

function _cached_refit_mv_trait(
    refit_cache::Dict{Tuple, NamedTuple},
    tree::CompactTree,
    trait_mat::AbstractMatrix{<:Real},
    trait_index::Integer,
    shift_edges::AbstractVector{<:Integer};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    profile_workspace::Union{Nothing, _ShiftCrossproductWorkspace} = nothing,
    profile_workspace_cache::Union{Nothing, Dict{Int, _ShiftCrossproductWorkspace}} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    ctx = _mv_exact_scoring_context(
        tree,
        OUShiftTreeCache(),
        trait_mat;
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        missing_context = missing_context,
        refit_cache = refit_cache,
        warm_start_cache = warm_start_cache,
        root_model = root_model,
    )
    return _cached_refit_mv_trait(
        ctx,
        trait_index,
        shift_edges;
        profile_workspace = profile_workspace,
        profile_workspace_cache = profile_workspace_cache,
    )
end

function _cached_refit_mv_trait(
    ctx::_MVExactScoringContext,
    trait_index::Integer,
    shift_edges::AbstractVector{<:Integer};
    profile_workspace::Union{Nothing, _ShiftCrossproductWorkspace} = nothing,
    profile_workspace_cache::Union{Nothing, Dict{Int, _ShiftCrossproductWorkspace}} =
        _mv_trait_profile_workspace_cache(ctx, trait_index),
)
    full_key = _mv_refit_cache_key(trait_index, shift_edges)
    if haskey(ctx.refit_cache, full_key)
        return ctx.refit_cache[full_key]
    end

    visible_edges = _mv_trait_visible_edges(ctx, trait_index, shift_edges)
    key = _mv_context_has_missing(ctx) ? _mv_refit_cache_key(trait_index, visible_edges) : full_key
    if haskey(ctx.refit_cache, key)
        refit = ctx.refit_cache[key]
        ctx.refit_cache[full_key] = refit
        return refit
    end

    warm = ctx.warm_start_cache === nothing ? nothing : get(ctx.warm_start_cache, Int(trait_index), nothing)
    tree = _mv_trait_tree(ctx, trait_index)
    y = _mv_trait_vector(ctx, trait_index)
    workspace =
        profile_workspace === nothing ?
        _mv_profile_workspace_from_cache!(profile_workspace_cache, tree, length(visible_edges)) :
        profile_workspace
    refit = _refit_ou_shift_config(
        tree,
        y,
        visible_edges;
        optimization = ctx.optimization,
        max_iterations = ctx.max_iterations,
        rel_tol = ctx.rel_tol,
        start_alpha = warm === nothing ? nothing : warm.alpha,
        start_sigma2 = warm === nothing ? nothing : warm.sigma2,
        start_theta_regimes =
            warm === nothing || length(warm.theta) != length(visible_edges) + 1 ?
            nothing :
            warm.theta,
        profile_workspace = workspace,
        profile_alpha_floor = _mv_context_has_missing(ctx) ? 1e-8 : 1e-4,
        root_model = ctx.root_model,
    )
    ctx.refit_cache[key] = refit
    ctx.refit_cache[full_key] = refit
    if ctx.warm_start_cache !== nothing && refit.success
        ctx.warm_start_cache[Int(trait_index)] = refit
    end
    return refit
end

