function _mv_full_theta_from_visible(
    corrected::AbstractVector{<:Integer},
    local_theta::AbstractVector{<:Real},
    missing_context::MVShiftMissingContext,
    trait_index::Integer;
    fill_invisible::Bool = false,
)
    theta_full = fill(NaN, length(corrected) + 1)
    theta_full[1] = Float64(local_theta[1])
    local_idx = 1
    @inbounds for (full_idx, edge0) in enumerate(corrected)
        edge = Int(edge0)
        if missing_context.edge_map[edge, Int(trait_index)] != 0
            local_idx += 1
            theta_full[full_idx + 1] = Float64(local_theta[local_idx])
        end
    end
    if fill_invisible
        @inbounds for i in 2:length(theta_full)
            isnan(theta_full[i]) && (theta_full[i] = theta_full[1])
        end
    end
    return theta_full
end

function _fill_mv_config_from_edges!(
    cfg::OUShiftConfiguration,
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    shift_edges::AbstractVector{<:Integer},
;
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    ctx = _mv_exact_scoring_context(
        tree,
        cache,
        trait_mat;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        missing_context = missing_context,
        refit_cache = refit_cache,
        warm_start_cache = warm_start_cache,
        root_model = root_model,
    )
    return _fill_mv_config_from_edges!(cfg, ctx, shift_edges)
end

function _fill_mv_config_from_edges!(
    cfg::OUShiftConfiguration,
    ctx::_MVExactScoringContext,
    shift_edges::AbstractVector{<:Integer},
)
    m = size(ctx.trait_mat, 2)
    corrected = Int.(shift_edges)
    alphas = Float64[]
    sigmas = Float64[]
    lls = Float64[]
    thetas = Vector{Float64}[]
    sizehint!(alphas, m)
    sizehint!(sigmas, m)
    sizehint!(lls, m)
    sizehint!(thetas, m)

    if isempty(corrected)
        for i in 1:m
            fit = _fit_ou1_for_shift_detection(ctx.tree, @view(ctx.trait_mat[:, i]);
                optimization = ctx.optimization,
                max_iterations = ctx.max_iterations,
                rel_tol = ctx.rel_tol,
                root_model = ctx.root_model)
            fit.profile.success || return false
            push!(alphas, fit.bundle.alpha[1])
            push!(sigmas, fit.bundle.sigma2[1])
            push!(lls, fit.profile.loglik)
            push!(thetas, copy(fit.bundle.theta))
        end
    else
        for i in 1:m
            refit = _cached_refit_mv_trait(ctx, i, corrected)
            refit.success || return false
            push!(alphas, refit.alpha)
            push!(sigmas, refit.sigma2)
            push!(lls, refit.loglik)
            push!(thetas, copy(refit.theta))
        end
    end

    cfg.alpha = alphas
    cfg.sigma2 = sigmas
    cfg.loglik = lls
    cfg.shift_edges = corrected
    edge_segments = shift_edges_to_edge_segments(ctx.tree, corrected)
    theta_mat = Matrix{Float64}(undef, length(corrected) + 1, m)
    if _mv_context_has_missing(ctx)
        fill!(theta_mat, NaN)
        for i in 1:m
            theta_mat[:, i] .= _mv_full_theta_from_visible(corrected, thetas[i], ctx.missing_context, i)
        end
    else
        theta_mat .= hcat(thetas...)
    end
    fitted = Matrix{Float64}(undef, ctx.tree.ntips, m)
    shift_values = fill(NaN, length(corrected), m)
    shift_means = fill(NaN, length(corrected), m)
    edge_optima = Matrix{Float64}(undef, ctx.tree.nedges, m)
    for i in 1:m
        if _mv_context_has_missing(ctx)
            visible_full = _mv_trait_visible_full_edges(ctx, i, corrected)
            theta_full = _mv_full_theta_from_visible(corrected, thetas[i], ctx.missing_context, i; fill_invisible = true)
            fitted[:, i] .= _ou_shift_fitted_means(ctx.tree, edge_segments, theta_full, alphas[i])
            if !isempty(visible_full)
                vals = _shift_values_from_theta(ctx.tree, edge_segments, visible_full, theta_full)
                for (idx, edge) in enumerate(visible_full)
                    full_idx = findfirst(==(edge), corrected)
                    shift_values[full_idx, i] = vals[idx]
                end
                shift_means[:, i] .= _shift_means_from_shift_values(ctx.tree, corrected, @view(shift_values[:, i]), alphas[i])
            end
            edge_optima[:, i] .= _edge_optima_from_theta(edge_segments, theta_full)
        else
            fitted[:, i] .= _ou_shift_fitted_means(ctx.tree, edge_segments, @view(theta_mat[:, i]), alphas[i])
            if !isempty(corrected)
                shift_values[:, i] .= _shift_values_from_theta(ctx.tree, edge_segments, corrected, @view(theta_mat[:, i]))
                shift_means[:, i] .= _shift_means_from_shift_values(ctx.tree, corrected, @view(shift_values[:, i]), alphas[i])
            end
            edge_optima[:, i] .= _edge_optima_from_theta(edge_segments, @view(theta_mat[:, i]))
        end
    end
    cfg.theta = theta_mat
    cfg.shift_values = shift_values
    cfg.shift_means = shift_means
    cfg.fitted_means = fitted
    cfg.residuals = Matrix{Float64}(ctx.trait_mat) .- fitted
    cfg.edge_optima = edge_optima
    cfg.n_shifts = length(corrected)
    cfg.score = _score_configuration_full_mv(ctx, lls, cfg.n_shifts, corrected)
    cfg.criterion = ctx.criterion
    return true
end

function _score_mv_config_from_edges!(
    cfg::OUShiftConfiguration,
    ctx::_MVExactScoringContext,
    shift_edges::AbstractVector{<:Integer};
    store_details::Bool = true,
    loglik_buffer::Union{Nothing, Vector{Float64}} = nothing,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
)
    corrected = Int.(shift_edges)
    if !store_details
        score_res = _score_mv_edges(ctx, corrected;
            loglik_buffer = loglik_buffer,
            mbic_covered_workspace = mbic_covered_workspace,
            mbic_edges_workspace = mbic_edges_workspace)
        score_res.success || return false
        cfg.shift_edges = corrected
        cfg.n_shifts = length(corrected)
        cfg.score = score_res.score
        cfg.criterion = ctx.criterion
        return true
    end

    return _fill_mv_config_from_edges!(cfg, ctx, corrected)
end

function _score_mv_config_from_edges!(
    cfg::OUShiftConfiguration,
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    shift_edges::AbstractVector{<:Integer},
;
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    store_details::Bool = true,
    loglik_buffer::Union{Nothing, Vector{Float64}} = nothing,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
    profile_workspace_caches::Union{Nothing, Vector{Dict{Int, _ShiftCrossproductWorkspace}}} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    ctx = _mv_exact_scoring_context(
        tree,
        cache,
        trait_mat;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        missing_context = missing_context,
        refit_cache = refit_cache,
        warm_start_cache = warm_start_cache,
        profile_workspace_caches = profile_workspace_caches,
        root_model = root_model,
    )
    return _score_mv_config_from_edges!(
        cfg,
        ctx,
        shift_edges;
        store_details = store_details,
        loglik_buffer = loglik_buffer,
        mbic_covered_workspace = mbic_covered_workspace,
        mbic_edges_workspace = mbic_edges_workspace,
    )
end

function _fill_best_mv_config!(
    configs::Vector{OUShiftConfiguration},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real};
    criterion::Symbol,
    optimization::Symbol,
    max_iterations::Integer,
    rel_tol::Float64,
    missing_context::Union{Nothing, MVShiftMissingContext},
    refit_cache::Dict{Tuple, NamedTuple},
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}},
    profile_workspace_caches::Union{Nothing, Vector{Dict{Int, _ShiftCrossproductWorkspace}}},
    root_model::Symbol,
    fill_best::Bool = true,
)
    warm_cache = warm_start_cache === nothing ? Dict{Int, NamedTuple}() : warm_start_cache
    ctx = _mv_exact_scoring_context(
        tree,
        cache,
        trait_mat;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        missing_context = missing_context,
        refit_cache = refit_cache,
        warm_start_cache = warm_cache,
        profile_workspace_caches = profile_workspace_caches,
        root_model = root_model,
    )
    return _fill_best_config!(
        configs,
        cfg -> _fill_mv_config_from_edges!(cfg, ctx, cfg.shift_edges);
        fill_best = fill_best,
    )
end
