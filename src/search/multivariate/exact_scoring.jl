function _score_mv_edges(
    ctx::_MVExactScoringContext,
    shift_edges::AbstractVector{<:Integer};
    loglik_buffer::Union{Nothing, Vector{Float64}} = nothing,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
    profile_workspaces::Union{Nothing, Vector{_ShiftCrossproductWorkspace}} = nothing,
)
    m = size(ctx.trait_mat, 2)
    lls = loglik_buffer === nothing ? Vector{Float64}(undef, m) : loglik_buffer
    length(lls) == m || resize!(lls, m)
    for i in 1:m
        profile_workspace =
            profile_workspaces === nothing || i > length(profile_workspaces) ?
            nothing :
            profile_workspaces[i]
        refit = _cached_refit_mv_trait(ctx, i, shift_edges; profile_workspace = profile_workspace)
        refit.success || return (success = false, score = Inf)
        lls[i] = refit.loglik
    end
    score = _score_configuration_full_mv(
        ctx,
        lls,
        length(shift_edges),
        shift_edges;
        mbic_covered_workspace = mbic_covered_workspace,
        mbic_edges_workspace = mbic_edges_workspace,
    )
    return (success = true, score = score)
end

function _score_mv_edges(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    shift_edges::AbstractVector{<:Integer},
    refit_cache::Dict{Tuple, NamedTuple};
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    loglik_buffer::Union{Nothing, Vector{Float64}} = nothing,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
    profile_workspaces::Union{Nothing, Vector{_ShiftCrossproductWorkspace}} = nothing,
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
    return _score_mv_edges(
        ctx,
        shift_edges;
        loglik_buffer = loglik_buffer,
        mbic_covered_workspace = mbic_covered_workspace,
        mbic_edges_workspace = mbic_edges_workspace,
        profile_workspaces = profile_workspaces,
    )
end

function _mv_edge_elimination_workspaces(
    ctx::_MVExactScoringContext,
    n_trial_edges::Integer,
)
    n_trial_edges > 0 || return _ShiftCrossproductWorkspace[]
    k = Int(n_trial_edges) + 2
    return [
        _shift_crossproduct_workspace(_mv_trait_tree(ctx, i), k)
        for i in 1:size(ctx.trait_mat, 2)
    ]
end

function _mv_edge_elimination_workspaces(
    tree::CompactTree,
    trait_mat::AbstractMatrix{<:Real},
    n_trial_edges::Integer,
    missing_context::Union{Nothing, MVShiftMissingContext},
)
    ctx = _mv_exact_scoring_context(
        tree,
        OUShiftTreeCache(),
        trait_mat;
        missing_context = missing_context,
    )
    return _mv_edge_elimination_workspaces(ctx, n_trial_edges)
end

function _prune_shift_config_mv(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    shift_edges::AbstractVector{<:Integer},
    refit_cache::Dict{Tuple, NamedTuple};
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    max_edge_elimination_passes::Integer = 1,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
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
    pruned = _prune_shift_config_mv_scored(
        ctx,
        shift_edges;
        max_edge_elimination_passes = max_edge_elimination_passes,
    )
    return Int.(pruned.shift_edges)
end

function _prune_shift_config_mv_scored(
    ctx::_MVExactScoringContext,
    shift_edges::AbstractVector{<:Integer};
    max_edge_elimination_passes::Integer = 1,
    initial_score = nothing,
)
    current = Int.(shift_edges)
    best =
        initial_score === nothing ?
        _score_mv_edges(ctx, current) :
        initial_score
    best.success || return (shift_edges = current, removed_edges = Int[], score = best, n_trials = 0, n_passes = 0)

    loglik_buffer = Vector{Float64}(undef, size(ctx.trait_mat, 2))
    mbic_covered_workspace = Vector{Bool}(undef, ctx.cache.ntips)
    mbic_edges_workspace = Int[]
    profile_workspaces =
        ctx.profile_workspace_caches === nothing ?
        _mv_edge_elimination_workspaces(ctx, max(length(current) - 1, 0)) :
        nothing
    score_fn = edges -> _score_mv_edges(ctx, edges;
        loglik_buffer = loglik_buffer,
        mbic_covered_workspace = mbic_covered_workspace,
        mbic_edges_workspace = mbic_edges_workspace,
        profile_workspaces = profile_workspaces)
    pruned = _prune_shift_edges_by_score(
        current,
        score_fn;
        max_edge_elimination_passes = max_edge_elimination_passes,
        min_prunable_shifts = 3,
        initial_score = best,
        on_accept = (edge, accepted, score) -> begin
            if ctx.profile_workspace_caches === nothing
                profile_workspaces = _mv_edge_elimination_workspaces(
                    ctx, max(length(accepted) - 1, 0),
                )
            end
            nothing
        end,
    )
    return pruned
end

function _score_and_sort_configs_mv!(
    configs::Vector{OUShiftConfiguration},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
;
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    initial_alpha::Union{Nothing, AbstractVector{<:Real}} = nothing,
    edge_elimination::Bool = true,
    max_edge_elimination_passes::Integer = 1,
    max_edge_elimination_configs::Union{Nothing, Integer} = 8,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    profile_workspace_caches::Union{Nothing, Vector{Dict{Int, _ShiftCrossproductWorkspace}}} = nothing,
    root_model::Symbol = :OUfixedRoot,
    parallel_scoring::Bool = Threads.nthreads() > 1,
    fill_best::Bool = true,
)
    root_model = _normalize_ou_root_model(root_model)
    warm_cache = warm_start_cache === nothing ? Dict{Int, NamedTuple}() : warm_start_cache
    loglik_buffer = Vector{Float64}(undef, size(trait_mat, 2))
    mbic_covered_workspace = Vector{Bool}(undef, cache.ntips)
    mbic_edges_workspace = Int[]
    profile_caches =
        profile_workspace_caches === nothing ?
        _mv_profile_workspace_caches(size(trait_mat, 2)) :
        profile_workspace_caches
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
        profile_workspace_caches = profile_caches,
        root_model = root_model,
    )
    if initial_alpha !== nothing
        @inbounds for i in 1:min(length(initial_alpha), size(trait_mat, 2))
            a = Float64(initial_alpha[i])
            if isfinite(a) && a > 0.0
                warm_cache[i] = (alpha = a, sigma2 = nothing, theta = Float64[])
            end
        end
    end
    initial_scoring_time = @elapsed begin
        if parallel_scoring && Threads.nthreads() > 1 && length(configs) > 1
            Threads.@threads :dynamic for idx in eachindex(configs)
                cfg = configs[idx]
                local_refit_cache = Dict{Tuple, NamedTuple}()
                local_warm_cache = Dict{Int, NamedTuple}()
                if initial_alpha !== nothing
                    @inbounds for i in 1:min(length(initial_alpha), size(trait_mat, 2))
                        a = Float64(initial_alpha[i])
                        if isfinite(a) && a > 0.0
                            local_warm_cache[i] = (alpha = a, sigma2 = nothing, theta = Float64[])
                        end
                    end
                end
                local_loglik_buffer = Vector{Float64}(undef, size(trait_mat, 2))
                local_mbic_covered_workspace = Vector{Bool}(undef, cache.ntips)
                local_mbic_edges_workspace = Int[]
                local_ctx = _mv_exact_scoring_context(
                    tree,
                    cache,
                    trait_mat;
                    criterion = criterion,
                    optimization = optimization,
                    max_iterations = max_iterations,
                    rel_tol = rel_tol,
                    missing_context = missing_context,
                    refit_cache = local_refit_cache,
                    warm_start_cache = local_warm_cache,
                    root_model = root_model,
                )
                corrected = _order_shift_edges_like(correct_shift_configuration_l1ou(cache, cfg.shift_edges), cfg.shift_edges)
                _score_mv_config_from_edges!(cfg, local_ctx, corrected;
                    store_details = false,
                    loglik_buffer = local_loglik_buffer,
                    mbic_covered_workspace = local_mbic_covered_workspace,
                    mbic_edges_workspace = local_mbic_edges_workspace)
            end
        else
            for cfg in configs
                corrected = _order_shift_edges_like(correct_shift_configuration_l1ou(cache, cfg.shift_edges), cfg.shift_edges)
                _score_mv_config_from_edges!(cfg, ctx, corrected;
                    store_details = false,
                    loglik_buffer = loglik_buffer,
                    mbic_covered_workspace = mbic_covered_workspace,
                    mbic_edges_workspace = mbic_edges_workspace)
            end
        end
    end
    sort_filter_time = @elapsed begin
        _sort_scorable_configs!(configs)
    end
    edge_elimination_time = 0.0
    edge_elimination_trials = 0
    edge_elimination_removed = 0
    if edge_elimination && (max_edge_elimination_configs === nothing || max_edge_elimination_configs > 0)
        edge_elimination_time = @elapsed begin
            ncheck =
                max_edge_elimination_configs === nothing ?
                length(configs) :
                min(length(configs), Int(max_edge_elimination_configs))
            if parallel_scoring && Threads.nthreads() > 1 && ncheck > 1
                local_trials = zeros(Int, ncheck)
                local_removed = zeros(Int, ncheck)
                Threads.@threads :dynamic for i in 1:ncheck
                    local_refit_cache = Dict{Tuple, NamedTuple}()
                    local_warm_cache = Dict{Int, NamedTuple}()
                    if initial_alpha !== nothing
                        @inbounds for j in 1:min(length(initial_alpha), size(trait_mat, 2))
                            a = Float64(initial_alpha[j])
                            if isfinite(a) && a > 0.0
                                local_warm_cache[j] = (alpha = a, sigma2 = nothing, theta = Float64[])
                            end
                        end
                    end
                    local_ctx = _mv_exact_scoring_context(
                        tree,
                        cache,
                        trait_mat;
                        criterion = criterion,
                        optimization = optimization,
                        max_iterations = max_iterations,
                        rel_tol = rel_tol,
                        missing_context = missing_context,
                        refit_cache = local_refit_cache,
                        warm_start_cache = local_warm_cache,
                        root_model = root_model,
                    )
                    pruned = _prune_shift_config_mv_scored(
                        local_ctx,
                        configs[i].shift_edges;
                        max_edge_elimination_passes = max_edge_elimination_passes,
                        initial_score = (success = true, score = configs[i].score),
                    )
                    counts = _record_edge_elimination_result!(configs[i], pruned; criterion = criterion)
                    local_trials[i] = counts.n_trials
                    local_removed[i] = counts.n_removed
                end
                edge_elimination_trials += sum(local_trials)
                edge_elimination_removed += sum(local_removed)
            else
                for i in 1:ncheck
                    pruned = _prune_shift_config_mv_scored(
                        ctx,
                        configs[i].shift_edges;
                        max_edge_elimination_passes = max_edge_elimination_passes,
                        initial_score = (success = true, score = configs[i].score),
                    )
                    counts = _record_edge_elimination_result!(configs[i], pruned; criterion = criterion)
                    edge_elimination_trials += counts.n_trials
                    edge_elimination_removed += counts.n_removed
                end
            end
            _sort_scorable_configs!(configs)
        end
    end
    final_fill_time = @elapsed begin
        _fill_best_config!(
            configs,
            cfg -> _fill_mv_config_from_edges!(cfg, ctx, cfg.shift_edges);
            fill_best = fill_best,
        )
    end
    return (
        initial_scoring = initial_scoring_time,
        sort_filter = sort_filter_time,
        edge_elimination = edge_elimination_time,
        edge_elimination_trials = edge_elimination_trials,
        edge_elimination_removed = edge_elimination_removed,
        final_fill = final_fill_time,
        fill_best = fill_best,
        parallel_scoring = parallel_scoring && Threads.nthreads() > 1,
        nthreads = Threads.nthreads(),
    )
end

function _score_path_family_mv!(
    configs::Vector{OUShiftConfiguration},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
;
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    initial_alpha::Union{Nothing, AbstractVector{<:Real}} = nothing,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    refit_cache::Dict{Tuple, NamedTuple} = Dict{Tuple, NamedTuple}(),
    warm_start_cache::Union{Nothing, Dict{Int, NamedTuple}} = nothing,
    profile_workspace_caches::Union{Nothing, Vector{Dict{Int, _ShiftCrossproductWorkspace}}} = nothing,
    root_model::Symbol = :OUfixedRoot,
    parallel_scoring::Bool = Threads.nthreads() > 1,
    fill_best::Bool = true,
)
    return _score_and_sort_configs_mv!(
        configs,
        tree,
        cache,
        trait_mat;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        initial_alpha = initial_alpha,
        edge_elimination = false,
        missing_context = missing_context,
        refit_cache = refit_cache,
        warm_start_cache = warm_start_cache,
        profile_workspace_caches = profile_workspace_caches,
        root_model = root_model,
        parallel_scoring = parallel_scoring,
        fill_best = fill_best,
    )
end
