Base.@kwdef struct _MVPathSearchSummary
    raw_path_configs::Vector{OUShiftConfiguration} = OUShiftConfiguration[]
    compressed_path_configs::Vector{OUShiftConfiguration} = OUShiftConfiguration[]
    edge_eliminated_configs::Vector{OUShiftConfiguration} = OUShiftConfiguration[]
    diagnostics::NamedTuple = (;)
end

@inline function _mv_path_source(round_index::Integer)
    return Int(round_index) == 1 ? :path_round1_raw : :path_round2_raw
end

@inline function _mv_compressed_source()
    return :path_compressed
end

@inline function _mv_edge_eliminated_source()
    return :edge_elimination
end

@inline function _mv_edge_set_score_key(shift_edges::AbstractVector{<:Integer})
    sorted = sort!(Int.(shift_edges))
    return Tuple(sorted)
end

function _mv_score_edges_memoized!(
    ctx::_MVExactScoringContext,
    shift_edges::AbstractVector{<:Integer},
    score_cache::Dict{Tuple, NamedTuple},
    score_cache_lock::Union{Nothing, ReentrantLock};
    loglik_buffer::Union{Nothing, Vector{Float64}} = nothing,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
    profile_workspaces::Union{Nothing, Vector{_ShiftCrossproductWorkspace}} = nothing,
    stats::Union{Nothing, Base.RefValue{NamedTuple{(:hits, :misses), Tuple{Int, Int}}}} = nothing,
)
    key = _mv_edge_set_score_key(shift_edges)
    cached =
        if score_cache_lock === nothing
            get(score_cache, key, nothing)
        else
            lock(score_cache_lock) do
                get(score_cache, key, nothing)
            end
        end
    if cached !== nothing
        if stats !== nothing
            s = stats[]
            stats[] = (hits = s.hits + 1, misses = s.misses)
        end
        return cached
    end

    score_res = _score_mv_edges(
        ctx,
        shift_edges;
        loglik_buffer = loglik_buffer,
        mbic_covered_workspace = mbic_covered_workspace,
        mbic_edges_workspace = mbic_edges_workspace,
        profile_workspaces = profile_workspaces,
    )
    if score_cache_lock === nothing
        score_cache[key] = score_res
    else
        lock(score_cache_lock) do
            get!(score_cache, key, score_res)
        end
    end
    if stats !== nothing
        s = stats[]
        stats[] = (hits = s.hits, misses = s.misses + 1)
    end
    return score_res
end

function _mv_push_raw_path_config!(
    out::Vector{OUShiftConfiguration},
    cache::OUShiftTreeCache,
    raw_edges::AbstractVector{<:Integer},
    round_index::Integer,
)
    corrected = _order_shift_edges_like(correct_shift_configuration_l1ou(cache, raw_edges), raw_edges)
    push!(out, OUShiftConfiguration(
        shift_edges = corrected,
        n_shifts = length(corrected),
        source = _mv_path_source(round_index),
    ))
    return out
end

function _mv_collect_raw_path_configs(
    cache::OUShiftTreeCache,
    configs1::AbstractVector{OUShiftConfiguration},
    configs2::AbstractVector{OUShiftConfiguration},
)
    raw = OUShiftConfiguration[]
    sizehint!(raw, length(configs1) + length(configs2))
    for cfg in configs1
        _mv_push_raw_path_config!(raw, cache, cfg.shift_edges, 1)
    end
    for cfg in configs2
        _mv_push_raw_path_config!(raw, cache, cfg.shift_edges, 2)
    end
    return raw
end

function _mv_skip_consecutive_duplicate_configs!(
    configs::Vector{OUShiftConfiguration},
)
    isempty(configs) && return configs
    write_idx = 1
    prev = copy(configs[1].shift_edges)
    for read_idx in 2:length(configs)
        current = configs[read_idx].shift_edges
        if current == prev
            continue
        end
        write_idx += 1
        write_idx != read_idx && (configs[write_idx] = configs[read_idx])
        prev = copy(current)
    end
    resize!(configs, write_idx)
    return configs
end

function _mv_shift_stability_table(configs::AbstractVector{OUShiftConfiguration})
    freq = Dict{Int, Int}()
    first_seen = Dict{Int, Int}()
    round_mask = Dict{Int, Int}()
    round1_freq = Dict{Int, Int}()
    round2_freq = Dict{Int, Int}()
    for (cfg_idx, cfg) in enumerate(configs)
        round_bit =
            cfg.source === :path_round1_raw ? 0x01 :
            cfg.source === :path_round2_raw ? 0x02 : 0x00
        seen_local = Set{Int}()
        for edge in cfg.shift_edges
            edge_i = Int(edge)
            edge_i in seen_local && continue
            push!(seen_local, edge_i)
            freq[edge_i] = get(freq, edge_i, 0) + 1
            first_seen[edge_i] = get(first_seen, edge_i, cfg_idx)
            round_mask[edge_i] = get(round_mask, edge_i, 0) | round_bit
            if round_bit == 0x01
                round1_freq[edge_i] = get(round1_freq, edge_i, 0) + 1
            elseif round_bit == 0x02
                round2_freq[edge_i] = get(round2_freq, edge_i, 0) + 1
            end
        end
    end
    return (
        freq = freq,
        first_seen = first_seen,
        round_mask = round_mask,
        round1_freq = round1_freq,
        round2_freq = round2_freq,
    )
end

function _mv_compress_path_configs!(
    configs::Vector{OUShiftConfiguration},
    max_shifts::Integer,
)
    filter!(cfg -> cfg.n_shifts <= Int(max_shifts), configs)
    _mv_skip_consecutive_duplicate_configs!(configs)
    for cfg in configs
        cfg.source = _mv_compressed_source()
    end
    return configs
end

function _mv_edge_elimination_pass_scored(
    ctx::_MVExactScoringContext,
    score_cache::Dict{Tuple, NamedTuple},
    score_cache_lock::Union{Nothing, ReentrantLock},
    shift_edges::AbstractVector{<:Integer};
    initial_score = nothing,
    loglik_buffer::Union{Nothing, Vector{Float64}} = nothing,
    mbic_covered_workspace::Union{Nothing, Vector{Bool}} = nothing,
    mbic_edges_workspace::Union{Nothing, Vector{Int}} = nothing,
    profile_workspaces::Union{Nothing, Vector{_ShiftCrossproductWorkspace}} = nothing,
    memo_stats::Union{Nothing, Base.RefValue{NamedTuple{(:hits, :misses), Tuple{Int, Int}}}} = nothing,
)
    current = Int.(shift_edges)
    best =
        initial_score === nothing ?
        _mv_score_edges_memoized!(
            ctx,
            current,
            score_cache,
            score_cache_lock;
            loglik_buffer = loglik_buffer,
            mbic_covered_workspace = mbic_covered_workspace,
            mbic_edges_workspace = mbic_edges_workspace,
            profile_workspaces = profile_workspaces,
            stats = memo_stats,
        ) :
        initial_score
    best.success || return (shift_edges = current, removed_edges = Int[], score = best, n_trials = 0, n_passes = 0)
    length(current) < 3 && return (shift_edges = current, removed_edges = Int[], score = best, n_trials = 0, n_passes = 0)

    removed = Int[]
    n_trials = 0
    local_loglik_buffer =
        loglik_buffer === nothing ? Vector{Float64}(undef, size(ctx.trait_mat, 2)) : loglik_buffer
    local_mbic_covered_workspace =
        mbic_covered_workspace === nothing ? Vector{Bool}(undef, ctx.cache.ntips) : mbic_covered_workspace
    local_mbic_edges_workspace =
        mbic_edges_workspace === nothing ? Int[] : mbic_edges_workspace
    local_profile_workspaces =
        profile_workspaces === nothing ?
        (ctx.profile_workspace_caches === nothing ?
            _mv_edge_elimination_workspaces(ctx, max(length(current) - 1, 0)) :
            nothing) :
        profile_workspaces

    for edge in copy(current)
        edge in current || continue
        trial_edges = Int[e for e in current if e != edge]
        n_trials += 1
        trial = _mv_score_edges_memoized!(
            ctx,
            trial_edges,
            score_cache,
            score_cache_lock;
            loglik_buffer = local_loglik_buffer,
            mbic_covered_workspace = local_mbic_covered_workspace,
            mbic_edges_workspace = local_mbic_edges_workspace,
            profile_workspaces = local_profile_workspaces,
            stats = memo_stats,
        )
        if trial.success && trial.score < best.score
            current = trial_edges
            best = trial
            push!(removed, edge)
            if profile_workspaces === nothing && ctx.profile_workspace_caches === nothing
                local_profile_workspaces = _mv_edge_elimination_workspaces(ctx, max(length(current) - 1, 0))
            end
        end
    end

    return (shift_edges = current, removed_edges = removed, score = best, n_trials = n_trials, n_passes = 1)
end

function _mv_edge_elimination_configs!(
    configs::Vector{OUShiftConfiguration},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    stats;
    criterion::Symbol = :mBIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    initial_alpha::Union{Nothing, AbstractVector{<:Real}} = nothing,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    root_model::Symbol = :OUfixedRoot,
    parallel_scoring::Bool = Threads.nthreads() > 1,
)
    root_model = _normalize_ou_root_model(root_model)
    n_trials_total = 0
    n_removed_total = 0
    score_cache = Dict{Tuple, NamedTuple}()
    score_cache_lock = parallel_scoring && Threads.nthreads() > 1 ? ReentrantLock() : nothing
    memo_stats = Ref((hits = 0, misses = 0))
    edge_elimination_time = @elapsed begin
        if parallel_scoring && Threads.nthreads() > 1 && length(configs) > 1
            local_trials = zeros(Int, length(configs))
            local_removed = zeros(Int, length(configs))
            nslots = Threads.maxthreadid()
            local_ctxs = Vector{_MVExactScoringContext}(undef, nslots)
            local_loglik_buffers = Vector{Vector{Float64}}(undef, nslots)
            local_mbic_covered_workspaces = Vector{Vector{Bool}}(undef, nslots)
            local_mbic_edges_workspaces = Vector{Vector{Int}}(undef, nslots)
            local_profile_workspaces = Vector{Union{Nothing, Vector{_ShiftCrossproductWorkspace}}}(undef, nslots)
            for tid in 1:nslots
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
                local_ctxs[tid] = _mv_exact_scoring_context(
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
                local_loglik_buffers[tid] = Vector{Float64}(undef, size(trait_mat, 2))
                local_mbic_covered_workspaces[tid] = Vector{Bool}(undef, cache.ntips)
                local_mbic_edges_workspaces[tid] = Int[]
                local_profile_workspaces[tid] =
                    local_ctxs[tid].profile_workspace_caches === nothing ?
                    _mv_edge_elimination_workspaces(local_ctxs[tid], 0) :
                    nothing
            end
            Threads.@threads :dynamic for idx in eachindex(configs)
                tid = Threads.threadid()
                local_ctx = local_ctxs[tid]
                pruned = _mv_edge_elimination_pass_scored(
                    local_ctx,
                    score_cache,
                    score_cache_lock,
                    configs[idx].shift_edges;
                    initial_score = (success = true, score = configs[idx].score),
                    loglik_buffer = local_loglik_buffers[tid],
                    mbic_covered_workspace = local_mbic_covered_workspaces[tid],
                    mbic_edges_workspace = local_mbic_edges_workspaces[tid],
                    profile_workspaces = local_profile_workspaces[tid],
                    memo_stats = memo_stats,
                )
                counts = _record_edge_elimination_result!(configs[idx], pruned; criterion = criterion)
                configs[idx].source = _mv_edge_eliminated_source()
                local_trials[idx] = counts.n_trials
                local_removed[idx] = counts.n_removed
            end
            n_trials_total += sum(local_trials)
            n_removed_total += sum(local_removed)
        else
            warm_cache = Dict{Int, NamedTuple}()
            if initial_alpha !== nothing
                @inbounds for j in 1:min(length(initial_alpha), size(trait_mat, 2))
                    a = Float64(initial_alpha[j])
                    if isfinite(a) && a > 0.0
                        warm_cache[j] = (alpha = a, sigma2 = nothing, theta = Float64[])
                    end
                end
            end
            ctx = _mv_exact_scoring_context(
                tree,
                cache,
                trait_mat;
                criterion = criterion,
                optimization = optimization,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                missing_context = missing_context,
                refit_cache = Dict{Tuple, NamedTuple}(),
                warm_start_cache = warm_cache,
                root_model = root_model,
            )
            loglik_buffer = Vector{Float64}(undef, size(trait_mat, 2))
            mbic_covered_workspace = Vector{Bool}(undef, ctx.cache.ntips)
            mbic_edges_workspace = Int[]
            profile_workspaces =
                ctx.profile_workspace_caches === nothing ?
                _mv_edge_elimination_workspaces(ctx, 0) :
                nothing
            for cfg in configs
                pruned = _mv_edge_elimination_pass_scored(
                    ctx,
                    score_cache,
                    score_cache_lock,
                    cfg.shift_edges;
                    initial_score = (success = true, score = cfg.score),
                    loglik_buffer = loglik_buffer,
                    mbic_covered_workspace = mbic_covered_workspace,
                    mbic_edges_workspace = mbic_edges_workspace,
                    profile_workspaces = profile_workspaces,
                    memo_stats = memo_stats,
                )
                counts = _record_edge_elimination_result!(cfg, pruned; criterion = criterion)
                cfg.source = _mv_edge_eliminated_source()
                n_trials_total += counts.n_trials
                n_removed_total += counts.n_removed
            end
        end
    end
    _sort_scorable_configs!(configs)
    return (
        edge_elimination = edge_elimination_time,
        edge_elimination_trials = n_trials_total,
        edge_elimination_removed = n_removed_total,
        edge_elimination_memo_hits = memo_stats[].hits,
        edge_elimination_memo_misses = memo_stats[].misses,
        edge_elimination_memo_entries = length(score_cache),
        parallel_scoring = parallel_scoring && Threads.nthreads() > 1,
        nthreads = Threads.nthreads(),
    )
end

function _mv_run_l1ou_style_search(
    raw_configs::Vector{OUShiftConfiguration},
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
;
    max_shifts::Integer,
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
    raw_path_configs = deepcopy(raw_configs)
    raw_path_count = length(raw_path_configs)
    _mv_skip_consecutive_duplicate_configs!(raw_path_configs)
    raw_path_deduped_count = length(raw_path_configs)
    compressed_path_configs = deepcopy(raw_path_configs)
    _mv_compress_path_configs!(compressed_path_configs, max_shifts)
    compressed_path_count = length(compressed_path_configs)
    stability = _mv_shift_stability_table(raw_path_configs)
    stability_summary = (
        n_edges = length(stability.freq),
        n_round1_edges = length(stability.round1_freq),
        n_round2_edges = length(stability.round2_freq),
        n_persistent_edges = count(mask -> mask == 0x03, values(stability.round_mask)),
    )

    refit_cache_local = refit_cache
    warm_cache_local = warm_start_cache === nothing ? Dict{Int, NamedTuple}() : warm_start_cache
    profile_caches =
        profile_workspace_caches === nothing ?
        _mv_profile_workspace_caches(size(trait_mat, 2)) :
        profile_workspace_caches
    scoring_timings = _score_path_family_mv!(
        compressed_path_configs,
        tree,
        cache,
        trait_mat;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        initial_alpha = initial_alpha,
        missing_context = missing_context,
        refit_cache = refit_cache_local,
        warm_start_cache = warm_cache_local,
        profile_workspace_caches = profile_caches,
        root_model = root_model,
        parallel_scoring = parallel_scoring,
        fill_best = false,
    )

    edge_eliminated_configs = deepcopy(compressed_path_configs)
    edge_elimination_timings = _mv_edge_elimination_configs!(
        edge_eliminated_configs,
        tree,
        cache,
        trait_mat,
        stability;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        initial_alpha = initial_alpha,
        missing_context = missing_context,
        root_model = root_model,
        parallel_scoring = parallel_scoring,
    )

    profile = OUShiftConfiguration[]
    append!(profile, compressed_path_configs)
    append!(profile, edge_eliminated_configs)
    sort!(profile; by = cfg -> cfg.score)
    _deduplicate_shift_configs!(profile)
    _sort_scorable_configs!(profile)
    fill_time = @elapsed begin
        _fill_best_mv_config!(
            profile,
            tree,
            cache,
            trait_mat;
            criterion = criterion,
            optimization = optimization,
            max_iterations = max_iterations,
            rel_tol = rel_tol,
            missing_context = missing_context,
            refit_cache = refit_cache_local,
            warm_start_cache = warm_cache_local,
            profile_workspace_caches = profile_caches,
            root_model = root_model,
            fill_best = fill_best,
        )
    end

    return _MVPathSearchSummary(
        raw_path_configs = raw_path_configs,
        compressed_path_configs = compressed_path_configs,
        edge_eliminated_configs = edge_eliminated_configs,
        diagnostics = (
            raw_path_configs = raw_path_count,
            raw_path_configs_deduped = raw_path_deduped_count,
            compressed_path_configs = compressed_path_count,
            edge_eliminated_configs = length(edge_eliminated_configs),
            stability = stability_summary,
            exact_scoring = scoring_timings,
            edge_elimination = edge_elimination_timings,
            final_fill = fill_time,
            profile_configs = length(profile),
            profile = profile,
        ),
    )
end

