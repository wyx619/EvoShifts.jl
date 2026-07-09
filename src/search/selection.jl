function _deduplicate_shift_configs!(configs::Vector{OUShiftConfiguration})
    seen = Set{Vector{Int}}()
    i = 1
    while i <= length(configs)
        if configs[i].shift_edges in seen
            deleteat!(configs, i)
        else
            push!(seen, copy(configs[i].shift_edges))
            i += 1
        end
    end
    return configs
end

function _sort_scorable_configs!(configs::Vector{OUShiftConfiguration})
    filter!(cfg -> isfinite(cfg.score) && cfg.score > -1e10, configs)
    sort!(configs; by = cfg -> cfg.score)
    return configs
end

function _apply_pruned_score_to_config!(
    cfg::OUShiftConfiguration,
    pruned;
    criterion::Symbol,
)
    cfg.shift_edges = pruned.shift_edges
    cfg.n_shifts = length(pruned.shift_edges)
    cfg.score = pruned.score.score
    cfg.criterion = criterion
    return cfg
end

function _fill_best_config!(
    configs::Vector{OUShiftConfiguration},
    fill_fn::F;
    fill_best::Bool = true,
) where {F}
    if fill_best && !isempty(configs)
        fill_fn(configs[1])
    end
    return configs
end

function _is_path_anchor_source(source::Symbol)
    return source === :screening_prefix_anchor_alpha0 ||
           source === :screening_prefix_anchor_alpha_refined ||
           source === :multivariate_screening_prefix_anchor_alpha0 ||
           source === :multivariate_screening_prefix_anchor_alpha_refined ||
           source === :multivariate_path_alpha0 ||
           source === :multivariate_path_alpha_refined
end

function _best_path_anchor_n_shifts(configs::Vector{OUShiftConfiguration})
    best_score = Inf
    best_n = nothing
    for cfg in configs
        _is_path_anchor_source(cfg.source) || continue
        cfg.n_shifts > 0 || continue
        isfinite(cfg.score) || continue
        if cfg.score < best_score
            best_score = cfg.score
            best_n = cfg.n_shifts
        end
    end
    return best_n
end

function _apply_path_anchor_complexity_ceiling!(configs::Vector{OUShiftConfiguration})
    anchor_n = _best_path_anchor_n_shifts(configs)
    anchor_n === nothing && return nothing
    filter!(cfg -> _is_path_anchor_source(cfg.source) || cfg.n_shifts <= anchor_n, configs)
    _sort_scorable_configs!(configs)
    return anchor_n
end

function _apply_path_anchor_selection!(configs::Vector{OUShiftConfiguration})
    anchor_n = _best_path_anchor_n_shifts(configs)
    anchor_n === nothing && return nothing
    filter!(cfg -> _is_path_anchor_source(cfg.source), configs)
    _sort_scorable_configs!(configs)
    return anchor_n
end
