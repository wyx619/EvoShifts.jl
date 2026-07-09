function profile_configurations(
    det::OUShiftDetectionResult;
    n_shifts::Union{Nothing, Integer} = nothing,
)
    configs = OUShiftConfiguration[]
    for cfg in det.profile
        n_shifts !== nothing && cfg.n_shifts != Int(n_shifts) && continue
        isfinite(cfg.score) || continue
        push!(configs, cfg)
    end
    sort!(configs; by = cfg -> cfg.score)
    return configs
end

function _limit_profile_configs_diverse!(configs::Vector{OUShiftConfiguration}, limit::Integer)
    limit >= 1 || throw(ArgumentError("max_profile_configs must be positive"))
    length(configs) <= limit && return configs

    selected = OUShiftConfiguration[]
    selected_ids = Set{UInt64}()
    seen_n = Set{Int}()
    for cfg in configs
        if !(cfg.n_shifts in seen_n)
            push!(selected, cfg)
            push!(selected_ids, objectid(cfg))
            push!(seen_n, cfg.n_shifts)
            length(selected) >= limit && break
        end
    end

    if length(selected) < limit
        for cfg in configs
            objectid(cfg) in selected_ids && continue
            push!(selected, cfg)
            length(selected) >= limit && break
        end
    end

    empty!(configs)
    append!(configs, selected)
    return configs
end

function get_shift_configuration(
    det::OUShiftDetectionResult,
    n_shifts::Integer;
    rank::Integer = 1,
)
    rank >= 1 || throw(ArgumentError("rank must be positive"))
    configs = profile_configurations(det; n_shifts = n_shifts)
    length(configs) >= rank || throw(ArgumentError("No configuration with n_shifts=$n_shifts and rank=$rank"))
    return copy(configs[Int(rank)].shift_edges)
end

function best_shift_configuration(det::OUShiftDetectionResult)
    det.success || throw(ArgumentError("shift detection result is not successful"))
    return copy(det.shift_edges)
end
