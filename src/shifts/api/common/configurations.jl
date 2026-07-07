function _deduplicate_configs!(configs::Vector{OUShiftConfiguration})
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


