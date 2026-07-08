function _prune_shift_edges_by_score(
    shift_edges::AbstractVector{<:Integer},
    score_fn::F;
    max_edge_elimination_passes::Integer = 1,
    min_prunable_shifts::Integer = 3,
    initial_score = nothing,
    on_accept::G = (edge, current, score) -> nothing,
) where {F, G}
    current = Int.(shift_edges)
    best = initial_score === nothing ? score_fn(current) : initial_score
    removed = Int[]
    best.success || return (shift_edges = current, removed_edges = removed, score = best, n_trials = 0, n_passes = 0)
    length(current) < Int(min_prunable_shifts) && return (shift_edges = current, removed_edges = removed, score = best, n_trials = 0, n_passes = 0)

    n_passes = max(Int(max_edge_elimination_passes), 1)
    trial_edges = Int[]
    accepted_edges = Int[]
    n_trials = 0
    n_passes_done = 0
    for _ in 1:n_passes
        n_passes_done += 1
        changed = false
        for edge in copy(current)
            edge in current || continue
            _without_edge!(trial_edges, current, edge)
            n_trials += 1
            trial = score_fn(trial_edges)
            if trial.success && trial.score < best.score
                empty!(accepted_edges)
                append!(accepted_edges, trial_edges)
                current, accepted_edges = accepted_edges, current
                best = trial
                push!(removed, edge)
                on_accept(edge, current, best)
                changed = true
            end
        end
        changed || break
    end
    return (shift_edges = current, removed_edges = removed, score = best, n_trials = n_trials, n_passes = n_passes_done)
end

function _record_edge_elimination_result!(
    cfg::OUShiftConfiguration,
    pruned;
    criterion::Symbol,
)
    _apply_pruned_score_to_config!(cfg, pruned; criterion = criterion)
    return (n_trials = pruned.n_trials, n_removed = length(pruned.removed_edges))
end

