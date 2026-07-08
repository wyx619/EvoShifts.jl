function _ou_optimize_from_initial(
    objective,
    init::AbstractVector{<:Real},
    spec::OUSpec,
    nregimes::Integer,
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache;
    max_iterations::Integer,
    rel_tol::Float64,
    lower_bounds::AbstractVector{<:Real},
)
    if spec.model === :OU1
        return optimizeobjective(
            objective,
            init;
            method = :L_BFGS,
            max_iterations = max_iterations,
            rel_tol = rel_tol,
            lower_bounds = lower_bounds,
        )
    end
    candidates = _ou_multistart_candidates(spec, init, nregimes, tree, trait, cache)
    return multistartserial(
        objective,
        candidates;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        lower_bounds = lower_bounds,
    )
end
