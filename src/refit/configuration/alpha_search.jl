function _profile_refit_ou_shift_config(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    cache::OUMEdgeSegmentCache;
    max_iterations::Integer = 80,
    rel_tol::Float64 = 1e-6,
    start_alpha::Union{Nothing, Real} = nothing,
    cross_workspace::Union{Nothing, _ShiftCrossproductWorkspace} = nothing,
    alpha_floor::Real = 1e-4,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])
    default_alpha = max(log(2.0) / max(tree_height / 4.0, 1e-8), 1e-8)
    floor = max(Float64(alpha_floor), eps(Float64))
    center = start_alpha === nothing ? default_alpha : max(Float64(start_alpha), floor)
    lower = floor < 1e-4 ? log(floor) : max(log(floor), log(center) - log(100.0))
    upper = min(log(max(1e3 / max(tree_height, 1e-8), 100.0)), log(center) + log(100.0))
    lower < upper || (lower, upper = (log(1e-8), log(max(1e3 / max(tree_height, 1e-8), 100.0))))

    k = cache.nregimes + 1
    if cross_workspace === nothing || !_compatible_shift_workspace(cross_workspace, tree, k)
        cross_workspace = _shift_crossproduct_workspace(tree, k)
    end
    objective = log_alpha -> begin
        fit = _profile_oum_fixed_alpha(
            tree,
            trait,
            cache,
            exp(Float64(log_alpha)),
            cross_workspace;
            keep_theta = false,
            root_model = root_model,
        )
        fit.success ? -fit.loglik : Inf
    end
    result = Optim.optimize(
        objective,
        lower,
        upper,
        Optim.Brent();
        iterations = Int(max_iterations),
        rel_tol = rel_tol,
        abs_tol = rel_tol,
    )
    alpha_hat = exp(Optim.minimizer(result))
    fit = _profile_oum_fixed_alpha(tree, trait, cache, alpha_hat, cross_workspace; root_model = root_model)
    return (fit = fit, result = result)
end

