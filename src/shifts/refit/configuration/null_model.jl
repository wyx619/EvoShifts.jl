function _fit_ou1_for_shift_detection(
    tree::CompactTree,
    trait::AbstractVector{<:Real};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    root_model::Symbol = :OUfixedRoot,
)
    _normalize_ou_root_model(root_model)
    spec = ou_spec(:OU1)
    return _ou_fit_with_starts(tree, trait, spec;
        method = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
    )
end

