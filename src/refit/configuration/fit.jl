function _refit_ou_shift_config(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer};
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    merge_map::Union{Nothing, Dict{Int,Int}} = nothing,
    start_alpha::Union{Nothing, Real} = nothing,
    start_sigma2::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
    profile_workspace::Union{Nothing, _ShiftCrossproductWorkspace} = nothing,
    profile_alpha_floor::Real = 1e-4,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    tr = _validate_univariate_trait(tree, trait)
    has_merge = merge_map !== nothing && !isempty(merge_map)
    expected_k = length(shift_edges) + 2
    cache =
        if has_merge
            edge_segments = _shift_edge_segments_with_merge(tree, shift_edges, merge_map)
            _prepare_oum_edge_cache(tree, edge_segments)
        elseif profile_workspace !== nothing && _compatible_shift_workspace(profile_workspace, tree, expected_k)
            _shift_oum_cache_from_edges!(profile_workspace, tree, shift_edges)
        else
            _shift_oum_cache_from_edges(tree, shift_edges)
        end
    nregimes = cache.nregimes

    profiled = _profile_refit_ou_shift_config(
        tree,
        tr,
        cache;
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        start_alpha = start_alpha,
        cross_workspace = profile_workspace,
        alpha_floor = profile_alpha_floor,
        root_model = root_model,
    )
    if profiled.fit.success
        return (
            success = true,
            loglik = profiled.fit.loglik,
            alpha = profiled.fit.alpha,
            sigma2 = profiled.fit.sigma2,
            theta = copy(profiled.fit.theta),
            nregimes = nregimes,
            n_shifts = nregimes - 1,
            nparams = nregimes + 2,
        )
    end

    spec = ou_spec(:OUM)
    alpha0 = start_alpha === nothing ? max(log(2.0) / max(maximum(tree.dist_from_root[tree.tip_ids]) / 4.0, 1e-8), 1e-8) : Float64(start_alpha)
    sigma0 = start_sigma2 === nothing ? max(var(tr), 1e-8) : Float64(start_sigma2)
    theta0 =
        start_theta_regimes === nothing || length(start_theta_regimes) != nregimes ?
        fill(mean(tr), nregimes) :
        Float64.(start_theta_regimes)

    fit = _ou_fit_with_starts(
        tree,
        trait,
        spec;
        cache = cache,
        method = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        start_alpha = alpha0,
        start_sigma2 = sigma0,
        start_theta_regimes = theta0,
    )

    return (
        success = fit.profile.success,
        loglik = fit.profile.loglik,
        alpha = fit.bundle.alpha[1],
        sigma2 = fit.bundle.sigma2[1],
        theta = copy(fit.bundle.theta),
        nregimes = nregimes,
        n_shifts = nregimes - 1,
        nparams = nregimes + 2,
    )
end

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

function fit_ou_shifts(
    tree::CompactTree,
    trait::AbstractVector{<:Real},
    shift_edges::AbstractVector{<:Integer};
    criterion::Symbol = :mBIC,
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    max_descendant_tips::Union{Nothing, Integer} = nothing,
    max_candidate_edges::Union{Nothing, Integer} = nothing,
    candidate_sort::Symbol = :descendant_length,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    start_alpha::Union{Nothing, Real} = nothing,
    start_sigma2::Union{Nothing, Real} = nothing,
    start_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
    guess_alpha::Union{Nothing, Real} = nothing,
    guess_sigma2::Union{Nothing, Real} = nothing,
    guess_theta_regimes::Union{Nothing, AbstractVector{<:Real}} = nothing,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    start_alpha = start_alpha === nothing ? guess_alpha : start_alpha
    start_sigma2 = start_sigma2 === nothing ? guess_sigma2 : start_sigma2
    start_theta_regimes = start_theta_regimes === nothing ? guess_theta_regimes : start_theta_regimes
    cache = build_shift_tree_cache(tree)
    corrected = correct_shift_configuration(cache, shift_edges)
    edge_segments = shift_edges_to_edge_segments(tree, corrected)
    fit = _refit_ou_shift_config(
        tree,
        trait,
        corrected;
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        start_alpha = start_alpha,
        start_sigma2 = start_sigma2,
        start_theta_regimes = start_theta_regimes,
        root_model = root_model,
    )

    m_candidates, _ = _candidate_count_for_ic(
        tree;
        candidate_edges = candidate_edges,
        edge_length_threshold = edge_length_threshold,
        min_descendant_tips = min_descendant_tips,
        max_descendant_tips = max_descendant_tips,
        max_candidate_edges = max_candidate_edges,
        candidate_sort = candidate_sort,
    )
    score =
        if !fit.success
            Inf
        elseif criterion === :mBIC
            _compute_l1ou_mbic(fit.loglik, cache, corrected; ntraits = 1)
        elseif criterion === :AICc
            _compute_aicc(fit.loglik, _ou_shift_parameter_df(length(corrected)), tree.ntips)
        elseif criterion === :BIC
            _compute_bic(fit.loglik, _ou_shift_parameter_df(length(corrected)), tree.ntips)
        else
            throw(ArgumentError("Unsupported criterion: $criterion"))
        end
    theta = copy(fit.theta)
    fitted_means = fit.success ? _ou_shift_fitted_means(tree, edge_segments, theta, fit.alpha) : Float64[]
    shift_values = fit.success ? _shift_values_from_theta(tree, edge_segments, corrected, theta) : Float64[]
    shift_means = fit.success ? _shift_means_from_shift_values(tree, corrected, shift_values, fit.alpha) : Float64[]
    residuals = fit.success ? Float64.(trait) .- fitted_means : Float64[]
    edge_optima = fit.success ? _edge_optima_from_theta(edge_segments, theta) : Float64[]

    return OUShiftFitResult(
        success = fit.success,
        shift_edges = corrected,
        n_shifts = length(corrected),
        alpha = fit.success ? fit.alpha : NaN,
        sigma2 = fit.success ? fit.sigma2 : NaN,
        theta = theta,
        shift_values = shift_values,
        shift_means = shift_means,
        fitted_means = fitted_means,
        residuals = residuals,
        edge_optima = edge_optima,
        loglik = fit.loglik,
        score = score,
        criterion = criterion,
        nparams = fit.nparams,
        nregimes = fit.nregimes,
        edge_regimes = _extract_edge_regimes(tree, edge_segments),
        edge_segments = edge_segments,
        fit = fit,
        diagnostics = (
            corrected_shift_edges = corrected,
            root_model = root_model,
            m_candidates = m_candidates,
            source_tree = tree,
            candidate_filters = (
                edge_length_threshold = edge_length_threshold,
                min_descendant_tips = min_descendant_tips,
                max_descendant_tips = max_descendant_tips,
                max_candidate_edges = max_candidate_edges,
                candidate_sort = candidate_sort,
            ),
        ),
    )
end

function fit_ou_shifts(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    shift_edges::AbstractVector{<:Integer};
    criterion::Symbol = :mBIC,
    candidate_edges::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    edge_length_threshold::Float64 = eps(Float64),
    min_descendant_tips::Integer = 1,
    max_descendant_tips::Union{Nothing, Integer} = nothing,
    max_candidate_edges::Union{Nothing, Integer} = nothing,
    candidate_sort::Symbol = :descendant_length,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 200,
    rel_tol::Float64 = 1e-8,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    tr_mat = _validate_multivariate_trait_shift(tree, trait)
    cache = build_shift_tree_cache(tree)
    missing_context = _build_mv_shift_missing_context(tree, cache, tr_mat)
    corrected = correct_shift_configuration(cache, shift_edges)
    cfg = OUShiftConfiguration(shift_edges = corrected, n_shifts = length(corrected))
    refit_cache = Dict{Tuple, NamedTuple}()
    ok = _fill_mv_config_from_edges!(
        cfg,
        tree,
        cache,
        tr_mat,
        corrected;
        criterion = criterion,
        optimization = optimization,
        max_iterations = max_iterations,
        rel_tol = rel_tol,
        refit_cache = refit_cache,
        missing_context = missing_context,
        root_model = root_model,
    )
    edge_segments = shift_edges_to_edge_segments(tree, corrected)
    m_candidates, _ = _candidate_count_for_ic(
        tree;
        candidate_edges = candidate_edges,
        edge_length_threshold = edge_length_threshold,
        min_descendant_tips = min_descendant_tips,
        max_descendant_tips = max_descendant_tips,
        max_candidate_edges = max_candidate_edges,
        candidate_sort = candidate_sort,
    )
    m = size(tr_mat, 2)
    return OUShiftFitResult(
        success = ok,
        model = :OUShiftsFit,
        shift_edges = corrected,
        n_shifts = length(corrected),
        alpha = ok ? cfg.alpha : fill(NaN, m),
        sigma2 = ok ? cfg.sigma2 : fill(NaN, m),
        theta = ok ? cfg.theta : Matrix{Float64}(undef, 0, m),
        shift_values = ok ? cfg.shift_values : Matrix{Float64}(undef, 0, m),
        shift_means = ok ? cfg.shift_means : Matrix{Float64}(undef, 0, m),
        fitted_means = ok ? cfg.fitted_means : Matrix{Float64}(undef, 0, m),
        residuals = ok ? cfg.residuals : Matrix{Float64}(undef, 0, m),
        edge_optima = ok ? cfg.edge_optima : Matrix{Float64}(undef, 0, m),
        loglik = ok ? cfg.loglik : fill(NaN, m),
        score = ok ? cfg.score : Inf,
        criterion = criterion,
        nparams = _ou_shift_parameter_df(length(corrected); ntraits = m),
        nregimes = length(corrected) + 1,
        edge_regimes = _extract_edge_regimes(tree, edge_segments),
        edge_segments = edge_segments,
        fit = cfg,
        diagnostics = (
            corrected_shift_edges = corrected,
            root_model = root_model,
            m_candidates = m_candidates,
            ntraits = m,
            source_tree = tree,
            candidate_filters = (
                edge_length_threshold = edge_length_threshold,
                min_descendant_tips = min_descendant_tips,
                max_descendant_tips = max_descendant_tips,
                max_candidate_edges = max_candidate_edges,
                candidate_sort = candidate_sort,
            ),
        ),
    )
end
