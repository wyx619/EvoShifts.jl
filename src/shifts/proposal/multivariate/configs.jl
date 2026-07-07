function _pruning_group_path_push_config_from_beta!(
    configs::Vector{Vector{Int}},
    path_rows::Union{Nothing, Vector{NamedTuple}},
    all_shifts::Vector{Int},
    prev::Base.RefValue{Union{Nothing, Set{Int}}},
    stopped::Base.RefValue{Bool},
    beta::AbstractMatrix{<:Real},
    idx::Integer,
    candidates::AbstractVector{<:Integer},
    cache::OUShiftTreeCache;
    max_shifts::Integer,
    vote_threshold::Float64,
    edge_visible::Union{Nothing, AbstractMatrix{Bool}} = nothing,
)
    stopped[] && return nothing
    raw = _pruning_group_path_raw_config(
        beta,
        candidates;
        vote_threshold = vote_threshold,
        edge_visible = edge_visible,
    )
    corrected = correct_shift_configuration_l1ou(cache, raw)
    if length(corrected) > max_shifts
        stopped[] = true
        path_rows !== nothing && push!(path_rows, (
            step = Int(idx),
            raw = raw,
            corrected = corrected,
            frequency_sorted = Int[],
            action = :break_max_shifts,
        ))
        return nothing
    end
    current_set = Set(corrected)
    if prev[] !== nothing && current_set == prev[]
        path_rows !== nothing && push!(path_rows, (
            step = Int(idx),
            raw = raw,
            corrected = corrected,
            frequency_sorted = Int[],
            action = :skip_same_as_previous,
        ))
        return nothing
    end
    prev[] = current_set
    append!(all_shifts, corrected)
    sorted_cfg = _pruning_group_path_frequency_sorted_config(corrected, all_shifts)
    push!(configs, sorted_cfg)
    path_rows !== nothing && push!(path_rows, (
        step = Int(idx),
        raw = raw,
        corrected = corrected,
        frequency_sorted = sorted_cfg,
        action = :keep,
    ))
    return nothing
end

function _pruning_group_path_raw_config(
    beta::AbstractMatrix{<:Real},
    candidate_edges::AbstractVector{<:Integer};
    vote_threshold::Float64 = 0.5,
    edge_visible::Union{Nothing, AbstractMatrix{Bool}} = nothing,
)
    ntraits, ncandidates = size(beta)
    edges = Int[]
    @inbounds for j in 1:ncandidates
        n_active = 0
        n_visible = 0
        for i in 1:ntraits
            if edge_visible === nothing || edge_visible[j, i]
                n_visible += 1
                beta[i, j] != 0.0 && (n_active += 1)
            end
        end
        n_visible == 0 && continue
        keep = vote_threshold <= 0.0 ? n_active > 0 : n_active / n_visible >= vote_threshold
        keep && push!(edges, Int(candidate_edges[j]))
    end
    return edges
end

function _pruning_group_path_frequency_sorted_config(
    corrected::AbstractVector{<:Integer},
    all_shifts::AbstractVector{<:Integer},
)
    isempty(corrected) && return Int[]
    freqs = Vector{Int}(undef, length(corrected))
    @inbounds for (i, edge0) in enumerate(corrected)
        edge = Int(edge0)
        c = 0
        for old in all_shifts
            Int(old) == edge && (c += 1)
        end
        freqs[i] = c
    end
    order = sortperm(1:length(corrected); by = i -> string(freqs[i]), rev = true, alg = Base.Sort.MergeSort)
    return Int[Int(corrected[i]) for i in order]
end

function _fit_multivariate_path_proposal_missing(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real},
    missing_context::MVShiftMissingContext;
    n_lambda::Integer = 100,
    max_iterations::Integer = 1000,
    tol::Float64 = 1e-6,
    intercept_mode::Symbol = :phylogenetic_intercept,
    max_shifts::Integer = typemax(Int),
    vote_threshold::Float64 = 0.0,
    keep_path::Bool = false,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    p = length(candidates)
    m = size(trait_mat, 2)
    @assert length(alpha_vec) == m

    y_w_list = Vector{Vector{Float64}}(undef, m)
    xw_std = [Vector{Union{Nothing, Vector{Float64}}}(undef, m) for _ in 1:p]
    visible = trues(p, m)
    nrows_total = 0
    @inbounds for i in 1:m
        nrows_total += count(@view(missing_context.observed_masks[i][1:(tree.ntips - 1)]))
    end

    alpha_order, alpha_groups = _l1ou_alpha_groups(alpha_vec)
    @inbounds for alpha in alpha_order
        sqrt_inv_cov_t = Matrix{Float64}(_l1ou_sqrt_inv_covariance_transpose(tree, alpha; root_model = root_model))
        X = _l1ou_design_matrix(tree, cache, alpha, candidates)
        XX = sqrt_inv_cov_t * X
        for i in alpha_groups[alpha]
            obs_idx = missing_context.observed_indices[i]
            rows = missing_context.proposal_rows[i]
            y_w = Vector{Float64}(undef, length(rows))
            _l1ou_whiten_observed_response_rows!(y_w, sqrt_inv_cov_t, @view(trait_mat[:, i]), rows, obs_idx)
            y_w_list[i] = y_w
            @inbounds for j in 1:p
                col = Vector{Float64}(undef, length(rows))
                cn = 0.0
                for (rr, row0) in enumerate(rows)
                    v = XX[Int(row0), j]
                    col[rr] = v
                    cn += v * v
                end
                if cn <= eps(Float64)
                    xw_std[j][i] = nothing
                    visible[j, i] = false
                else
                    LinearAlgebra.rmul!(col, sqrt(nrows_total / cn))
                    xw_std[j][i] = col
                end
            end
        end
    end

    beta = zeros(Float64, m, p)
    group_sizes = zeros(Int, p)
    @inbounds for j in 1:p
        c = 0
        for i in 1:m
            xw_std[j][i] !== nothing && (c += 1)
        end
        group_sizes[j] = max(c, 1)
    end
    scores = zeros(Float64, m, p)
    for j in 1:p
        cols = xw_std[j]
        @inbounds for i in 1:m
            col = cols[i]
            scores[i, j] = col === nothing ? 0.0 : LinearAlgebra.dot(col, y_w_list[i])
        end
    end
    lambda_max = 0.0
    @inbounds for j in 1:p
        s = 0.0
        for i in 1:m
            v = -2.0 * scores[i, j]
            s += v * v
        end
        lambda_max = max(lambda_max, sqrt(s) / sqrt(Float64(group_sizes[j])))
    end
    lambda_max <= 0.0 && (lambda_max = 1e-8)
    fit_for_base_seq = function(base_seq, lmax)
        beta_tmp = zeros(Float64, m, p)
        df_vec = zeros(Int, length(base_seq))
        conv = _pruning_group_path_fit_cached!(
            beta_tmp, y_w_list, xw_std, _pruning_group_path_lambdas(lambda_max, base_seq; lmax = lmax);
            max_iterations = max_iterations,
            tol = 0.01,
            group_sizes = group_sizes,
            on_solution = (pos, b) -> (df_vec[pos] = _pruning_group_path_df_count(b)),
        )
        return df_vec, conv
    end
    base_seq, lmax = _pruning_group_path_adaptive_lambda_base_seq(max_shifts, lambda_max, fit_for_base_seq)
    lambdas = _pruning_group_path_lambdas(lambda_max, base_seq; lmax = lmax)
    configs = Vector{Vector{Int}}()
    path_rows = keep_path ? NamedTuple[] : nothing
    all_shifts = Int[]
    prev = Ref{Union{Nothing, Set{Int}}}(nothing)
    stopped = Ref(false)
    n_nonzero = zeros(Int, length(lambdas))
    converged = _pruning_group_path_fit_cached!(
        beta, y_w_list, xw_std, lambdas;
        max_iterations = max_iterations,
        tol = tol,
        group_sizes = group_sizes,
        on_solution = (pos, b) -> begin
            n_nonzero[pos] = _pruning_group_path_df_count(b)
            _pruning_group_path_push_config_from_beta!(
                configs, path_rows, all_shifts, prev, stopped, b, pos, candidates, cache;
                max_shifts = max_shifts,
                vote_threshold = vote_threshold,
                edge_visible = visible,
            )
        end,
    )

    return (
        configs = configs,
        path_rows = keep_path ? path_rows : NamedTuple[],
        lambdas = lambdas,
        nz = n_nonzero,
        converged = converged,
        proposal_method = :l1ou_sqrt_inv_cov_missing,
    )
end

function _fit_multivariate_path_proposal(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    n_lambda::Integer = 100,
    max_iterations::Integer = 1000,
    tol::Float64 = 1e-6,
    intercept_mode::Symbol = :phylogenetic_intercept,
    max_shifts::Integer = typemax(Int),
    vote_threshold::Float64 = 0.0,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    column_cache::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
    keep_path::Bool = false,
    root_model::Symbol = :OUfixedRoot,
)
    root_model = _normalize_ou_root_model(root_model)
    if missing_context !== nothing && missing_context.has_missing
        return _fit_multivariate_path_proposal_missing(
            tree,
            cache,
            trait_mat,
            candidates,
            alpha_vec,
            missing_context;
            n_lambda = n_lambda,
            max_iterations = max_iterations,
            tol = tol,
            intercept_mode = intercept_mode,
            max_shifts = max_shifts,
            vote_threshold = vote_threshold,
            keep_path = keep_path,
            root_model = root_model,
        )
    end

    p = length(candidates)
    m = size(trait_mat, 2)
    @assert length(alpha_vec) == m

    y_w_list = Vector{Vector{Float64}}(undef, m)
    weights_list = Vector{Vector{Float64}}(undef, m)
    edge_a_list = Vector{Vector{Float64}}(undef, m)
    edge_v_list = Vector{Vector{Float64}}(undef, m)
    whiten_ws_list = Vector{_TreeWhitenColumnWorkspace}(undef, m)

    for i in 1:m
        alpha_i = _l1ou_proposal_alpha(alpha_vec[i])
        tr_i = Float64.(trait_mat[:, i])
        edge_a, edge_v = _shift_screening_edges(tree, alpha_i, 1.0)
        edge_a_list[i] = edge_a
        edge_v_list[i] = edge_v
        whiten_ws_list[i] = _tree_whiten_column_workspace(tree, edge_a, edge_v)
        yw = _tree_whiten_vector(tree, tr_i, edge_a, edge_v)
        if intercept_mode !== :phylogenetic_intercept && intercept_mode !== :none
            throw(ArgumentError("Unsupported intercept_mode: $intercept_mode"))
        end
        y_w_list[i] = yw
        weights_list[i] = _precompute_design_weights(tree, candidates, alpha_i)
    end

    beta = zeros(Float64, m, p)
    lambdas = Float64[]
    ncontrasts = tree.ntips - 1
    col_scale = zeros(Float64, p, m)
    hessian = fill(0.01, p)
    scores = zeros(Float64, p, m)
    xw_tmp = [Vector{Float64}(undef, ncontrasts) for _ in 1:m]
    for i in 1:m
        xw = xw_tmp[i]
        @inbounds for j in 1:p
            _tree_whiten_shift_column!(
                xw,
                tree,
                cache,
                candidates[j],
                weights_list[i][j],
                edge_a_list[i],
                edge_v_list[i],
                whiten_ws_list[i],
            )
            cn = max(LinearAlgebra.dot(xw, xw), eps(Float64))
            scale = sqrt((ncontrasts * m) / cn)
            LinearAlgebra.rmul!(xw, scale)
            col_scale[j, i] = scale
            scores[j, i] = LinearAlgebra.dot(xw, y_w_list[i])
            hessian[j] = max(hessian[j], _pruning_group_path_hessian_col(xw))
        end
    end
    col_buffers = [Vector{Float64}(undef, ncontrasts) for _ in 1:m]
    cols_buf = Vector{Union{Nothing, Vector{Float64}}}(undef, m)
    function get_group_cols!(j::Int)
        @inbounds for i in 1:m
            col = col_buffers[i]
            _tree_whiten_shift_column!(
                col,
                tree,
                cache,
                candidates[j],
                weights_list[i][j],
                edge_a_list[i],
                edge_v_list[i],
                whiten_ws_list[i],
            )
            LinearAlgebra.rmul!(col, col_scale[j, i])
            cols_buf[i] = col
        end
        return cols_buf
    end
    lambda_max = 0.0
    @inbounds for j in 1:p
        s = 0.0
        for i in 1:m
            v = -2.0 * scores[j, i]
            s += v * v
        end
        lambda_max = max(lambda_max, sqrt(s) / sqrt(Float64(m)))
    end
    lambda_max <= 0.0 && (lambda_max = 1e-8)
    fit_for_base_seq = function(base_seq, lmax)
        beta_tmp = zeros(Float64, m, p)
        df_vec = zeros(Int, length(base_seq))
        conv = _pruning_group_path_fit_operator!(
            beta_tmp, y_w_list, p, _pruning_group_path_lambdas(lambda_max, base_seq; lmax = lmax);
            get_group_cols! = get_group_cols!,
            hessian = hessian,
            max_iterations = max_iterations,
            tol = 0.01,
            on_solution = (pos, b) -> (df_vec[pos] = _pruning_group_path_df_count(b)),
        )
        return df_vec, conv
    end
    base_seq, lmax = _pruning_group_path_adaptive_lambda_base_seq(max_shifts, lambda_max, fit_for_base_seq)
    lambdas = _pruning_group_path_lambdas(lambda_max, base_seq; lmax = lmax)
    configs = Vector{Vector{Int}}()
    path_rows = keep_path ? NamedTuple[] : nothing
    all_shifts = Int[]
    prev = Ref{Union{Nothing, Set{Int}}}(nothing)
    stopped = Ref(false)
    n_nonzero = zeros(Int, length(lambdas))
    converged = _pruning_group_path_fit_operator!(
        beta, y_w_list, p, lambdas;
        get_group_cols! = get_group_cols!,
        hessian = hessian,
        max_iterations = max_iterations,
        tol = tol,
        on_solution = (pos, b) -> begin
            n_nonzero[pos] = _pruning_group_path_df_count(b)
            _pruning_group_path_push_config_from_beta!(
                configs, path_rows, all_shifts, prev, stopped, b, pos, candidates, cache;
                max_shifts = max_shifts,
                vote_threshold = vote_threshold,
            )
        end,
    )

    return (
        configs = configs,
        path_rows = keep_path ? path_rows : NamedTuple[],
        lambdas = lambdas,
        nz = n_nonzero,
        converged = converged,
        proposal_method = :tree_pruning_multivariate_group_path_full,
    )
end

function _propose_shift_configs_multivariate_group_path(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    candidates::AbstractVector{<:Integer},
    alpha_vec::AbstractVector{<:Real};
    max_shifts::Integer = typemax(Int),
    n_lambda::Integer = 100,
    max_iterations::Integer = 1000,
    tol::Float64 = 1e-6,
    intercept_mode::Symbol = :phylogenetic_intercept,
    vote_threshold::Float64 = 0.0,
    missing_context::Union{Nothing, MVShiftMissingContext} = nothing,
    column_cache::Union{Nothing, Vector{Matrix{Float64}}} = nothing,
    keep_path::Bool = false,
    root_model::Symbol = :OUfixedRoot,
)
    path = _fit_multivariate_path_proposal(
        tree,
        cache,
        trait_mat,
        candidates,
        alpha_vec;
        n_lambda = n_lambda,
        max_iterations = max_iterations,
        tol = tol,
        intercept_mode = intercept_mode,
        max_shifts = max_shifts,
        vote_threshold = vote_threshold,
        missing_context = missing_context,
        column_cache = column_cache,
        keep_path = keep_path,
        root_model = root_model,
    )
    return (
        configs = path.configs,
        diagnostics = (
            proposal_method = path.proposal_method,
            n_lambda = n_lambda,
            n_lambda_used = length(path.converged),
            n_configs = length(path.configs),
            converged = path.converged,
            path_rows = keep_path ? path.path_rows : NamedTuple[],
            lambdas = keep_path ? path.lambdas : Float64[],
            nz = keep_path ? path.nz : Int[],
        ),
    )
end

