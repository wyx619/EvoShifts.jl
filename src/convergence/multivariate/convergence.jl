function _merge_convergent_regimes_mv(
    tree::CompactTree,
    cache::OUShiftTreeCache,
    trait_mat::AbstractMatrix{<:Real},
    shift_edges::Vector{Int},
    best_score::Float64;
    criterion::Symbol = :mBIC,
    max_iterations::Integer = 50,
    start_alpha::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    m_ntraits = size(trait_mat, 2)
    if length(shift_edges) < 2
        return (shift_edges = shift_edges, score = best_score, merge_map = Dict{Int,Int}())
    end

    alpha0 =
        start_alpha === nothing || length(start_alpha) != m_ntraits ?
        fill(0.0, m_ntraits) :
        Float64.(start_alpha)

    vertices = vcat(0, shift_edges)
    cr_ctx = _cr_design_context(cache, shift_edges)
    cr_workspaces = [_cr_trait_workspace(tree, cr_ctx) for _ in 1:m_ntraits]
    cr_group_buffers = [Vector{Int}(undef, cr_ctx.d) for _ in 1:m_ntraits]
    state_by_edge = _shift_state_by_edge(tree, shift_edges)
    start_components = [[v] for v in vertices]
    score_cache = Dict{Tuple{Int,Vararg{Int}}, Tuple{Float64, Dict{Int,Int}}}()

    function canonical_components(components::Vector{Vector{Int}})
        comps = [sort!(copy(c)) for c in components]
        sort!(comps; by = c -> (minimum(c), length(c)))
        return comps
    end

    function components_key(comps::Vector{Vector{Int}})
        out = Int[]
        for comp in comps
            append!(out, comp)
            push!(out, 0)
        end
        return Tuple(out)
    end

    function score_components(components::Vector{Vector{Int}})
        comps = canonical_components(components)
        key = components_key(comps)
        cached = get(score_cache, key, nothing)
        cached !== nothing && return cached
        mp = _merge_map_from_regime_components(shift_edges, comps, state_by_edge)
        sc = _score_cr_components_l1ou_optim_mv(
            tree,
            cache,
            trait_mat,
            shift_edges,
            comps,
            alpha0;
            criterion = criterion,
            max_iterations = max_iterations,
            ctx = cr_ctx,
            workspaces = cr_workspaces,
            group_buffers = cr_group_buffers,
        )
        val = (sc, mp)
        score_cache[key] = val
        return val
    end

    best = Inf
    best_map = Dict{Int,Int}()
    graph_edges = Tuple{Int,Int}[(v, v) for v in vertices]
    vertex_index = _convergent_vertex_index(vertices)
    uf_parent = Vector{Int}(undef, length(vertices))
    uf_rank = Vector{UInt8}(undef, length(vertices))
    uf_root_slot = Vector{Int}(undef, length(vertices))
    name_pos = _l1ou_convergent_order_position(cache, shift_edges)
    name_buf = Vector{Int}(undef, length(shift_edges))
    current_components = start_components
    current_ncomponents = length(current_components)
    prev_names = nothing

    for _ in 1:min(max_iterations, 2 * length(shift_edges))
        has_progress = false
        best_graph_edges = graph_edges
        best_components_this_round = current_components

        for u in vertices
            for v in vertices
                u == v && continue
                comps = canonical_components(_connected_components_indexed(
                    graph_edges,
                    vertices,
                    vertex_index,
                    uf_parent,
                    uf_rank;
                    root_slot = uf_root_slot,
                    extra = (u, v),
                ))
                length(comps) >= current_ncomponents && continue
                _l1ou_convergent_name_vector!(name_buf, cache, comps, name_pos)
                if prev_names !== nothing && name_buf == prev_names
                    continue
                end
                prev_names = copy(name_buf)
                sc, mp = score_components(comps)
                isfinite(sc) || continue
                if sc < best
                    best = sc
                    best_map = mp
                    best_graph_edges = copy(graph_edges)
                    push!(best_graph_edges, (u, v))
                    best_components_this_round = comps
                    has_progress = true
                end
            end
        end

        graph_edges = best_graph_edges
        current_components = best_components_this_round
        current_ncomponents = length(current_components)

        if has_progress
            for idx in eachindex(graph_edges)
                u, v = graph_edges[idx]
                u == v && continue
                comps = canonical_components(_connected_components_indexed(
                    graph_edges,
                    vertices,
                    vertex_index,
                    uf_parent,
                    uf_rank;
                    root_slot = uf_root_slot,
                    skip_index = idx,
                ))
                length(comps) <= current_ncomponents && continue
                sc, mp = score_components(comps)
                isfinite(sc) || continue
                if sc < best
                    best = sc
                    best_map = mp
                    best_graph_edges = Tuple{Int,Int}[graph_edges[j] for j in eachindex(graph_edges) if j != idx]
                    best_components_this_round = comps
                end
            end
            graph_edges = best_graph_edges
            current_components = best_components_this_round
            current_ncomponents = length(current_components)
        else
            break
        end
    end

    return (shift_edges = shift_edges, score = best, merge_map = best_map)
end

function _merge_convergent_regimes_multivariate(
    tree::CompactTree,
    trait::AbstractMatrix{<:Real},
    det::OUShiftDetectionResult;
    criterion::Symbol = :BIC,
    optimization::Symbol = :L_BFGS,
    max_iterations::Integer = 50,
    rel_tol::Float64 = 1e-8,
    regime_edge_groups::Union{Nothing, AbstractVector} = nothing,
)
    if !det.success || isempty(det.shift_edges)
        return det
    end
    n = tree.ntips
    cache = build_shift_tree_cache(tree)

    if regime_edge_groups !== nothing
        merge_map = _merge_map_from_edge_groups(tree, det.shift_edges, regime_edge_groups)
        m = size(trait, 2)
        alphas = Vector{Float64}(undef, m)
        sigmas = Vector{Float64}(undef, m)
        lls = Vector{Float64}(undef, m)
        thetas = Vector{Vector{Float64}}(undef, m)
        n_sh = 0
        for i in 1:m
            refit = _refit_ou_shift_config(
                tree,
                @view(trait[:, i]),
                det.shift_edges;
                optimization = optimization,
                max_iterations = max_iterations,
                rel_tol = rel_tol,
                merge_map = merge_map,
            )
            refit.success || return det
            alphas[i] = refit.alpha
            sigmas[i] = refit.sigma2
            lls[i] = refit.loglik
            thetas[i] = copy(refit.theta)
            n_sh = refit.n_shifts
        end
        score = _score_configuration_full_mv(
            cache, lls, n_sh, det.shift_edges, n;
            criterion = criterion,
            merge_map = merge_map,
        )
        edge_segments = _shift_edge_segments_with_merge(tree, det.shift_edges, merge_map)
        theta_mat = hcat(thetas...)
        fitted = Matrix{Float64}(undef, tree.ntips, m)
        shift_values = Matrix{Float64}(undef, length(det.shift_edges), m)
        shift_means = Matrix{Float64}(undef, length(det.shift_edges), m)
        edge_optima = Matrix{Float64}(undef, tree.nedges, m)
        for i in 1:m
            fitted[:, i] .= _ou_shift_fitted_means(tree, edge_segments, @view(theta_mat[:, i]), alphas[i])
            shift_values[:, i] .= _shift_values_from_theta(tree, edge_segments, det.shift_edges, @view(theta_mat[:, i]))
            shift_means[:, i] .= _shift_means_from_shift_values(tree, det.shift_edges, @view(shift_values[:, i]), alphas[i])
            edge_optima[:, i] .= _edge_optima_from_theta(edge_segments, @view(theta_mat[:, i]))
        end
        return OUShiftDetectionResult(
            success = true,
            model = :OUShiftsConvergent,
            ntraits = det.ntraits,
            shift_edges = copy(det.shift_edges),
            n_shifts = length(det.shift_edges),
            alpha = alphas,
            sigma2 = sigmas,
            loglik = lls,
            theta = theta_mat,
            shift_values = shift_values,
            shift_means = shift_means,
            fitted_means = fitted,
            residuals = Matrix{Float64}(trait) .- fitted,
            edge_optima = edge_optima,
            score = score,
            criterion = criterion,
            edge_regimes = _extract_edge_regimes(tree, edge_segments),
            edge_segments = edge_segments,
            profile = det.profile,
            diagnostics = merge(
                det.diagnostics,
                (
                    convergent_merge = true,
                    convergent_mode = :fixed_regime_edge_groups,
                    merged_score = score,
                    merge_map = merge_map,
                ),
            ),
        )
    end

    merge_result = _merge_convergent_regimes_mv(
        tree, cache, trait, det.shift_edges, det.score;
        criterion = criterion,
        max_iterations = max_iterations,
        start_alpha = det.alpha,
    )

    if isempty(merge_result.merge_map)
        return det
    end

    edge_segments = _shift_edge_segments_with_merge(tree, det.shift_edges, merge_result.merge_map)

    return OUShiftDetectionResult(
        success = true,
        model = :OUShiftsConvergent,
        ntraits = det.ntraits,
        shift_edges = copy(det.shift_edges),
        n_shifts = length(det.shift_edges),
        alpha = copy(det.alpha),
        sigma2 = copy(det.sigma2),
        loglik = copy(det.loglik),
        theta = copy(det.theta),
        shift_values = copy(det.shift_values),
        shift_means = copy(det.shift_means),
        fitted_means = copy(det.fitted_means),
        residuals = copy(det.residuals),
        edge_optima = copy(det.edge_optima),
        score = merge_result.score,
        criterion = criterion,
        edge_regimes = _extract_edge_regimes(tree, edge_segments),
        edge_segments = edge_segments,
        profile = det.profile,
        diagnostics = merge(
            det.diagnostics,
            (
                convergent_merge = true,
                merged_score = merge_result.score,
                convergent_search_score = merge_result.score,
                merge_map = merge_result.merge_map,
            ),
        ),
    )
end

