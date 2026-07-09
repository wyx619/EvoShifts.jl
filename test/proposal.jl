using Test
using LinearAlgebra
using EvoShifts

function _proposal_shift_tree()
    path = joinpath(mktempdir(), "shift_proposal.tre")
    write(path, "(((A:1,B:1):2,(C:1.5,D:1.5):1.5):1,(E:2,F:2):2);")
    return to_compact_tree(load_newick_tree(path))
end

function _dense_l1ou_sqrt_inv_covariance_transpose_for_test(
    tree::CompactTree,
    alpha::Float64;
    root_model::Symbol = :OUfixedRoot,
)
    root_model = EvoShifts._normalize_ou_root_model(root_model)
    n = tree.ntips
    r_node_id = EvoShifts._l1ou_r_node_ids(tree)
    edge_order = EvoShifts._l1ou_postorder_edges(tree, r_node_id)
    tree_height = maximum(tree.dist_from_root[tree.tip_ids])
    F = zeros(Float64, n, 2n - 1)
    D = zeros(Float64, n, n)
    @inbounds for i in 1:n
        F[i, i] = 1.0
    end
    edge_len = zeros(Float64, tree.nedges)
    edge_by_child = Dict{Int, Int}()
    @inbounds for e in 1:tree.nedges
        edge_len[e] = EvoShifts._l1ou_ou_edge_length(tree, e, alpha, tree_height)
        edge_by_child[Int(r_node_id[Int(tree.child_of_edge[e])])] = e
    end

    active = collect(1:n)
    root_edge_len = 0.0
    counter = 1
    idx = 1
    while idx <= length(edge_order) - 1 && length(active) > 1
        e1 = edge_order[idx]
        e2 = edge_order[idx + 1]
        p1 = Int(r_node_id[Int(tree.parent_of_edge[e1])])
        p2 = Int(r_node_id[Int(tree.parent_of_edge[e2])])
        if p1 != p2
            idx += 1
            continue
        end
        i1 = Int(r_node_id[Int(tree.child_of_edge[e1])])
        i2 = Int(r_node_id[Int(tree.child_of_edge[e2])])
        t1 = edge_len[e1]
        t2 = edge_len[e2]
        u = t1 + t2
        @inbounds for r in 1:n
            D[r, counter] = (F[r, i1] - F[r, i2]) / sqrt(u)
            F[r, p1] = (F[r, i1] * t2 + F[r, i2] * t1) / u
        end
        e3 = get(edge_by_child, p1, 0)
        if e3 != 0
            edge_len[e3] += 1.0 / (1.0 / t1 + 1.0 / t2)
        else
            root_edge_len += 1.0 / (1.0 / t1 + 1.0 / t2)
        end
        filter!(x -> x != i1 && x != i2, active)
        push!(active, p1)
        counter += 1
        idx += 2
    end
    root_len = root_edge_len + EvoShifts._l1ou_root_edge_length(tree, alpha, root_model, tree_height)
    @inbounds for r in 1:n
        D[r, counter] = F[r, active[1]] / sqrt(root_len)
    end
    return transpose(D)
end

function _dense_l1ou_design_matrix_for_test(
    tree::CompactTree,
    cache::EvoShifts.OUShiftTreeCache,
    alpha::Float64,
    candidates::AbstractVector{<:Integer},
)
    X = zeros(Float64, tree.ntips, length(candidates))
    weights = EvoShifts._precompute_design_weights(tree, candidates, alpha)
    @inbounds for (j, edge0) in enumerate(candidates)
        edge = Int(edge0)
        for tip_pos in cache.descendant_tip_positions[edge]
            X[tip_pos, j] = weights[j]
        end
    end
    return X
end

function _dense_l1ou_whiten_observed_response_rows_for_test!(
    out::AbstractVector{Float64},
    sqrt_inv_cov_t::AbstractMatrix{Float64},
    trait::AbstractVector{<:Real},
    rows::AbstractVector{<:Integer},
    obs_idx::AbstractVector{<:Integer},
)
    @inbounds for (rr, row0) in enumerate(rows)
        row = Int(row0)
        s = 0.0
        for col0 in obs_idx
            col = Int(col0)
            s += sqrt_inv_cov_t[row, col] * Float64(trait[col])
        end
        out[rr] = s
    end
    return out
end

@testset "univariate screening proposal" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)

    proposal = EvoShifts._propose_shift_configs_screening(
        tree,
        cache,
        trait,
        candidates,
        0.0;
        max_shifts = 4,
        n_lambda = 20,
        lambda_min_ratio = 0.01,
    )

    @test proposal.diagnostics.proposal_method == :tree_pruning_screening
    @test proposal.diagnostics.proposal_family == :tree_pruning_screening
    @test proposal.diagnostics.n_configs == length(proposal.configs)
    @test !isempty(proposal.configs)
    @test all(cfg -> length(cfg) <= 4, proposal.configs)
    @test all(cfg -> all(e -> e in candidates, cfg), proposal.configs)
end

@testset "univariate screening candidate builder" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)
    root_edges = Int[e for e in candidates if Int(cache.edge_parent[e]) == Int(cache.root)]
    nonroot_edges = Int[e for e in candidates if Int(cache.edge_parent[e]) != Int(cache.root)]
    @test !isempty(nonroot_edges)

    ranked = isempty(root_edges) ? nonroot_edges : vcat(first(root_edges), nonroot_edges)
    configs = EvoShifts._build_univariate_screening_candidates(
        cache,
        ranked;
        max_shifts = 3,
    )

    @test !isempty(configs)
    @test all(cfg -> all(e -> !(e in root_edges), cfg), configs)
    @test all(cfg -> length(cfg) <= 3, configs)
    @test length(unique(Tuple.(configs))) == length(configs)
end

@testset "screening candidate builder keeps beam alternatives" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)
    ranked = Int[e for e in candidates if Int(cache.edge_parent[e]) != Int(cache.root)]
    @test length(ranked) >= 6

    configs = EvoShifts._build_univariate_screening_candidates(
        cache,
        ranked;
        max_shifts = 3,
    )
    target = EvoShifts.correct_shift_configuration_l1ou(cache, [ranked[1], ranked[2], ranked[6]])
    target_set = Set(target)

    @test !isempty(target)
    @test !([ranked[1], ranked[2], ranked[6]] in [[ranked[1]], [ranked[1], ranked[2]], [ranked[1], ranked[2], ranked[3]]])
    @test any(cfg -> Set(cfg) == target_set, configs)
end

@testset "screening prefix continues after correction removes nested edges" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    ranked = [3, 5, 6, 4, 7, 8, 9, 10]

    configs = EvoShifts._build_univariate_screening_candidates(
        cache,
        ranked;
        max_shifts = 4,
    )

    @test maximum(length.(configs)) == 4
    @test any(cfg -> Set(cfg) == Set([5, 6, 4, 7]), configs)
end

@testset "screening prefix anchor records corrected prefixes" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    ranked = [3, 5, 6, 4, 7, 8, 9, 10]

    configs = EvoShifts._build_univariate_screening_prefix_anchor_candidates(
        cache,
        ranked;
        max_shifts = 4,
    )

    @test !isempty(configs)
    @test maximum(length.(configs)) == 4
    @test any(cfg -> Set(cfg) == Set([5, 6, 4, 7]), configs)
    @test length(unique(Tuple.(sort.(copy.(configs))))) == length(configs)
end

@testset "screening prefix anchor can stop by tree-pruning z threshold" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)

    full = EvoShifts._propose_shift_configs_screening_prefix_anchor(
        tree,
        cache,
        trait,
        candidates,
        0.0;
        max_shifts = 4,
    )
    limited = EvoShifts._propose_shift_configs_screening_prefix_anchor(
        tree,
        cache,
        trait,
        candidates,
        0.0;
        max_shifts = 4,
        min_standardized_score = 2.0,
    )

    @test limited.diagnostics.prefix_edges_used <= full.diagnostics.prefix_edges_used
    @test length(limited.configs) <= length(full.configs)
    @test limited.diagnostics.min_standardized_score == 2.0
end

@testset "multivariate group proposal API" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    traits = [
        1.0 0.8;
        1.2 1.0;
        1.8 1.5;
        2.0 1.7;
        2.5 2.2;
        2.3 2.0;
    ]
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)

    proposal = EvoShifts._propose_shift_configs_multivariate_group_path(
        tree,
        cache,
        traits,
        candidates,
        [0.0, 0.0];
        max_shifts = 4,
        n_lambda = 12,
    )

    @test proposal.diagnostics.proposal_method == :tree_pruning_multivariate_group_path_full
    @test proposal.diagnostics.n_lambda == 12
    @test proposal.diagnostics.n_configs == length(proposal.configs)
    @test all(cfg -> length(cfg) <= 4, proposal.configs)
    @test all(cfg -> all(e -> e in candidates, cfg), proposal.configs)
end

@testset "l1ou row-filtered tree-pruning operator matches dense rows" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    trait = [1.0, NaN, 1.8, 2.0, NaN, 2.3]
    obs_idx = [1, 3, 4, 6]
    rows = [1, 2, 4, tree.ntips]
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)

    for (alpha, root_model) in ((0.0, :OUfixedRoot), (0.35, :OUrandomRoot))
        dense = Matrix{Float64}(_dense_l1ou_sqrt_inv_covariance_transpose_for_test(tree, alpha; root_model = root_model))
        plan = EvoShifts._l1ou_row_whitening_plan(tree, alpha; root_model = root_model)
        workspace = EvoShifts._l1ou_row_whitening_workspace(tree, plan)

        y_dense = Vector{Float64}(undef, length(rows))
        _dense_l1ou_whiten_observed_response_rows_for_test!(y_dense, dense, trait, rows, obs_idx)
        y_pruning = similar(y_dense)
        EvoShifts._l1ou_whiten_observed_response_rows_pruning!(
            y_pruning,
            plan,
            tree,
            trait,
            rows,
            obs_idx,
            workspace,
        )
        @test y_pruning ≈ y_dense atol = 1e-10 rtol = 1e-10

        X = _dense_l1ou_design_matrix_for_test(tree, cache, alpha, candidates)
        XX = dense * X
        weights = EvoShifts._precompute_design_weights(tree, candidates, alpha)
        for (j, edge) in enumerate(candidates)
            col_pruning = Vector{Float64}(undef, length(rows))
            EvoShifts._l1ou_whiten_shift_column_rows_pruning!(
                col_pruning,
                plan,
                tree,
                cache,
                edge,
                weights[j],
                rows,
                workspace,
            )
            @test col_pruning ≈ XX[rows, j] atol = 1e-10 rtol = 1e-10
        end
    end
end

@testset "multivariate screening proposal API" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    traits = [
        1.0 0.8;
        1.2 1.0;
        1.8 1.5;
        2.0 1.7;
        2.5 2.2;
        2.3 2.0;
    ]
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)

    proposal = EvoShifts._propose_shift_configs_multivariate_screening(
        tree,
        cache,
        traits,
        candidates,
        [0.0, 0.0];
        max_shifts = 4,
        n_lambda = 12,
    )

    @test proposal.diagnostics.proposal_method == :tree_pruning_multivariate_screening
    @test proposal.diagnostics.n_configs == length(proposal.configs)
    @test !isempty(proposal.configs)
    @test all(cfg -> length(cfg) <= 4, proposal.configs)
    @test all(cfg -> all(e -> e in candidates, cfg), proposal.configs)
end

@testset "multivariate screening prefix anchor proposal API" begin
    tree = _proposal_shift_tree()
    cache = build_shift_tree_cache(tree)
    traits = [
        1.0 0.8;
        1.2 1.0;
        1.8 1.5;
        2.0 1.7;
        2.5 2.2;
        2.3 2.0;
    ]
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)

    full = EvoShifts._propose_shift_configs_multivariate_screening_prefix_anchor(
        tree,
        cache,
        traits,
        candidates,
        [0.0, 0.0];
        max_shifts = 4,
    )
    limited = EvoShifts._propose_shift_configs_multivariate_screening_prefix_anchor(
        tree,
        cache,
        traits,
        candidates,
        [0.0, 0.0];
        max_shifts = 4,
        min_shared_score = 2.0,
    )

    @test full.diagnostics.proposal_method == :tree_pruning_multivariate_screening_prefix_anchor
    @test full.diagnostics.proposal_family == :tree_pruning_multivariate_screening
    @test full.diagnostics.n_configs == length(full.configs)
    @test !isempty(full.configs)
    @test all(cfg -> length(cfg) <= 4, full.configs)
    @test all(cfg -> all(e -> e in candidates, cfg), full.configs)
    @test limited.diagnostics.prefix_edges_used <= full.diagnostics.prefix_edges_used
    @test length(limited.configs) <= length(full.configs)
    @test limited.diagnostics.min_shared_score == 2.0
end
