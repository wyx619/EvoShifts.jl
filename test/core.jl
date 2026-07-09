function _core_shift_tree()
    path = joinpath(mktempdir(), "shift_core.tre")
    write(path, "(((A:1,B:1):2,(C:1.5,D:1.5):1.5):1,(E:2,F:2):2);")
    return to_compact_tree(load_newick_tree(path))
end

function _core_quadratic_from_pruning(tree, x, edge_a, edge_v)
    edge_b = zeros(Float64, tree.nedges)
    zero = zeros(Float64, tree.ntips)
    ll0 = EvoShifts._linear_gaussian_loglik(
        tree, zero, edge_a, edge_b, edge_v;
        root_prior_var = Inf,
        profile_root = true,
    )
    ll = EvoShifts._linear_gaussian_loglik(
        tree, x, edge_a, edge_b, edge_v;
        root_prior_var = Inf,
        profile_root = true,
    )
    @test ll0.success
    @test ll.success
    return -2.0 * ll.loglik + 2.0 * ll0.loglik
end

function _core_pruning_inner_product(tree, x, y, edge_a, edge_v)
    qx = _core_quadratic_from_pruning(tree, x, edge_a, edge_v)
    qy = _core_quadratic_from_pruning(tree, y, edge_a, edge_v)
    qxy = _core_quadratic_from_pruning(tree, x .+ y, edge_a, edge_v)
    return 0.5 * (qxy - qx - qy)
end

@testset "shift edge segments semantics" begin
    tree = _core_shift_tree()
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 2, exclude_root_edges = true)
    shift_edge = first(candidates)
    edge_segments = shift_edges_to_edge_segments(tree, [shift_edge])

    @test length(edge_segments) == tree.nedges
    @test all(e -> length(edge_segments[e]) == 1, 1:tree.nedges)
    @test edge_segments[shift_edge][1].state == 2
    @test isapprox(edge_segments[shift_edge][1].length, tree.edge_length[shift_edge])

    shifted_positions = Set(cache.descendant_tip_positions[shift_edge])
    for edge in 1:tree.nedges
        edge_state = edge_segments[edge][1].state
        desc = cache.descendant_tip_positions[edge]
        if !isempty(desc) && all(p -> p in shifted_positions, desc) && edge != shift_edge
            @test edge_state in (1, 2)
        end
        @test edge_state >= 1
    end

    root_edges = [e for e in 1:tree.nedges if tree.parent_of_edge[e] == tree.root]
    @test isempty(EvoShifts.correct_shift_configuration(cache, root_edges))
end

@testset "shift design weights" begin
    tree = _core_shift_tree()
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)
    height = maximum(tree.dist_from_root[tree.tip_ids])

    weights_bm = EvoShifts._precompute_design_weights(tree, candidates, 0.0)
    for (j, edge) in enumerate(candidates)
        parent = tree.parent_of_edge[edge]
        expected_age = height - tree.dist_from_root[parent]
        @test isapprox(weights_bm[j], expected_age; atol = 1e-12)
    end

    alpha = 0.7
    weights_ou = EvoShifts._precompute_design_weights(tree, candidates, alpha)
    for (j, edge) in enumerate(candidates)
        parent = tree.parent_of_edge[edge]
        age = height - tree.dist_from_root[parent]
        @test isapprox(weights_ou[j], 1.0 - exp(-alpha * age); atol = 1e-12)
    end
end

@testset "tree whitening equals pruning inner products" begin
    tree = _core_shift_tree()
    x = [1.0, -0.5, 0.2, 1.5, -1.0, 0.7]
    y = [0.3, 1.2, -0.4, 0.9, 0.1, -0.8]

    for (alpha, sigma2) in ((0.0, 1.0), (0.7, 1.3))
        edge_a, edge_v = EvoShifts._shift_screening_edges(tree, alpha, sigma2)
        wx = EvoShifts._tree_whiten_vector(tree, x, edge_a, edge_v)
        wy = EvoShifts._tree_whiten_vector(tree, y, edge_a, edge_v)
        pruning_cross = _core_pruning_inner_product(tree, x, y, edge_a, edge_v)
        @test isapprox(dot(wx, wy), pruning_cross; atol = 1e-8, rtol = 1e-8)
    end
end

@testset "shift screening ranks tree-operator columns" begin
    tree = _core_shift_tree()
    cache = build_shift_tree_cache(tree)
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]
    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = false)
    weights = EvoShifts._precompute_design_weights(tree, candidates, 0.0)
    ranked = EvoShifts._rank_shift_screening_edges(
        tree,
        cache,
        trait,
        candidates,
        0.0;
        weights = weights,
    )

    @test sort(ranked.edges) == sort(Int.(candidates))
    @test issorted(abs.(ranked.standardized_scores); rev = true)
    @test all(isfinite, ranked.scores)
    @test all(>(0.0), ranked.norms)
end

@testset "common shift pruning helper" begin
    scores = Dict(
        (1, 2, 3) => 10.0,
        (2, 3) => 8.0,
        (1, 3) => 11.0,
        (1, 2) => 11.0,
        (3,) => 9.0,
        (2,) => 9.0,
    )
    calls = Vector{Tuple{Vararg{Int}}}()
    accepted = Int[]
    score_fn = edges -> begin
        key = Tuple(edges)
        push!(calls, key)
        (success = true, score = get(scores, key, 100.0))
    end
    pruned = EvoShifts._prune_shift_edges_by_score(
        [1, 2, 3],
        score_fn;
        initial_score = (success = true, score = 10.0),
        on_accept = (edge, current, score) -> push!(accepted, edge),
    )

    @test pruned.shift_edges == [2, 3]
    @test pruned.removed_edges == [1]
    @test pruned.score.score == 8.0
    @test accepted == [1]
    @test !((1, 2, 3) in calls)

    worse_score_fn = edges -> (success = true, score = length(edges) == 3 ? 1.0 : 2.0)
    unchanged = EvoShifts._prune_shift_edges_by_score([4, 5, 6], worse_score_fn)
    @test unchanged.shift_edges == [4, 5, 6]
    @test isempty(unchanged.removed_edges)
    @test unchanged.score.score == 1.0
end


