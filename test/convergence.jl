using LinearAlgebra
using Random
using Statistics
using Test
using EvoShifts

function _shift_test_tree(seed::Integer = 900)
    simtree = simulate_yule_simtree(100; tree_height = 1.0, rng = MersenneTwister(seed))
    return to_compact_tree(simtree)
end

function _shift_test_traits(tree::CompactTree, seed::Integer = 901)
    Sigma = [
        0.7 0.15;
        0.15 0.5;
    ]
    return simulate_mvbm1(tree, Sigma; rng = MersenneTwister(seed))
end

function _toy_shift_tree(name::AbstractString)
    tree_path = joinpath(mktempdir(), name)
    write(tree_path, "(((A:1,B:1):2,(C:1.5,D:1.5):1.5):1,(E:2,F:2):2);")
    return to_compact_tree(load_newick_tree(tree_path))
end

@testset "OUShift tree cache" begin
    tree = _shift_test_tree(900)
    cache = build_shift_tree_cache(tree)
    @test cache.ntips == 100
    @test cache.nedges == 198
    @test cache.tree_height > 0.0

    candidates = filter_candidate_edges(cache; min_descendant_tips = 2)
    @test length(candidates) > 0
    @test all(e -> cache.edge_length[e] >= eps(Float64), candidates)
end

@testset "OUShift edge signatures" begin
    toy_tree = _toy_shift_tree("toy_edge_signature_tree.tre")
    cache = build_shift_tree_cache(toy_tree)

    edge_ab = only([
        e for e in 1:cache.nedges
        if shift_edge_signature(toy_tree, e; cache = cache) == "A|B"
    ])
    @test shift_edge_signature(toy_tree, edge_ab; cache = cache) == "A|B"
    @test shift_edge_signatures(toy_tree, [edge_ab]; cache = cache) == ["A|B"]
    @test shift_edges_from_signatures(toy_tree, ["B|A"]; cache = cache) == [edge_ab]

    edge_cd = only([
        e for e in 1:cache.nedges
        if shift_edge_signature(toy_tree, e; cache = cache) == "C|D"
    ])
    @test shift_edges_from_signatures(toy_tree, ["D|C", "A|B"]; cache = cache) == [edge_cd, edge_ab]

    tab = shift_edge_table(toy_tree; edges = [edge_ab, edge_cd], cache = cache)
    @test names(tab) == ["edge_id", "tipX", "tipY"]
    @test tab.edge_id == [edge_ab, edge_cd]
    @test collect(zip(tab.tipX, tab.tipY)) == [("A", "B"), ("C", "D")]
    @test EvoShifts.shift_branch_anchor(toy_tree, edge_ab) == ("A", "B")

    tip_a_edge = only([
        e for e in 1:cache.nedges
        if shift_edge_signature(toy_tree, e; cache = cache) == "A"
    ])
    tip_tab = shift_edge_table(toy_tree; edges = [tip_a_edge], cache = cache)
    @test tip_tab.tipX == ["A"]
    @test tip_tab.tipY == ["A"]
end

@testset "OUShift large-tree candidate filters" begin
    tree = _shift_test_tree(910)
    cache = build_shift_tree_cache(tree)
    all_candidates = filter_candidate_edges(cache; exclude_root_edges = false)

    limited = filter_candidate_edges(
        cache;
        exclude_root_edges = false,
        min_descendant_tips = 2,
        max_candidate_edges = 5,
        candidate_sort = :descendant_length,
    )
    @test length(limited) <= 5
    @test all(e -> e in all_candidates, limited)
    @test all(e -> length(cache.descendant_tip_positions[e]) >= 2, limited)

    bounded = filter_candidate_edges(
        cache;
        exclude_root_edges = false,
        min_descendant_tips = 2,
        max_descendant_tips = 10,
        candidate_sort = :descendant_tips,
    )
    @test all(e -> 2 <= length(cache.descendant_tip_positions[e]) <= 10, bounded)
end

@testset "OUShift configuration correction" begin
    toy_tree = _toy_shift_tree("toy_config_tree.tre")
    cache = build_shift_tree_cache(toy_tree)
    corrected = EvoShifts.correct_shift_configuration(cache, [3, 3, 2, 2])
    @test corrected == [3]

    root_edges = [e for e in 1:toy_tree.nedges if toy_tree.parent_of_edge[e] == toy_tree.root]
    if !isempty(root_edges)
        @test isempty(EvoShifts.correct_shift_configuration(cache, root_edges))
        @test all(e -> toy_tree.parent_of_edge[e] != toy_tree.root, filter_candidate_edges(cache; candidate_edges = root_edges))
    end

    candidates = filter_candidate_edges(cache; min_descendant_tips = 1, exclude_root_edges = true)
    if !isempty(candidates)
        duplicated = [candidates[1], candidates[1]]
        corrected_dup = EvoShifts.correct_shift_configuration(cache, duplicated)
        @test corrected_dup == [candidates[1]]
    end
end

@testset "OUShift information criteria formulas" begin
    loglik = -10.0
    n = 20
    n_shifts = 2
    df_uni = EvoShifts._ou_shift_parameter_df(n_shifts)

    @test df_uni == 5
    @test isapprox(
        EvoShifts._compute_bic(loglik, df_uni, n),
        -2loglik + df_uni * log(n),
    )

    df_mv = EvoShifts._ou_shift_parameter_df(n_shifts; ntraits = 3)
    @test df_mv == 15
    @test isapprox(
        EvoShifts._compute_bic(loglik, df_mv, 3n),
        -2loglik + df_mv * log(3n),
    )
end

@testset "detect_ou_shifts on 100-tip tree" begin
    tree = _shift_test_tree(901)
    trait1 = _shift_test_traits(tree, 902)[:, 1]

    result = detect_ou_shifts(tree, trait1;
        criterion = :mBIC,
        max_shifts = 10,
    )

    @test result.success
    @test result.model == :OUShifts
    @test result.ntraits == 1
    @test result.n_shifts >= 0
    @test length(result.shift_edges) == result.n_shifts
    @test !isempty(result.edge_segments)
    @test length(result.edge_regimes) == tree.nedges
    @test result.diagnostics.criterion == :mBIC
    @test result.diagnostics.intercept_mode == :phylogenetic_intercept
    @test result.diagnostics.proposal_method == :tree_pruning_screening_with_prefix_z_anchor
    @test result.diagnostics.path_anchor_source == :tree_pruning_prefix_z
    @test result.diagnostics.prefix_anchor_min_standardized_score == 2.0
    @test haskey(result.diagnostics, :n_configs_round1)
    @test haskey(result.diagnostics, :n_configs_scored)
    @test length(result.theta) == result.n_shifts + 1
    @test length(result.shift_values) == result.n_shifts
    @test length(result.shift_means) == result.n_shifts
    @test length(result.fitted_means) == tree.ntips
    @test length(result.residuals) == tree.ntips
    @test length(result.edge_optima) == tree.nedges

    limited = detect_ou_shifts(tree, trait1;
        criterion = :mBIC,
        max_shifts = 10,
    )
    @test limited.success
    @test limited.diagnostics.max_profile_configs === nothing

    if result.n_shifts > 0
        edge_segs = shift_edges_to_edge_segments(tree, result.shift_edges)
        @test length(edge_segs) == tree.nedges
        for e in 1:tree.nedges
            @test length(edge_segs[e]) >= 1
            @test edge_segs[e][1].state >= 1
        end
    end
end

@testset "Multivariate OUShift smoke test" begin
    tree = _shift_test_tree(903)
    traits = _shift_test_traits(tree, 904)

    result = detect_ou_shifts(tree, traits;
        criterion = :mBIC,
        max_shifts = 5,
    )

    @test result.success
    @test result.ntraits == 2
    @test length(result.shift_edges) == result.n_shifts
    @test length(result.edge_segments) == tree.nedges
    @test size(result.theta, 1) == result.n_shifts + 1
    @test size(result.theta, 2) == 2
    @test size(result.shift_values) == (result.n_shifts, 2)
    @test size(result.shift_means) == (result.n_shifts, 2)
    @test size(result.fitted_means) == (tree.ntips, 2)
    @test size(result.residuals) == (tree.ntips, 2)
    @test size(result.edge_optima) == (tree.nedges, 2)
    @test result.diagnostics.intercept_mode == :phylogenetic_intercept
    @test result.diagnostics.proposal_method == :multivariate_path
    @test !result.diagnostics.path_anchor
    @test result.diagnostics.path_anchor_source == :none
end

@testset "Multivariate convergent merge preserves detection surface" begin
    tree = _toy_shift_tree("toy_mv_convergent_tree.tre")
    trait = [
        1.0 0.9;
        1.1 1.0;
        3.0 2.8;
        3.1 2.9;
        2.0 1.8;
        2.1 1.9;
    ]
    cache = build_shift_tree_cache(tree)
    shift_edges = shift_edges_from_signatures(tree, ["A", "B"]; cache = cache)
    fits = [
        fit_ou_shifts(tree, @view(trait[:, j]), shift_edges; criterion = :BIC, max_iterations = 80)
        for j in 1:2
    ]
    @test all(f -> f.success, fits)
    edge_segments = shift_edges_to_edge_segments(tree, shift_edges)
    det = OUShiftDetectionResult(
        success = true,
        model = :OUShifts,
        ntraits = 2,
        shift_edges = shift_edges,
        n_shifts = length(shift_edges),
        alpha = [f.alpha for f in fits],
        sigma2 = [f.sigma2 for f in fits],
        loglik = [f.loglik for f in fits],
        theta = hcat([f.theta for f in fits]...),
        shift_values = hcat([f.shift_values for f in fits]...),
        shift_means = hcat([f.shift_means for f in fits]...),
        fitted_means = hcat([f.fitted_means for f in fits]...),
        residuals = hcat([f.residuals for f in fits]...),
        edge_optima = hcat([f.edge_optima for f in fits]...),
        score = sum(f.score for f in fits),
        criterion = :BIC,
        edge_regimes = EvoShifts._extract_edge_regimes(tree, edge_segments),
        edge_segments = edge_segments,
        diagnostics = (source_tree = tree, source_trait = trait),
    )

    merged = merge_convergent_regimes(det; criterion = :BIC, max_iterations = 10)
    @test merged.success
    @test merged.model == :OUShiftsConvergent
    @test merged.n_shifts == det.n_shifts
    @test size(merged.theta, 2) == 2
    @test size(merged.fitted_means) == size(trait)
    @test merged.alpha == det.alpha
    @test merged.sigma2 == det.sigma2
    @test merged.loglik == det.loglik
    @test haskey(merged.diagnostics, :merge_map)
    @test haskey(merged.diagnostics, :convergent_search_score)
    @test merged.diagnostics.merge_map == Dict(3 => 2)
end

@testset "Convergent regime merge on detected shifts" begin
    tree = _shift_test_tree(905)
    trait1 = _shift_test_traits(tree, 906)[:, 1]

    det = detect_ou_shifts(tree, trait1;
        criterion = :mBIC,
        max_shifts = 6,
    )
    @test det.success

    if det.n_shifts >= 2
        merged = merge_convergent_regimes(det; criterion = :BIC)
        @test merged.success
        @test merged.model in (:OUShifts, :OUShiftsConvergent)
        @test merged.n_shifts == det.n_shifts
        @test !isempty(merged.edge_segments)
        @test length(merged.edge_regimes) == tree.nedges
    end
end

@testset "detect_ou_shifts then merge_convergent_regimes" begin
    tree = _shift_test_tree(907)
    trait1 = _shift_test_traits(tree, 908)[:, 1]

    det = detect_ou_shifts(tree, trait1;
        criterion = :mBIC,
        max_shifts = 6,
    )
    result = merge_convergent_regimes(det; criterion = :BIC, max_iterations = 10, rel_tol = 1e-7)

    @test result.success
    @test result.model in (:OUShifts, :OUShiftsConvergent)
    @test !isempty(result.edge_segments)
    @test length(result.edge_regimes) == tree.nedges
end
