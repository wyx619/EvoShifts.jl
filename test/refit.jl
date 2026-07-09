using Test
using DataFrames
using EvoShifts

function _fit_ic_tree()
    path = joinpath(mktempdir(), "shift_fit_ic.tre")
    write(path, "(((A:1,B:1):2,(C:1.5,D:1.5):1.5):1,(E:2,F:2):2);")
    return to_compact_tree(load_newick_tree(path))
end

@testset "shift detection result surface" begin
    tree = _fit_ic_tree()
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]

    empty = detect_ou_shifts(
        tree,
        trait;
        candidate_edges = Int[],
        criterion = :mBIC,
    )
    @test empty.success
    @test empty.criterion == :mBIC
    @test empty.diagnostics.criterion == :mBIC
    @test empty.n_shifts == 0

    fit = detect_ou_shifts(
        tree,
        trait;
        max_shifts = 1,
        criterion = :mBIC,
    )
    @test fit.success
    @test fit.ntraits == 1
    @test length(fit.loglik) == 1
    @test isfinite(fit.loglik[1])
    @test length(fit.alpha) == 1
    @test length(fit.sigma2) == 1

    summary = shift_detection_summary(fit)
    @test summary.success
    @test summary.ntraits == 1
    @test summary.n_shifts == fit.n_shifts
    @test length(summary.shift_branches) == fit.n_shifts
    @test length(summary.R_edge_ids_postorder) == fit.n_shifts
    @test length(summary.descendant_signatures) == fit.n_shifts
    @test isfinite(summary.loglik)

    table = shift_detection_summary_table(fit)
    @test nrow(table) == 1
    @test table.n_shifts[1] == fit.n_shifts
    @test "R_edge_ids_postorder" in names(table)
    @test "shift_branches" in names(table)
    @test "descendant_signatures" in names(table)
    @test !("shift_edges" in names(table))
end

@testset "shift trait alignment helper" begin
    tree = _fit_ic_tree()
    labels = collect(tree.tip_labels)
    values = collect(1.0:length(labels))
    shuffled = reverse(eachindex(labels))
    df = DataFrame(
        taxon = labels[shuffled],
        trait1 = values[shuffled],
        trait2 = 10.0 .* values[shuffled],
        extra = fill("ignored", length(labels)),
    )

    y = align_traits_to_tree(tree, df; taxon_col = :taxon, trait_cols = :trait1)
    @test y == values

    X = align_traits_to_tree(tree, df; taxon_col = :taxon, trait_cols = [:trait1, :trait2])
    @test X == hcat(values, 10.0 .* values)

    ordered = align_traits_to_tree(tree, DataFrame(trait = values); trait_cols = :trait)
    @test ordered == values

    @test_throws ArgumentError align_traits_to_tree(tree, df; taxon_col = :taxon, trait_cols = :missing_col)
    @test_throws ArgumentError align_traits_to_tree(tree, DataFrame(taxon = labels[1:end-1], trait = values[1:end-1]); taxon_col = :taxon, trait_cols = :trait)
end

@testset "shift criteria formulas" begin
    loglik = -10.0
    n = 20
    n_shifts = 2
    df_uni = EvoShifts._ou_shift_parameter_df(n_shifts)

    @test df_uni == 5
    @test isapprox(
        EvoShifts._compute_bic(loglik, df_uni, n),
        -2loglik + df_uni * log(n),
    )
    tree = _fit_ic_tree()
    cache = build_shift_tree_cache(tree)
    @test isapprox(EvoShifts._compute_l1ou_mbic(loglik, cache, Int[]; ntraits = 1), -2loglik + 3log(tree.ntips))
end

@testset "shift refit delegates to OU family" begin
    tree = _fit_ic_tree()
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 2, exclude_root_edges = true)
    shift_edge = first(candidates)

    zero = EvoShifts._score_refit_univariate(
        tree,
        cache,
        trait,
        Int[],
        tree.ntips;
        criterion = :mBIC,
        max_iterations = 80,
    )
    one = EvoShifts._score_refit_univariate(
        tree,
        cache,
        trait,
        [shift_edge],
        tree.ntips;
        criterion = :mBIC,
        max_iterations = 80,
    )

    @test zero.success
    @test one.success
    @test zero.n_shifts == 0
    @test one.n_shifts == 1
    @test isfinite(zero.score)
    @test isfinite(one.score)
end

@testset "public shift fit and configuration IC" begin
    tree = _fit_ic_tree()
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 2, exclude_root_edges = true)
    shift_edge = first(candidates)

    fit = fit_ou_shifts(
        tree,
        trait,
        [shift_edge];
        criterion = :mBIC,
        candidate_edges = candidates,
        max_iterations = 80,
    )
    ic = configuration_ic(
        tree,
        trait,
        [shift_edge];
        criterion = :mBIC,
        candidate_edges = candidates,
        max_iterations = 80,
    )

    @test fit.success
    @test fit.n_shifts == 1
    @test fit.nregimes == 2
    @test length(fit.theta) == fit.nregimes
    @test length(fit.shift_values) == fit.n_shifts
    @test length(fit.shift_means) == fit.n_shifts
    @test length(fit.fitted_means) == tree.ntips
    @test length(fit.residuals) == tree.ntips
    @test length(fit.edge_optima) == tree.nedges
    @test length(fit.edge_segments) == tree.nedges
    @test isfinite(fit.loglik)
    @test isfinite(fit.score)
    @test isapprox(ic, fit.score)
end

@testset "OU shift root_model option" begin
    tree = _fit_ic_tree()
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]
    X = hcat(trait, trait .+ [0.0, 0.1, -0.1, 0.0, 0.2, -0.2])
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 2, exclude_root_edges = true)
    shift_edge = first(candidates)

    fixed = fit_ou_shifts(tree, trait, [shift_edge]; criterion = :mBIC, max_iterations = 80)
    explicit_fixed = fit_ou_shifts(tree, trait, [shift_edge];
        criterion = :mBIC, max_iterations = 80, root_model = :OUfixedRoot)
    random = fit_ou_shifts(tree, trait, [shift_edge];
        criterion = :mBIC, max_iterations = 80, root_model = :OUrandomRoot)
    random_mv = fit_ou_shifts(tree, X, [shift_edge];
        criterion = :mBIC, max_iterations = 80, root_model = :OUrandomRoot)

    @test fixed.success
    @test explicit_fixed.success
    @test random.success
    @test random_mv.success
    @test isapprox(fixed.loglik, explicit_fixed.loglik)
    @test random.diagnostics.root_model == :OUrandomRoot
    @test random_mv.diagnostics.root_model == :OUrandomRoot
    @test all(isfinite, random_mv.loglik)
    @test_throws ArgumentError fit_ou_shifts(tree, trait, [shift_edge]; root_model = :badRoot)
end

@testset "public multivariate shift fit and configuration IC" begin
    tree = _fit_ic_tree()
    trait = [
        1.0 0.8;
        1.2 1.0;
        1.8 1.5;
        2.0 1.7;
        2.5 2.2;
        2.3 2.0;
    ]
    cache = build_shift_tree_cache(tree)
    candidates = filter_candidate_edges(cache; min_descendant_tips = 2, exclude_root_edges = true)
    shift_edge = first(candidates)

    fit = fit_ou_shifts(
        tree,
        trait,
        [shift_edge];
        criterion = :BIC,
        candidate_edges = candidates,
        max_iterations = 80,
    )
    ic = configuration_ic(
        tree,
        trait,
        [shift_edge];
        criterion = :BIC,
        candidate_edges = candidates,
        max_iterations = 80,
    )

    @test fit.success
    @test fit.n_shifts == 1
    @test fit.nregimes == 2
    @test length(fit.alpha) == 2
    @test length(fit.sigma2) == 2
    @test length(fit.loglik) == 2
    @test size(fit.theta) == (fit.nregimes, 2)
    @test size(fit.shift_values) == (fit.n_shifts, 2)
    @test size(fit.shift_means) == (fit.n_shifts, 2)
    @test size(fit.fitted_means) == size(trait)
    @test size(fit.residuals) == size(trait)
    @test size(fit.edge_optima) == (tree.nedges, 2)
    @test length(fit.edge_segments) == tree.nedges
    @test all(isfinite, fit.loglik)
    @test isfinite(fit.score)
    @test isapprox(ic, fit.score)
end

@testset "convergent regime merge preserves detection surface" begin
    tree = _fit_ic_tree()
    trait = [1.0, 1.1, 3.0, 3.1, 2.0, 2.1]
    cache = build_shift_tree_cache(tree)
    shift_edges = shift_edges_from_signatures(tree, ["A", "B"]; cache = cache)
    fit = fit_ou_shifts(tree, trait, shift_edges; criterion = :BIC, max_iterations = 80)
    det = OUShiftDetectionResult(
        success = true,
        model = :OUShifts,
        ntraits = 1,
        shift_edges = shift_edges,
        n_shifts = length(shift_edges),
        alpha = [fit.alpha],
        sigma2 = [fit.sigma2],
        loglik = [fit.loglik],
        theta = fit.theta,
        shift_values = fit.shift_values,
        shift_means = fit.shift_means,
        fitted_means = fit.fitted_means,
        residuals = fit.residuals,
        edge_optima = fit.edge_optima,
        score = fit.score,
        criterion = :BIC,
        edge_regimes = fit.edge_regimes,
        edge_segments = fit.edge_segments,
        diagnostics = (source_tree = tree, source_trait = trait),
    )

    conv = merge_convergent_regimes(det; criterion = :BIC, max_iterations = 10)
    @test conv.model == :OUShiftsConvergent
    @test isfinite(conv.score)
    @test length(conv.loglik) == 1
    @test isfinite(conv.loglik[1])
    @test conv.alpha == det.alpha
    @test conv.sigma2 == det.sigma2
    @test conv.loglik == det.loglik
    @test haskey(conv.diagnostics, :merge_map)
    @test haskey(conv.diagnostics, :convergent_search_score)
    @test conv.diagnostics.merge_map == Dict(3 => 2)
    @test conv.n_shifts == det.n_shifts
    @test length(unique(conv.edge_regimes)) == 2
end

@testset "profile configuration access" begin
    tree = _fit_ic_tree()
    trait = [1.0, 1.2, 1.8, 2.0, 2.5, 2.3]
    det = detect_ou_shifts(
        tree,
        trait;
        max_shifts = 2,
        criterion = :mBIC,
    )

    @test det.success
    prof = profile_configurations(det)
    @test !isempty(prof)
    best = best_shift_configuration(det)
    @test best == det.shift_edges
    chosen = get_shift_configuration(det, det.n_shifts)
    @test chosen == det.shift_edges
    @test all(cfg.score <= prof[end].score for cfg in prof)
end
