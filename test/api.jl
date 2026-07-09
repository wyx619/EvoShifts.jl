using Test
using DataFrames
using EvoShifts

function _shift_missing_tree()
    path = joinpath(mktempdir(), "shift_missing_tree.tre")
    write(path, "(((A:1,B:1):1,(C:1,D:1):1):1,(E:1,F:1):2);")
    return to_compact_tree(load_newick_tree(path))
end

function _edge_by_signature(tree, sig::AbstractString)
    sigs = shift_edge_signatures(tree, collect(1:tree.nedges))
    idx = findfirst(==(sig), sigs)
    idx === nothing && error("signature $sig not found")
    return idx
end

@testset "multivariate shift detection missing validation" begin
    tree = _shift_missing_tree()
    X = [
        1.0  NaN;
        1.1  NaN;
        2.0  2.1;
        2.1  2.2;
        3.0  2.9;
        3.1  3.0;
    ]
    @test isequal(EvoShifts._validate_multivariate_trait_shift(tree, X), X)

    row_missing = copy(X)
    row_missing[3, :] .= NaN
    @test_throws ArgumentError EvoShifts._validate_multivariate_trait_shift(tree, row_missing)

    col_missing = copy(X)
    col_missing[:, 2] .= NaN
    @test_throws ArgumentError EvoShifts._validate_multivariate_trait_shift(tree, col_missing)

    y = [1.0, NaN, 2.0, 2.1, 3.0, 3.1]
    @test_throws ArgumentError detect_ou_shifts(tree, y)
end

@testset "align_traits_to_tree multivariate missing" begin
    tree = _shift_missing_tree()
    labels = collect(tree.tip_labels)
    df = DataFrame(
        species = reverse(labels),
        trait1 = reverse([1.0, 1.1, 2.0, 2.1, 3.0, 3.1]),
        trait2 = reverse([missing, missing, 2.1, 2.2, 2.9, 3.0]),
    )
    X = align_traits_to_tree(tree, df; taxon_col = :species, trait_cols = [:trait1, :trait2])
    @test size(X) == (tree.ntips, 2)
    @test isnan(X[1, 2])
    @test isnan(X[2, 2])
    @test X[:, 1] == [1.0, 1.1, 2.0, 2.1, 3.0, 3.1]

    @test_throws ArgumentError align_traits_to_tree(tree, df; taxon_col = :species, trait_cols = :trait2)

    df_string = DataFrame(
        species = labels,
        trait1 = string.([1.0, 1.1, 2.0, 2.1, 3.0, 3.1]),
        trait2 = ["NA", "1.2", "NaN", "2.2", "3.0", "3.1"],
    )
    X_string = align_traits_to_tree(tree, df_string; taxon_col = :species, trait_cols = [:trait1, :trait2])
    @test X_string[:, 1] == [1.0, 1.1, 2.0, 2.1, 3.0, 3.1]
    @test isnan(X_string[1, 2])
    @test isnan(X_string[3, 2])
    @test_throws ArgumentError align_traits_to_tree(tree, df_string; taxon_col = :species, trait_cols = :trait2)
end

@testset "multivariate shift fit and detect with partial missing" begin
    tree = _shift_missing_tree()
    edge_ab = _edge_by_signature(tree, "A|B")
    edge_cd = _edge_by_signature(tree, "C|D")
    X = [
        1.0  NaN;
        1.1  NaN;
        2.0  2.1;
        2.1  2.2;
        3.0  2.9;
        3.1  3.0;
    ]

    fit = fit_ou_shifts(
        tree,
        X,
        [edge_ab];
        criterion = :BIC,
        max_iterations = 80,
        rel_tol = 1e-6,
    )
    @test fit.success
    @test size(fit.shift_values) == (1, 2)
    @test isfinite(fit.shift_values[1, 1])
    @test isnan(fit.shift_values[1, 2])
    @test all(isfinite, fit.loglik)
    @test all(isfinite, fit.fitted_means)
    @test isnan(fit.residuals[1, 2])
    @test isnan(fit.residuals[2, 2])

    det = detect_ou_shifts(
        tree,
        X;
        criterion = :BIC,
        max_shifts = 1,
        candidate_edges = [edge_ab, edge_cd],
    )
    @test det.success
    @test det.diagnostics.missing_pattern == :multivariate_partial
    @test det.diagnostics.proposal_method == :multivariate_path
    @test !det.diagnostics.path_anchor
    @test det.diagnostics.path_anchor_source == :none
    @test det.diagnostics.observed_counts == [6, 4]
    @test all(isfinite, det.loglik)
    @test isfinite(det.score)

    cache = EvoShifts.build_shift_tree_cache(tree)
    ctx = EvoShifts._build_mv_shift_missing_context(tree, cache, X)
    proposal = EvoShifts._propose_shift_configs_multivariate_group_path(
        tree,
        cache,
        X,
        [edge_ab, edge_cd],
        [0.0, 0.0];
        max_shifts = 1,
        missing_context = ctx,
    )
    @test proposal.diagnostics.proposal_method == :l1ou_sqrt_inv_cov_missing
end
