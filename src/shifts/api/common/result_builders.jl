function _empty_shift_detection_result(
    tree::CompactTree;
    ntraits::Integer = 1,
    criterion::Symbol = :mBIC,
    message::AbstractString = "",
    diagnostics::NamedTuple = (;),
)
    edge_segments = shift_edges_to_edge_segments(tree, Int[])
    base_diagnostics = (
        message = String(message),
        criterion = criterion,
    )
    return OUShiftDetectionResult(
        success = true,
        model = :OUShifts,
        ntraits = Int(ntraits),
        shift_edges = Int[],
        n_shifts = 0,
        score = Inf,
        criterion = criterion,
        edge_regimes = _extract_edge_regimes(tree, edge_segments),
        edge_segments = edge_segments,
        diagnostics = merge(base_diagnostics, diagnostics),
    )
end

function _best_shift_detection_result(
    tree::CompactTree,
    best::OUShiftConfiguration,
    profile::Vector{OUShiftConfiguration};
    ntraits::Integer = 1,
    criterion::Symbol = :mBIC,
    diagnostics::NamedTuple = (;),
)
    edge_segments = shift_edges_to_edge_segments(tree, best.shift_edges)
    return OUShiftDetectionResult(
        success = true,
        model = :OUShifts,
        ntraits = Int(ntraits),
        shift_edges = copy(best.shift_edges),
        n_shifts = best.n_shifts,
        alpha = copy(best.alpha),
        sigma2 = copy(best.sigma2),
        loglik = copy(best.loglik),
        theta = copy(best.theta),
        shift_values = copy(best.shift_values),
        shift_means = copy(best.shift_means),
        fitted_means = copy(best.fitted_means),
        residuals = copy(best.residuals),
        edge_optima = copy(best.edge_optima),
        score = best.score,
        criterion = criterion,
        edge_regimes = _extract_edge_regimes(tree, edge_segments),
        edge_segments = edge_segments,
        profile = profile,
        diagnostics = diagnostics,
    )
end

