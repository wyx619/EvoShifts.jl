function _as_vector_float(x)
    x isa AbstractVector && return Float64.(x)
    x isa Real && return [Float64(x)]
    return Float64[]
end

function _shift_branch_rows(result)
    isempty(result.shift_edges) && return NamedTuple{(:R_edge_id_postorder, :tipX, :tipY, :descendant_signature), Tuple{Int, String, String, String}}[]
    haskey(result.diagnostics, :source_tree) ||
        throw(ArgumentError("result diagnostics do not include source_tree; cannot map shift edges to branch identities"))
    tree = result.diagnostics.source_tree
    tbl = R_edge_table(tree; edges = Int.(result.shift_edges), order = :postorder)
    branches = Vector{NamedTuple{(:R_edge_id_postorder, :tipX, :tipY, :descendant_signature), Tuple{Int, String, String, String}}}(undef, nrow(tbl))
    @inbounds for i in 1:nrow(tbl)
        branches[i] = (
            R_edge_id_postorder = Int(tbl.R_edge_id[i]),
            tipX = String(tbl.tipX[i]),
            tipY = String(tbl.tipY[i]),
            descendant_signature = String(tbl.descendant_signature[i]),
        )
    end
    return branches
end

function _shift_branch_strings(branches)
    return String["$(b.tipX)|$(b.tipY)" for b in branches]
end

function _shift_branch_signatures(branches)
    return String[b.descendant_signature for b in branches]
end

function _shift_branch_R_ids(branches)
    return Int[b.R_edge_id_postorder for b in branches]
end

function _shift_summary_common(result)
    loglik_vec = _as_vector_float(result.loglik)
    branches = _shift_branch_rows(result)
    return (
        success = result.success,
        model = result.model,
        ntraits = hasproperty(result, :ntraits) ? result.ntraits : length(loglik_vec),
        n_shifts = result.n_shifts,
        shift_branches = branches,
        R_edge_ids_postorder = _shift_branch_R_ids(branches),
        descendant_signatures = _shift_branch_signatures(branches),
        criterion = result.criterion,
        score = result.score,
        loglik = isempty(loglik_vec) ? NaN : sum(loglik_vec),
        alpha = _as_vector_float(result.alpha),
        sigma2 = _as_vector_float(result.sigma2),
    )
end

"""
    shift_detection_summary(result)

Return a compact `NamedTuple` summary for an OU shift detection or fixed
configuration fit result.
"""
function shift_detection_summary(result::OUShiftDetectionResult)
    return _shift_summary_common(result)
end

function shift_detection_summary(result::OUShiftFitResult)
    return _shift_summary_common(result)
end

"""
    shift_detection_summary_table(result)

Return a one-row `DataFrame` summary. Vector fields such as `shift_branches`,
`alpha`, and `sigma2` are stored as comma-separated strings for easy CSV output.
"""
function shift_detection_summary_table(result::Union{OUShiftDetectionResult, OUShiftFitResult})
    s = shift_detection_summary(result)
    return DataFrames.DataFrame(
        success = [s.success],
        model = [s.model],
        ntraits = [s.ntraits],
        n_shifts = [s.n_shifts],
        R_edge_ids_postorder = [join(s.R_edge_ids_postorder, ",")],
        shift_branches = [join(_shift_branch_strings(s.shift_branches), ",")],
        descendant_signatures = [join(s.descendant_signatures, ",")],
        criterion = [s.criterion],
        score = [s.score],
        loglik = [s.loglik],
        alpha = [join(s.alpha, ",")],
        sigma2 = [join(s.sigma2, ",")],
    )
end

