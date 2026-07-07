"""
    simulate_mvbm1(tree, Sigma; root_state=nothing, rng=Random.GLOBAL_RNG)

Simulate a BM1 trait matrix on a preprocessed tree.
"""
function simulate_mvbm1(
    tree::CompactTree,
    Sigma::AbstractMatrix;
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    size(Sigma, 1) == size(Sigma, 2) || throw(ArgumentError("Sigma must be square"))
    p = size(Sigma, 1)
    p >= 1 || throw(ArgumentError("Sigma must be at least 1x1"))
    issymmetric(Sigma) || throw(ArgumentError("Sigma must be symmetric"))
    chol = cholesky(Symmetric(Matrix{Float64}(Sigma)))
    L = Matrix{Float64}(chol.L)
    ws = _mvsim_gaussian_workspace(p)

    root_vec = if root_state === nothing
        zeros(Float64, p)
    else
        vec = collect(Float64, root_state)
        length(vec) == p || throw(ArgumentError("root_state length must match Sigma size"))
        vec
    end

    node_states = Matrix{Float64}(undef, tree.nnodes, p)
    node_states[tree.root, :] .= root_vec

    for node in tree.preorder
        parent_state = @view node_states[node, :]
        first_edge = tree.first_child_edge[node]
        first_edge == 0 && continue
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child = tree.child_of_edge[edge]
            t = tree.edge_length[edge]
            child_state = @view node_states[child, :]
            child_state .= parent_state
            _mvsim_add_chol_noise!(child_state, L, t, rng, ws)
        end
    end

    tip_data = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        @views tip_data[i, :] .= node_states[tip, :]
    end
    return tip_data
end

"""
    simulate_mvbmm(tree, edge_segments, Sigmas; root_state=nothing, rng=Random.GLOBAL_RNG)

Simulate a multi-regime BM trait matrix. `Sigmas[r]` is the diffusion covariance
for regime `r`.
"""
function simulate_mvbmm(
    tree::CompactTree,
    edge_segments::Vector{Vector{SimmapSegment}},
    Sigmas::AbstractVector{<:AbstractMatrix};
    root_state = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    _mvsim_validate_edge_segments(tree, edge_segments)
    nregimes = _mvsim_nregimes(edge_segments)
    length(Sigmas) == nregimes || throw(ArgumentError("Sigmas must have one covariance matrix per regime"))
    Sigma_mats = [_mvsim_validate_spd_matrix("Sigma[$i]", Sigmas[i]) for i in 1:nregimes]
    p = size(Sigma_mats[1], 1)
    all(size(S, 1) == p && size(S, 2) == p for S in Sigma_mats) || throw(ArgumentError("All regime covariance matrices must share the same dimensions"))
    chol_factors = [Matrix{Float64}(cholesky(Symmetric(S)).L) for S in Sigma_mats]
    ws = _mvsim_gaussian_workspace(p)

    root_vec = root_state === nothing ? zeros(Float64, p) : collect(Float64, root_state)
    length(root_vec) == p || throw(ArgumentError("root_state length must match covariance size"))

    node_states = Matrix{Float64}(undef, tree.nnodes, p)
    node_states[tree.root, :] .= root_vec

    for node in tree.preorder
        parent_state = @view node_states[node, :]
        first_edge = tree.first_child_edge[node]
        first_edge == 0 && continue
        last_edge = tree.last_child_edge[node]
        for edge in first_edge:last_edge
            child_state = @view node_states[tree.child_of_edge[edge], :]
            child_state .= parent_state
            for seg in edge_segments[edge]
                _mvsim_add_chol_noise!(child_state, chol_factors[Int(seg.state)], seg.length, rng, ws)
            end
        end
    end

    tip_data = Matrix{Float64}(undef, tree.ntips, p)
    for (i, tip) in enumerate(tree.tip_ids)
        @views tip_data[i, :] .= node_states[tip, :]
    end
    return tip_data
end
