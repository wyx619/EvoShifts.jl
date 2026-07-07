Base.@kwdef struct SimulatedTree
    parent::Vector{Int32}
    children::Vector{Vector{Int32}}
    node_time::Vector{Float64}
    is_tip::BitVector
    tip_labels::Vector{String}
end

@inline function _mvsim_symmetrize(M::AbstractMatrix)
    return Symmetric((Matrix{Float64}(M) + Matrix{Float64}(M)') / 2)
end

function _mvsim_sample_cov(cov::AbstractMatrix, rng::AbstractRNG)
    S = _mvsim_symmetrize(cov)
    vals, vecs = eigen(S)
    any(x -> x < -1e-8, vals) && throw(ArgumentError("Covariance matrix is not positive semidefinite"))
    scales = sqrt.(max.(vals, 0.0))
    return vecs * (scales .* randn(rng, length(vals)))
end

mutable struct _MVGaussianStepWorkspace
    z::Vector{Float64}
    noise::Vector{Float64}
end

function _mvsim_gaussian_workspace(p::Integer)
    return _MVGaussianStepWorkspace(zeros(Float64, p), zeros(Float64, p))
end

@inline function _mvsim_add_chol_noise!(
    dest::AbstractVector{Float64},
    L::AbstractMatrix{Float64},
    scale::Float64,
    rng::AbstractRNG,
    ws::_MVGaussianStepWorkspace,
)
    scale == 0.0 && return dest
    scale >= 0.0 || throw(ArgumentError("Gaussian variance scale must be non-negative"))
    p = length(ws.z)
    if p == 1
        dest[1] += sqrt(scale) * L[1, 1] * randn(rng)
        return dest
    end
    randn!(rng, ws.z)
    mul!(ws.noise, L, ws.z)
    s = sqrt(scale)
    @inbounds for i in 1:p
        dest[i] += s * ws.noise[i]
    end
    return dest
end

mutable struct _MVOUProcessWorkspace
    p::Int
    K::Matrix{Float64}
    K_lu::Any
    rhs::Vector{Float64}
    vec_work::Vector{Float64}
    z::Vector{Float64}
    centered::Vector{Float64}
    noise::Vector{Float64}
    Phi::Matrix{Float64}
    cov::Matrix{Float64}
    expK::Matrix{Float64}
    A_scaled::Matrix{Float64}
    K_scaled::Matrix{Float64}
end

function _mvsim_ou_workspace(A::AbstractMatrix, Sigma::AbstractMatrix)
    p = size(A, 1)
    K = kron(I(p), A) + kron(A, I(p))
    return _MVOUProcessWorkspace(
        p,
        Matrix{Float64}(K),
        lu(Matrix{Float64}(K)),
        vec(Matrix{Float64}(Sigma)),
        zeros(Float64, p * p),
        zeros(Float64, p),
        zeros(Float64, p),
        zeros(Float64, p),
        zeros(Float64, p, p),
        zeros(Float64, p, p),
        zeros(Float64, p * p, p * p),
        zeros(Float64, p, p),
        zeros(Float64, p * p, p * p),
    )
end

function _mvsim_validate_spd_matrix(name::AbstractString, M::AbstractMatrix)
    size(M, 1) == size(M, 2) || throw(ArgumentError("$name must be square"))
    size(M, 1) >= 1 || throw(ArgumentError("$name must be at least 1x1"))
    issymmetric(M) || throw(ArgumentError("$name must be symmetric"))
    vals = eigvals(_mvsim_symmetrize(M))
    any(x -> x <= 0.0, vals) && throw(ArgumentError("$name must be positive definite"))
    return Matrix{Float64}(M)
end

function _mvsim_validate_stable_matrix(name::AbstractString, M::AbstractMatrix)
    size(M, 1) == size(M, 2) || throw(ArgumentError("$name must be square"))
    size(M, 1) >= 1 || throw(ArgumentError("$name must be at least 1x1"))
    A = Matrix{Float64}(M)
    vals = eigvals(A)
    any(z -> real(z) <= 0.0, vals) && throw(ArgumentError("$name must have eigenvalues with positive real parts"))
    return A
end

function _mvsim_nregimes(edge_segments::Vector{Vector{SimmapSegment}})
    max_state = 0
    for segments in edge_segments, seg in segments
        max_state = max(max_state, Int(seg.state))
    end
    max_state >= 1 || throw(ArgumentError("edge_segments must contain at least one regime state"))
    return max_state
end

function _mvsim_validate_edge_segments(tree::CompactTree, edge_segments::Vector{Vector{SimmapSegment}}; atol::Float64 = 1e-8)
    length(edge_segments) == tree.nedges || throw(ArgumentError("edge_segments must have $(tree.nedges) entries"))
    for edge in 1:tree.nedges
        segments = edge_segments[edge]
        isempty(segments) && throw(ArgumentError("edge_segments[$edge] is empty"))
        total = sum(seg.length for seg in segments)
        isapprox(total, tree.edge_length[edge]; atol = atol) ||
            throw(ArgumentError("edge_segments[$edge] does not sum to branch length"))
        for seg in segments
            seg.state >= 1 || throw(ArgumentError("Regime states must be positive integers"))
            seg.length >= 0.0 || throw(ArgumentError("Segment lengths must be non-negative"))
        end
        for i in 2:length(segments)
            segments[i - 1].state == segments[i].state &&
                throw(ArgumentError("Adjacent segments on edge $edge must be merged"))
        end
    end
    return edge_segments
end

function _mvsim_ou_covariance(A::AbstractMatrix, Sigma::AbstractMatrix, t::Real)
    p = size(A, 1)
    t == 0 && return zeros(Float64, p, p)
    K = kron(I(p), A) + kron(A, I(p))
    rhs = vec(Matrix{Float64}(Sigma))
    vecV = K \ ((I - exp(-Matrix(K) * Float64(t))) * rhs)
    return reshape(vecV, p, p)
end

function _mvsim_ou_covariance!(
    dest::AbstractMatrix{Float64},
    A::AbstractMatrix{Float64},
    Sigma::AbstractMatrix{Float64},
    t::Float64,
    ws::_MVOUProcessWorkspace,
)
    p = ws.p
    if t == 0.0
        fill!(dest, 0.0)
        return dest
    elseif p == 1
        a = A[1, 1]
        dest[1, 1] = Sigma[1, 1] * (1.0 - exp(-2.0 * a * t)) / (2.0 * a)
        return dest
    end

    @. ws.K_scaled = -ws.K * t
    ws.expK .= exp(ws.K_scaled)
    mul!(ws.vec_work, ws.expK, ws.rhs)
    @. ws.vec_work = ws.rhs - ws.vec_work
    ldiv!(ws.K_lu, ws.vec_work)
    copyto!(dest, reshape(ws.vec_work, p, p))
    dest .= (dest .+ dest') ./ 2.0
    return dest
end

function _mvsim_sample_cov!(dest::AbstractVector{Float64}, cov::AbstractMatrix{Float64}, rng::AbstractRNG, ws::_MVOUProcessWorkspace)
    p = ws.p
    if p == 1
        dest[1] = sqrt(max(cov[1, 1], 0.0)) * randn(rng)
        return dest
    end
    F = cholesky!(Symmetric(cov); check = false)
    if !issuccess(F)
        cov .+= 1e-10I
        F = cholesky!(Symmetric(cov); check = false)
        issuccess(F) || throw(ArgumentError("Covariance matrix is not positive semidefinite"))
    end
    randn!(rng, ws.z)
    mul!(dest, F.L, ws.z)
    return dest
end

function _mvsim_ou_step!(
    dest::AbstractVector{Float64},
    state::AbstractVector,
    A::AbstractMatrix{Float64},
    Sigma::AbstractMatrix{Float64},
    theta::AbstractVector{Float64},
    t::Real,
    rng::AbstractRNG,
    ws::_MVOUProcessWorkspace,
)
    p = ws.p
    tval = Float64(t)
    if p == 1
        a = A[1, 1]
        phi = exp(-a * tval)
        var = Sigma[1, 1] * (1.0 - exp(-2.0 * a * tval)) / (2.0 * a)
        dest[1] = theta[1] + phi * (state[1] - theta[1]) + sqrt(max(var, 0.0)) * randn(rng)
        return dest
    end

    @. ws.A_scaled = -A * tval
    ws.Phi .= exp(ws.A_scaled)
    @inbounds for i in 1:p
        ws.centered[i] = state[i] - theta[i]
    end
    mul!(dest, ws.Phi, ws.centered)
    @inbounds for i in 1:p
        dest[i] += theta[i]
    end
    _mvsim_ou_covariance!(ws.cov, A, Sigma, tval, ws)
    _mvsim_sample_cov!(ws.noise, ws.cov, rng, ws)
    @inbounds for i in 1:p
        dest[i] += ws.noise[i]
    end
    return dest
end

function _mvsim_ou_step(
    state::AbstractVector,
    A::AbstractMatrix,
    Sigma::AbstractMatrix,
    theta::AbstractVector,
    t::Real,
    rng::AbstractRNG,
)
    p = length(state)
    p == size(A, 1) == size(A, 2) || throw(ArgumentError("State and A dimensions do not match"))
    p == size(Sigma, 1) == size(Sigma, 2) || throw(ArgumentError("State and Sigma dimensions do not match"))
    length(theta) == p || throw(ArgumentError("State and theta dimensions do not match"))
    Phi = exp(-Matrix{Float64}(A) * Float64(t))
    mean = theta .+ Phi * (Vector{Float64}(state) .- Vector{Float64}(theta))
    cov = _mvsim_ou_covariance(A, Sigma, t)
    return mean .+ _mvsim_sample_cov(cov, rng)
end
