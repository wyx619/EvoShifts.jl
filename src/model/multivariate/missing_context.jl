function _validate_multivariate_trait_shift(tree::CompactTree, trait::AbstractMatrix{<:Real})
    size(trait, 1) == tree.ntips || throw(ArgumentError("trait matrix must have $(tree.ntips) rows, got $(size(trait, 1))"))
    size(trait, 2) >= 1 || throw(ArgumentError("trait matrix must have at least 1 column"))
    data = Matrix{Float64}(trait)
    any(isinf, data) && throw(ArgumentError("trait matrix contains infinite values"))
    @inbounds for i in axes(data, 1)
        all(isnan, @view(data[i, :])) && throw(ArgumentError("trait matrix row $i contains no observed values"))
    end
    @inbounds for j in axes(data, 2)
        all(isnan, @view(data[:, j])) && throw(ArgumentError("trait matrix column $j contains no observed values"))
    end
    return data
end

function _l1ou_rescale_matrix(Y::AbstractMatrix{<:Real})
    X = Matrix{Float64}(Y)
    n = size(X, 1)
    @inbounds for j in axes(X, 2)
        nobs = 0
        s = 0.0
        ss_raw = 0.0
        for i in axes(X, 1)
            val = X[i, j]
            isnan(val) && continue
            nobs += 1
            s += val
            ss_raw += val * val
        end
        nobs > 0 || throw(ArgumentError("trait column $j contains no observed values"))
        mu = s / nobs
        for i in axes(X, 1)
            isnan(X[i, j]) && continue
            X[i, j] -= mu
        end
        normj = sqrt(ss_raw)
        scale = normj == 0.0 ? 1.0 : normj
        factor = 0.1 * n / scale
        for i in axes(X, 1)
            isnan(X[i, j]) && continue
            X[i, j] *= factor
        end
    end
    return X
end

Base.@kwdef struct MVShiftMissingContext
    has_missing::Bool = false
    observed_masks::Vector{BitVector} = BitVector[]
    observed_indices::Vector{Vector{Int}} = Vector{Int}[]
    proposal_rows::Vector{Vector{Int}} = Vector{Int}[]
    observed_counts::Vector{Int} = Int[]
    observed_traits::Vector{Vector{Float64}} = Vector{Float64}[]
    pruned_trees::Vector{CompactTree} = CompactTree[]
    pruned_caches::Vector{OUShiftTreeCache} = OUShiftTreeCache[]
    edge_map::Matrix{Int} = zeros(Int, 0, 0)
    full_rank_to_edge::Vector{Int} = Int[]
end

function _build_mv_shift_missing_context(tree::CompactTree, cache::OUShiftTreeCache, trait_mat::AbstractMatrix{<:Real})
    n, m = size(trait_mat)
    masks = Vector{BitVector}(undef, m)
    observed_indices = Vector{Vector{Int}}(undef, m)
    proposal_rows = Vector{Vector{Int}}(undef, m)
    counts = Vector{Int}(undef, m)
    observed_traits = Vector{Vector{Float64}}(undef, m)
    has_missing = false
    @inbounds for j in 1:m
        mask = falses(n)
        for i in 1:n
            observed = !isnan(Float64(trait_mat[i, j]))
            mask[i] = observed
            has_missing |= !observed
        end
        c = count(mask)
        c > 0 || throw(ArgumentError("trait matrix column $j contains no observed values"))
        masks[j] = mask
        obs_idx = Int[]
        sizehint!(obs_idx, c)
        prop_rows = Int[]
        sizehint!(prop_rows, min(c, max(n - 1, 0)))
        for i in 1:n
            if mask[i]
                push!(obs_idx, i)
                i < n && push!(prop_rows, i)
            end
        end
        observed_indices[j] = obs_idx
        proposal_rows[j] = prop_rows
        counts[j] = c
        y = Vector{Float64}(undef, c)
        out = 0
        for i in 1:n
            if mask[i]
                out += 1
                y[out] = Float64(trait_mat[i, j])
            end
        end
        observed_traits[j] = y
    end
    @inbounds for i in 1:n
        any(mask -> mask[i], masks) || throw(ArgumentError("trait matrix row $i contains no observed values"))
    end

    edge_map = zeros(Int, cache.nedges, m)
    full_rank_to_edge = zeros(Int, cache.nedges)
    @inbounds for e in 1:cache.nedges
        rank = cache.r_postorder_edge_rank[e]
        1 <= rank <= cache.nedges && (full_rank_to_edge[rank] = e)
    end
    pruned_trees = Vector{CompactTree}(undef, m)
    pruned_caches = Vector{OUShiftTreeCache}(undef, m)
    for j in 1:m
        if counts[j] == tree.ntips
            pruned_trees[j] = tree
            pruned_caches[j] = cache
            @inbounds for e in 1:cache.nedges
                edge_map[e, j] = e
            end
            continue
        end

        keep_labels = String[tree.tip_labels[i] for i in 1:n if masks[j][i]]
        ptr = keep_tip(tree, keep_labels)
        pcache = build_shift_tree_cache(ptr)
        pruned_trees[j] = ptr
        pruned_caches[j] = pcache
        set_to_edge = Dict{String, Int}()
        for e in 1:pcache.nedges
            labs = String[ptr.tip_labels[p] for p in pcache.descendant_tip_positions[e]]
            sort!(labs)
            set_to_edge[join(labs, "\0")] = e
        end
        for e in 1:cache.nedges
            labs = String[]
            for tip_pos in cache.descendant_tip_positions[e]
                masks[j][tip_pos] && push!(labs, tree.tip_labels[tip_pos])
            end
            isempty(labs) && continue
            length(labs) == counts[j] && continue
            sort!(labs)
            pe = get(set_to_edge, join(labs, "\0"), 0)
            if pe != 0 && Int(pcache.edge_parent[pe]) != Int(pcache.root)
                edge_map[e, j] = pe
            end
        end
    end
    return MVShiftMissingContext(
        has_missing = has_missing,
        observed_masks = masks,
        observed_indices = observed_indices,
        proposal_rows = proposal_rows,
        observed_counts = counts,
        observed_traits = observed_traits,
        pruned_trees = pruned_trees,
        pruned_caches = pruned_caches,
        edge_map = edge_map,
        full_rank_to_edge = full_rank_to_edge,
    )
end

@inline function _mv_shift_visible_edges(ctx::MVShiftMissingContext, trait_index::Integer, shift_edges::AbstractVector{<:Integer})
    return Int[ctx.edge_map[Int(e), Int(trait_index)] for e in shift_edges if ctx.edge_map[Int(e), Int(trait_index)] != 0]
end

@inline function _mv_shift_visible_full_edges(ctx::MVShiftMissingContext, trait_index::Integer, shift_edges::AbstractVector{<:Integer})
    return Int[Int(e) for e in shift_edges if ctx.edge_map[Int(e), Int(trait_index)] != 0]
end

function _mv_shift_observed_trait(ctx::MVShiftMissingContext, trait_mat::AbstractMatrix{<:Real}, trait_index::Integer)
    j = Int(trait_index)
    return ctx.observed_traits[j]
end
