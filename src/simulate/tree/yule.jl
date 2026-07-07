function simulate_yule_simtree(
    n_tips::Integer;
    birth_rate::Real = 1.0,
    tree_height::Union{Nothing, Real} = nothing,
    tip_prefix::AbstractString = "t",
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    n_tips >= 2 || throw(ArgumentError("simulate_yule_tree requires at least 2 tips"))
    birth_rate > 0 || throw(ArgumentError("birth_rate must be positive"))
    tree_height === nothing || tree_height > 0 || throw(ArgumentError("tree_height must be positive"))

    max_nodes = 2 * Int(n_tips) - 1
    parent = zeros(Int32, max_nodes)
    children = [Int32[] for _ in 1:max_nodes]
    node_time = zeros(Float64, max_nodes)
    active = Int32[1]
    next_node = 2
    current_time = 0.0

    while length(active) < n_tips
        current_time += randexp(rng) / (Float64(birth_rate) * length(active))
        idx = rand(rng, 1:length(active))
        lineage = active[idx]
        left = Int32(next_node)
        right = Int32(next_node + 1)
        next_node += 2
        node_time[Int(left)] = current_time
        node_time[Int(right)] = current_time
        parent[Int(left)] = lineage
        parent[Int(right)] = lineage
        push!(children[Int(lineage)], left)
        push!(children[Int(lineage)], right)
        active[idx] = left
        push!(active, right)
    end

    final_time = current_time
    if tree_height !== nothing
        scale = Float64(tree_height) / final_time
        node_time .*= scale
        final_time = Float64(tree_height)
    end
    is_tip = falses(max_nodes)
    for tip in active
        is_tip[Int(tip)] = true
        node_time[Int(tip)] = final_time
    end
    tip_labels = ["$(tip_prefix)$(i)" for i in 1:n_tips]
    return SimulatedTree(
        parent = parent,
        children = children,
        node_time = node_time,
        is_tip = is_tip,
        tip_labels = tip_labels,
    )
end

function simulate_yule_tree(args...; kwargs...)
    return to_compact_tree(simulate_yule_simtree(args...; kwargs...))
end
