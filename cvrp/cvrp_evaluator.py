from .cvrp_solution import CvrpSolution
from .cvrp_instance import CvrpInstance

class CvrpEvaluator:
    def __init__(self, instance: CvrpInstance):
        self.instance = instance

    def evaluate_objfun(self, solution: CvrpSolution) -> float:
        if solution.objfun_val is not None:
            return solution.objfun_val

        total_dist = 0.0
        dist_matrix = self.instance.distance_matrix
        depot = self.instance.depot

        for route in solution.routes:
            if not route:
                continue
            
            # From depot to first
            total_dist += dist_matrix[depot][route[0]]
            
            # Between customers
            for i in range(len(route) - 1):
                total_dist += dist_matrix[route[i]][route[i+1]]
            
            # From last to depot
            total_dist += dist_matrix[route[-1]][depot]

        solution.objfun_val = total_dist
        return total_dist

    def is_feasible(self, solution: CvrpSolution) -> bool:
        # Check capacity constraints
        for route in solution.routes:
            load = sum(self.instance.demands[node] for node in route)
            if load > self.instance.capacity:
                return False
        
        # Check if all customers are visited exactly once
        visited = set()
        for route in solution.routes:
            for node in route:
                if node in visited:
                    return False # Duplicate
                visited.add(node)
        
        if len(visited) != self.instance.dimension - 1: # Exclude depot
            return False # Not all visited
            
        return True

    def evaluate_relocate_delta(self, solution: CvrpSolution, 
                                source_route_idx: int, source_node_idx: int, 
                                target_route_idx: int, target_pos_idx: int) -> float:
        """
        Evaluates the change in cost if we move the node at solution.routes[source_route_idx][source_node_idx]
        to solution.routes[target_route_idx] at position target_pos_idx.
        """
        dist_matrix = self.instance.distance_matrix
        depot = self.instance.depot
        
        source_route = solution.routes[source_route_idx]
        target_route = solution.routes[target_route_idx]
        
        node = source_route[source_node_idx]
        
        # Cost removal from source
        prev_node = source_route[source_node_idx - 1] if source_node_idx > 0 else depot
        next_node = source_route[source_node_idx + 1] if source_node_idx < len(source_route) - 1 else depot
        
        cost_removed = dist_matrix[prev_node][node] + dist_matrix[node][next_node]
        cost_added_source = dist_matrix[prev_node][next_node]
        
        delta = cost_added_source - cost_removed
        
        # Cost insertion into target
        # Note: if source_route == target_route, indices might shift. 
        # We assume indices are based on the state BEFORE the move.
        # If same route, we need to be careful.
        
        if source_route_idx == target_route_idx:
            # Same route relocation
            # It's easier to simulate or handle logic carefully.
            # For simplicity in this "delta" function, let's assume we calculate based on "removing then inserting".
            # But if we remove first, the target_pos_idx might change if it was after source_node_idx.
            
            # Let's handle same route separately or just use a simpler approach for now?
            # To be robust, let's just calculate the cost of the modified route vs original route.
            # It's O(N) for the route length, which is small.
            
            # Construct new route
            new_route = source_route[:]
            del new_route[source_node_idx]
            
            # Adjust insertion index
            insert_idx = target_pos_idx
            if source_node_idx < target_pos_idx:
                insert_idx -= 1
            
            new_route.insert(insert_idx, node)
            
            # Calculate diff
            return self._calculate_route_cost(new_route) - self._calculate_route_cost(source_route)

        else:
            # Different routes
            # Removal delta already calculated (delta)
            
            # Insertion delta
            # Target route is currently target_route
            # We insert at target_pos_idx
            
            prev_node_t = target_route[target_pos_idx - 1] if target_pos_idx > 0 else depot
            next_node_t = target_route[target_pos_idx] if target_pos_idx < len(target_route) else depot
            
            cost_removed_target = dist_matrix[prev_node_t][next_node_t]
            cost_added_target = dist_matrix[prev_node_t][node] + dist_matrix[node][next_node_t]
            
            delta += (cost_added_target - cost_removed_target)
            
            return delta

    def _calculate_route_cost(self, route: list) -> float:
        if not route:
            return 0.0
        dist = 0.0
        depot = self.instance.depot
        matrix = self.instance.distance_matrix
        
        dist += matrix[depot][route[0]]
        for i in range(len(route)-1):
            dist += matrix[route[i]][route[i+1]]
        dist += matrix[route[-1]][depot]
        return dist

    def check_capacity(self, solution: CvrpSolution, 
                       source_route_idx: int, source_node_idx: int, 
                       target_route_idx: int) -> bool:
        """
        Checks if moving a node from source to target route violates capacity of target route.
        """
        if source_route_idx == target_route_idx:
            return True # Capacity doesn't change for same route move
            
        node = solution.routes[source_route_idx][source_node_idx]
        demand = self.instance.demands[node]
        
        current_load = sum(self.instance.demands[n] for n in solution.routes[target_route_idx])
        
        return (current_load + demand) <= self.instance.capacity

    def check_swap_capacity(self, solution: CvrpSolution, 
                            r1_idx: int, p1_idx: int, 
                            r2_idx: int, p2_idx: int) -> bool:
        """
        Checks if swapping two nodes violates capacity constraints.
        """
        if r1_idx == r2_idx:
            return True
            
        node1 = solution.routes[r1_idx][p1_idx]
        node2 = solution.routes[r2_idx][p2_idx]
        
        demand1 = self.instance.demands[node1]
        demand2 = self.instance.demands[node2]
        
        load1 = sum(self.instance.demands[n] for n in solution.routes[r1_idx])
        load2 = sum(self.instance.demands[n] for n in solution.routes[r2_idx])
        
        if load1 - demand1 + demand2 > self.instance.capacity:
            return False
        if load2 - demand2 + demand1 > self.instance.capacity:
            return False
            
        return True

    def evaluate_swap_delta(self, solution: CvrpSolution, 
                            r1_idx: int, p1_idx: int, 
                            r2_idx: int, p2_idx: int) -> float:
        """
        Evaluates the cost change of swapping two nodes.
        """
        if r1_idx == r2_idx and p1_idx == p2_idx:
            return 0.0
            
        route1 = solution.routes[r1_idx]
        route2 = solution.routes[r2_idx]
        
        node1 = route1[p1_idx]
        node2 = route2[p2_idx]
        
        if r1_idx == r2_idx:
            # Intra-route swap
            # Create new route and calculate diff
            new_route = route1[:]
            new_route[p1_idx] = node2
            new_route[p2_idx] = node1
            return self._calculate_route_cost(new_route) - self._calculate_route_cost(route1)
        else:
            # Inter-route swap
            # Calculate delta for route 1
            # Remove node1, insert node2 at p1_idx
            
            # Optimization: calculate local delta instead of full route cost
            # But full route cost is safer and O(N) is small for VRP routes usually.
            
            new_route1 = route1[:]
            new_route1[p1_idx] = node2
            
            new_route2 = route2[:]
            new_route2[p2_idx] = node1
            
            cost1_diff = self._calculate_route_cost(new_route1) - self._calculate_route_cost(route1)
            cost2_diff = self._calculate_route_cost(new_route2) - self._calculate_route_cost(route2)
            
            return cost1_diff + cost2_diff
