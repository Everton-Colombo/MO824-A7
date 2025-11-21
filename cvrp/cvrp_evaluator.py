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
        
        prev_node = source_route[source_node_idx - 1] if source_node_idx > 0 else depot
        next_node = source_route[source_node_idx + 1] if source_node_idx < len(source_route) - 1 else depot
        
        cost_removed = dist_matrix[prev_node][node] + dist_matrix[node][next_node]
        cost_added_source = dist_matrix[prev_node][next_node]
        
        delta = cost_added_source - cost_removed
        
        if source_route_idx == target_route_idx:
            new_route = source_route[:]
            del new_route[source_node_idx]
            
            insert_idx = target_pos_idx
            if source_node_idx < target_pos_idx:
                insert_idx -= 1
            
            new_route.insert(insert_idx, node)
            
            return self._calculate_route_cost(new_route) - self._calculate_route_cost(source_route)

        else:
            prev_node_t = target_route[target_pos_idx - 1] if target_pos_idx > 0 else depot
            next_node_t = target_route[target_pos_idx] if target_pos_idx < len(target_route) else depot
            
            cost_removed_target = dist_matrix[prev_node_t][next_node_t]
            cost_added_target = dist_matrix[prev_node_t][node] + dist_matrix[node][next_node_t]
            
            delta += (cost_added_target - cost_removed_target)
            
            return delta

    def evaluate_swap_delta(self, solution: CvrpSolution, 
                            r1_idx: int, p1_idx: int, 
                            r2_idx: int, p2_idx: int) -> float:
        """Calculate the change in cost when swapping the positions of two nodes."""
        r1 = solution.routes[r1_idx]
        r2 = solution.routes[r2_idx]
        
        node1 = r1[p1_idx]
        node2 = r2[p2_idx]
        
        if r1_idx == r2_idx and abs(p1_idx - p2_idx) == 1:
            new_route = r1[:]
            new_route[p1_idx], new_route[p2_idx] = new_route[p2_idx], new_route[p1_idx]
            return self._calculate_route_cost(new_route) - self._calculate_route_cost(r1)

        cost_rem_1 = self._calc_removal_gain(r1, p1_idx)
        cost_rem_2 = self._calc_removal_gain(r2, p2_idx)

        cost_old = self._calculate_route_cost(r1)
        if r1_idx != r2_idx:
             cost_old += self._calculate_route_cost(r2)
             
        if r1_idx == r2_idx:
            new_r1 = r1[:]
            new_r1[p1_idx], new_r1[p2_idx] = new_r1[p2_idx], new_r1[p1_idx]
            cost_new = self._calculate_route_cost(new_r1)
        else:
            new_r1 = r1[:]
            new_r1[p1_idx] = node2
            new_r2 = r2[:]
            new_r2[p2_idx] = node1
            cost_new = self._calculate_route_cost(new_r1) + self._calculate_route_cost(new_r2)
            
        return cost_new - cost_old

    def evaluate_2opt_delta(self, solution: CvrpSolution, route_idx: int, i: int, j: int) -> float:
        """Calculate the discriminant (delta) of reversing the segment between i and j on the same route."""
        route = solution.routes[route_idx]
        if i >= j: return 0.0
        
        dist_matrix = self.instance.distance_matrix
        depot = self.instance.depot
        
        node_A = route[i-1] if i > 0 else depot
        node_B = route[i]
        node_C = route[j]
        node_D = route[j+1] if j < len(route) - 1 else depot
        
        removed = dist_matrix[node_A][node_B] + dist_matrix[node_C][node_D]
        
        added = dist_matrix[node_A][node_C] + dist_matrix[node_B][node_D]
        
        return added - removed

    def check_oropt_capacity(self, solution: CvrpSolution, 
                             r1_idx: int, p1_idx: int, k: int, 
                             r2_idx: int) -> bool:
        """
        Checks whether moving a chain of 'k' nodes from R1 to R2 violates the capacity of R2.
        """
        if r1_idx == r2_idx:
            return True
        
        route1 = solution.routes[r1_idx]
        chain_demand = sum(self.instance.demands[node] for node in route1[p1_idx : p1_idx + k])
        
        current_load_r2 = sum(self.instance.demands[node] for node in solution.routes[r2_idx])
        
        return (current_load_r2 + chain_demand) <= self.instance.capacity

    def evaluate_oropt_delta(self, solution: CvrpSolution, 
                             r1_idx: int, p1_idx: int, k: int,
                             r2_idx: int, p2_idx: int) -> float:
        """
        Calculate the cost variance when moving a chain of 'k' nodes from route r1 (index p1)
        to route r2 (inserting at position p2).
        """
        r1 = solution.routes[r1_idx]
        r2 = solution.routes[r2_idx]
        
        if r1_idx == r2_idx:
            new_route = r1[:]
            
            chain = new_route[p1_idx : p1_idx + k]
            del new_route[p1_idx : p1_idx + k]
            
            insert_pos = p2_idx
            if p1_idx < p2_idx:
                insert_pos -= k
            
            new_route[insert_pos:insert_pos] = chain
            
            return self._calculate_route_cost(new_route) - self._calculate_route_cost(r1)
            
        else:
            new_r1 = r1[:]
            new_r2 = r2[:]
            
            chain = new_r1[p1_idx : p1_idx + k]
            del new_r1[p1_idx : p1_idx + k]
            
            new_r2[p2_idx:p2_idx] = chain
            
            cost_old = self._calculate_route_cost(r1) + self._calculate_route_cost(r2)
            cost_new = self._calculate_route_cost(new_r1) + self._calculate_route_cost(new_r2)
            
            return cost_new - cost_old

    def check_cross_exchange_capacity(self, solution: CvrpSolution, 
                                      r1_idx: int, p1_idx: int, len1: int,
                                      r2_idx: int, p2_idx: int, len2: int) -> bool:
        """Check capacity by swapping segment 1 (of r1) with segment 2 (of r2)."""
        route1 = solution.routes[r1_idx]
        route2 = solution.routes[r2_idx]
        
        dem1 = sum(self.instance.demands[n] for n in route1[p1_idx : p1_idx + len1])
        dem2 = sum(self.instance.demands[n] for n in route2[p2_idx : p2_idx + len2])
        
        load1 = sum(self.instance.demands[n] for n in route1)
        load2 = sum(self.instance.demands[n] for n in route2)
        
        if load1 - dem1 + dem2 > self.instance.capacity: return False
        if load2 - dem2 + dem1 > self.instance.capacity: return False
        
        return True

    def evaluate_cross_exchange_delta(self, solution: CvrpSolution, 
                                      r1_idx: int, p1_idx: int, len1: int,
                                      r2_idx: int, p2_idx: int, len2: int) -> float:
        r1 = solution.routes[r1_idx]
        r2 = solution.routes[r2_idx]
        
        seg1 = r1[p1_idx : p1_idx + len1]
        seg2 = r2[p2_idx : p2_idx + len2]
        
        new_r1 = r1[:p1_idx] + seg2 + r1[p1_idx + len1:]
        new_r2 = r2[:p2_idx] + seg1 + r2[p2_idx + len2:]
        
        cost_old = self._calculate_route_cost(r1) + self._calculate_route_cost(r2)
        cost_new = self._calculate_route_cost(new_r1) + self._calculate_route_cost(new_r2)
        
        return cost_new - cost_old

    def _calc_removal_gain(self, route, idx):
        """Helper para calcular quanto custa remover um nó (sem inserir nada)."""
        dist = self.instance.distance_matrix
        depot = self.instance.depot
        prev_n = route[idx-1] if idx > 0 else depot
        node = route[idx]
        next_n = route[idx+1] if idx < len(route)-1 else depot
        
        current_edges = dist[prev_n][node] + dist[node][next_n]
        new_edge = dist[prev_n][next_n]
        return current_edges - new_edge

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
