from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple, Dict
import math
import heapq
import itertools

from ...cvrp_instance import CvrpInstance
from ...cvrp_evaluator import CvrpEvaluator
from ...cvrp_solution import CvrpSolution

class ConstructiveHeuristicType(Enum):
    """Types of constructive heuristics available for CVRP."""
    SAVINGS = 'savings'
    CHEAPEST_INSERTION = 'cheapest_insertion'
    ROUTE_FIRST_CLUSTER_SECOND = 'route_first_cluster_second'

class BaseConstructiveHeuristic(ABC):
    """Abstract base class for all CVRP constructive heuristics."""

    def __init__(self, instance: CvrpInstance, evaluator: CvrpEvaluator):
        self.instance = instance
        self.evaluator = evaluator

    @abstractmethod
    def construct(self, randomize: bool = False) -> CvrpSolution:
        """
        Generates and returns a valid initial solution.
        """
        pass

class SavingsHeuristic(BaseConstructiveHeuristic):
    """
    Implementation of the Clarke & Wright Savings Algorithm.
    1. Start with n routes: depot -> i -> depot.
    2. Calculate savings S_ij = d_0i + d_0j - d_ij.
    3. Merge routes based on best savings.
    """

    def construct(self, randomize: bool = False) -> CvrpSolution:
        n = self.instance.dimension
        depot = self.instance.depot
        demands = self.instance.demands
        dist_matrix = self.instance.distance_matrix
        capacity = self.instance.capacity

        savings = []
        customers = [i for i in range(n) if i != depot]
        
        for i, j in itertools.combinations(customers, 2):
            s_ij = dist_matrix[depot][i] + dist_matrix[depot][j] - dist_matrix[i][j]
            
            if randomize:
                noise = random.uniform(0.85, 1.15)
                s_ij *= noise
            
            savings.append((s_ij, i, j))
        
        savings.sort(key=lambda x: x[0], reverse=True)

        routes = [[node] for node in customers]
        route_loads = {idx: demands[node] for idx, node in enumerate(customers)}
        node_to_route_idx = {node: idx for idx, node in enumerate(customers)}

        for s_val, i, j in savings:
            r_i_idx = node_to_route_idx[i]
            r_j_idx = node_to_route_idx[j]

            if r_i_idx == r_j_idx: continue

            route_i = routes[r_i_idx]
            route_j = routes[r_j_idx]

            if route_loads[r_i_idx] + route_loads[r_j_idx] > capacity: continue
            
            i_is_first = (route_i[0] == i)
            i_is_last = (route_i[-1] == i)
            j_is_first = (route_j[0] == j)
            j_is_last = (route_j[-1] == j)

            merged_route = None

            if i_is_last and j_is_first:
                merged_route = route_i + route_j
            elif i_is_first and j_is_last:
                merged_route = route_j + route_i
            elif i_is_first and j_is_first:
                merged_route = route_i[::-1] + route_j
            elif i_is_last and j_is_last:
                merged_route = route_i + route_j[::-1]
            
            if merged_route:
                routes[r_i_idx] = merged_route
                route_loads[r_i_idx] += route_loads[r_j_idx]
                for node in route_j:
                    node_to_route_idx[node] = r_i_idx
                routes[r_j_idx] = []
                route_loads[r_j_idx] = 0

        final_routes = [r for r in routes if r]
        sol = CvrpSolution(routes=final_routes)
        self.evaluator.evaluate_objfun(sol)
        return sol

class CheapestInsertionHeuristic(BaseConstructiveHeuristic):
    """
    Cheapest Insertion.
    Start with empty routes. Iteratively insert the unvisited node that minimizes 
    cost increase across all feasible positions in all active routes.
    """

    def construct(self, randomize: bool = False) -> CvrpSolution:
        n = self.instance.dimension
        depot = self.instance.depot
        capacity = self.instance.capacity
        demands = self.instance.demands
        dist_matrix = self.instance.distance_matrix
        
        unvisited = set(i for i in range(n) if i != depot)
        routes = []
        route_loads = []

        if unvisited:
            farthest = max(unvisited, key=lambda x: dist_matrix[depot][x])
            routes.append([farthest])
            route_loads.append(demands[farthest])
            unvisited.remove(farthest)
        else:
            return CvrpSolution(routes=[])

        while unvisited:
            best_cost_increase = float('inf')
            best_node = -1
            best_route_idx = -1
            best_position = -1
            
            for node in unvisited:
                node_demand = demands[node]
                
                for r_idx, route in enumerate(routes):
                    if route_loads[r_idx] + node_demand <= capacity:
                        
                        for pos in range(len(route) + 1):
                            prev_node = route[pos-1] if pos > 0 else depot
                            next_node = route[pos] if pos < len(route) else depot
                            
                            increase = (dist_matrix[prev_node][node] + 
                                        dist_matrix[node][next_node] - 
                                        dist_matrix[prev_node][next_node])
                            
                            if increase < best_cost_increase:
                                best_cost_increase = increase
                                best_node = node
                                best_route_idx = r_idx
                                best_position = pos

            for node in unvisited:
                increase = dist_matrix[depot][node] + dist_matrix[node][depot]
                if increase < best_cost_increase:
                    best_cost_increase = increase
                    best_node = node
                    best_route_idx = -1
                    best_position = 0

            if best_node != -1:
                unvisited.remove(best_node)
                if best_route_idx == -1:
                    routes.append([best_node])
                    route_loads.append(demands[best_node])
                else:
                    routes[best_route_idx].insert(best_position, best_node)
                    route_loads[best_route_idx] += demands[best_node]
            else:
                break

        sol = CvrpSolution(routes=routes)
        self.evaluator.evaluate_objfun(sol)
        return sol

class RouteFirstClusterSecondHeuristic(BaseConstructiveHeuristic):
    """
    Implementation of Beasley's (1983) Route-first, Cluster-second method.
    Phase 1: Create a Giant Tour (TSP) visiting all nodes.
    Phase 2: Use 'Split' algorithm (Shortest Path on DAG) to partition the tour into feasible routes.
    """

    def construct(self, randomize: bool = False) -> CvrpSolution:
        giant_tour = self._generate_giant_tour()
        
        routes = self._split(giant_tour)
        
        sol = CvrpSolution(routes=routes)
        self.evaluator.evaluate_objfun(sol)
        return sol

    def _generate_giant_tour(self) -> List[int]:
        """
        Generates a TSP tour using Nearest Neighbor followed by 2-opt.
        """
        n = self.instance.dimension
        depot = self.instance.depot
        dist_matrix = self.instance.distance_matrix
        
        unvisited = set(i for i in range(n) if i != depot)
        current = depot
        tour = []
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    if j - i == 1: continue
                    
                    node_a = tour[i]
                    node_b = tour[i+1]
                    node_c = tour[j]
                    node_d = tour[j+1] if j+1 < len(tour) else tour[0]
                    
                    d_ab = dist_matrix[node_a][node_b]
                    d_cd = dist_matrix[node_c][node_d]
                    d_ac = dist_matrix[node_a][node_c]
                    d_bd = dist_matrix[node_b][node_d]
                    
                    if d_ac + d_bd < d_ab + d_cd:
                        tour[i+1:j+1] = tour[i+1:j+1][::-1]
                        improved = True
                        
        return tour

    def _split(self, giant_tour: List[int]) -> List[List[int]]:
        """
        The Split algorithm.
        Constructs an auxiliary graph where an edge (i, j) represents a valid vehicle route
        serving customers from index i to j in the giant tour.
        Finds shortest path from 0 to len(giant_tour).
        """
        depot = self.instance.depot
        capacity = self.instance.capacity
        demands = self.instance.demands
        dist_matrix = self.instance.distance_matrix
        
        n_tour = len(giant_tour)
        
        V = [float('inf')] * (n_tour + 1)
        V[0] = 0.0
        
        P = [-1] * (n_tour + 1)
        
        for i in range(n_tour):
            if V[i] == float('inf'):
                continue
                
            current_load = 0
            current_dist = 0.0
            
            for j in range(i, n_tour):
                node = giant_tour[j]
                current_load += demands[node]
                
                if current_load > capacity:
                    break
                
                if j == i:
                    current_dist = dist_matrix[depot][node]
                else:
                    prev_node = giant_tour[j-1]
                    current_dist += dist_matrix[prev_node][node]
                
                route_cost = current_dist + dist_matrix[node][depot]
                
                if V[i] + route_cost < V[j+1]:
                    V[j+1] = V[i] + route_cost
                    P[j+1] = i
                    
        routes = []
        curr = n_tour
        while curr > 0:
            prev = P[curr]
            segment = giant_tour[prev:curr]
            routes.append(segment)
            curr = prev
            
        return routes[::-1]