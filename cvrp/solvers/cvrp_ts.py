import random
import time
from collections import deque
from typing import Literal, List, Tuple
from dataclasses import dataclass

from ..cvrp_instance import CvrpInstance
from ..cvrp_evaluator import CvrpEvaluator
from ..cvrp_solution import CvrpSolution
from .abc_solver import CVRP_Solver, TerminationCriteria, DebugOptions

@dataclass
class TSStrategy:
    search_strategy: Literal['first', 'best'] = 'best'
    neighborhood: Literal['relocate'] = 'relocate' # Can extend to swap later
    
    # Diversification parameters
    enable_diversification: bool = False
    diversification_patience: int = 100
    diversification_multiplier: float = 1.5
    max_tenure_multiplier: float = 5.0
    
    # Intensification parameters
    enable_intensification: bool = False
    intensification_patience: int = 1000

class CvrpTS(CVRP_Solver):
    def __init__(self, instance: CvrpInstance, tenure: int = 7, strategy: TSStrategy = TSStrategy(),
                 termination_criteria: TerminationCriteria = TerminationCriteria(),
                 debug_options: DebugOptions = DebugOptions()):
        super().__init__(instance, termination_criteria, debug_options)
        self.strategy = strategy
        self.initial_tenure = tenure
        self.tenure = tenure
        # Tabu list stores (customer_id, iteration_expiry) or just customer_id in a deque
        # Simple implementation: deque of customer_ids that are tabu
        self.tabu_list = deque(maxlen=tenure)
        
        if debug_options.log_history:
            self.history: List[tuple] = []

    def solve(self) -> CvrpSolution:
        self._reset_execution_state()
        
        # Reset tenure and tabu list
        self.tenure = self.initial_tenure
        self.tabu_list = deque(maxlen=self.tenure)
        
        # Constructive Heuristic
        self.best_solution = self._constructive_heuristic()
        self._current_solution = self.best_solution.copy()
        
        # Evaluate initial solution
        self.evaluator.evaluate_objfun(self.best_solution)
        self.evaluator.evaluate_objfun(self._current_solution)
        
        while not self._check_termination():
            self._perform_debug_actions()
            
            # Intensification Strategy (Restart)
            if (self.strategy.enable_intensification and 
                self._iters_wo_improvement > 0 and 
                self._iters_wo_improvement % self.strategy.intensification_patience == 0):
                
                # Restart from best solution
                self._current_solution = self.best_solution.copy()
                
                # Reset tenure to initial
                self.tenure = self.initial_tenure
                self.tabu_list = deque(maxlen=self.tenure)
                
                if self.debug_options.verbose:
                    print(f"Intensification (Restart) applied at iteration {self._iters}. Resetting to best solution and initial tenure.")

            # Diversification Strategy
            elif (self.strategy.enable_diversification and 
                self._iters_wo_improvement > 0 and 
                self._iters_wo_improvement % self.strategy.diversification_patience == 0):
                
                new_tenure = int(self.tenure * self.strategy.diversification_multiplier)
                max_tenure = int(self.initial_tenure * self.strategy.max_tenure_multiplier)
                
                if new_tenure <= max_tenure:
                    self.tenure = new_tenure
                    self.tabu_list = deque(self.tabu_list, maxlen=self.tenure)
                    if self.debug_options.verbose:
                        print(f"Diversification applied at iteration {self._iters}. New tenure: {self.tenure}")

            self._current_solution = self._neighborhood_move(self._current_solution)
            
            self._update_execution_state()
            
            # Reset tenure if improvement found
            if self._iters_wo_improvement == 0 and self.tenure != self.initial_tenure:
                self.tenure = self.initial_tenure
                self.tabu_list = deque(self.tabu_list, maxlen=self.tenure)
                if self.debug_options.verbose:
                    print(f"Improvement found at iteration {self._iters}. Resetting tenure to: {self.tenure}")
            
        self.execution_time = time.time() - self._start_time
        return self.best_solution

    def _constructive_heuristic(self) -> CvrpSolution:
        """
        Simple sequential insertion heuristic.
        """
        unvisited = set(range(self.instance.dimension))
        unvisited.remove(self.instance.depot) # Depot is 0-based index of depot node? 
        # In our instance, depot is an index.
        # Customers are indices.
        
        routes = []
        current_route = []
        current_load = 0
        
        # Simple logic: append nearest neighbor to current route end
        # If full, start new route.
        
        # Actually, let's do a slightly better one: Cheapest Insertion into current route.
        # If no feasible insertion in current route, start new one.
        
        routes.append(current_route)
        
        while unvisited:
            best_cost = float('inf')
            best_node = -1
            best_pos = -1
            
            # Try to insert into current route (last one)
            route_idx = len(routes) - 1
            route = routes[route_idx]
            
            # Check all unvisited nodes
            for node in unvisited:
                demand = self.instance.demands[node]
                if current_load + demand <= self.instance.capacity:
                    # Try all positions in current route
                    # For sequential building, usually we just append or insert.
                    # Let's try all positions for better quality.
                    for pos in range(len(route) + 1):
                        # Calculate insertion cost
                        # We can use evaluator logic but we need a temporary route
                        # Or just calculate delta locally
                        
                        prev_node = route[pos-1] if pos > 0 else self.instance.depot
                        next_node = route[pos] if pos < len(route) else self.instance.depot
                        
                        cost_increase = (self.instance.distance_matrix[prev_node][node] + 
                                         self.instance.distance_matrix[node][next_node] - 
                                         self.instance.distance_matrix[prev_node][next_node])
                        
                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_node = node
                            best_pos = pos
            
            if best_node != -1:
                # Found a feasible insertion in current route
                routes[route_idx].insert(best_pos, best_node)
                current_load += self.instance.demands[best_node]
                unvisited.remove(best_node)
            else:
                # No feasible insertion in current route (capacity full)
                # Start new route
                routes.append([])
                current_load = 0
                # Loop will continue and try to insert into this new empty route
                
        return CvrpSolution(routes=routes)

    def _neighborhood_move(self, solution: CvrpSolution) -> CvrpSolution:
        # Relocate move: Move customer from (r1, p1) to (r2, p2)
        
        best_delta = float('inf')
        best_move = None # (source_r, source_p, target_r, target_p, customer)
        
        current_obj = self.evaluator.evaluate_objfun(solution)
        best_obj = self.evaluator.evaluate_objfun(self.best_solution)
        
        # Iterate over all customers (source)
        for r1_idx, r1 in enumerate(solution.routes):
            for p1_idx, customer in enumerate(r1):
                
                # Iterate over all target positions
                for r2_idx, r2 in enumerate(solution.routes):
                    # Optimization: if we have empty routes, we only need to try one empty route
                    # But for simplicity, iterate all.
                    
                    for p2_idx in range(len(r2) + 1):
                        # Skip if same position
                        if r1_idx == r2_idx:
                            if p1_idx == p2_idx or p1_idx + 1 == p2_idx:
                                continue
                        
                        # Check capacity
                        if not self.evaluator.check_capacity(solution, r1_idx, p1_idx, r2_idx):
                            continue
                            
                        # Calculate delta
                        delta = self.evaluator.evaluate_relocate_delta(solution, r1_idx, p1_idx, r2_idx, p2_idx)
                        
                        # Tabu check
                        is_tabu = customer in self.tabu_list
                        
                        # Aspiration
                        is_aspiration = (current_obj + delta < best_obj) # Minimization
                        
                        if not is_tabu or is_aspiration:
                            if delta < best_delta:
                                best_delta = delta
                                best_move = (r1_idx, p1_idx, r2_idx, p2_idx, customer)
                                
                                if self.strategy.search_strategy == 'first' and delta < -0.0001:
                                    # Found improvement
                                    return self._apply_move(solution, best_move)

        if best_move:
            return self._apply_move(solution, best_move)
        
        # If no move found (e.g. all tabu and no aspiration), we should probably just return current
        # Or pick the best tabu move (not implemented here for simplicity)
        return solution

    def _apply_move(self, solution: CvrpSolution, move) -> CvrpSolution:
        r1_idx, p1_idx, r2_idx, p2_idx, customer = move
        
        # Create new solution (deep copy of routes structure)
        # We can optimize by only copying affected routes
        new_routes = [r[:] for r in solution.routes]
        
        # Remove from source
        # Note: if r1 == r2, indices shift.
        # If r1 == r2 and p1 < p2, p2 shifts down by 1.
        
        node = new_routes[r1_idx][p1_idx]
        del new_routes[r1_idx][p1_idx]
        
        target_p = p2_idx
        if r1_idx == r2_idx and p1_idx < p2_idx:
            target_p -= 1
            
        new_routes[r2_idx].insert(target_p, node)
        
        # Remove empty routes if any?
        new_routes = [r for r in new_routes if r]
        
        # Update Tabu List
        self.tabu_list.append(customer)
        
        new_sol = CvrpSolution(routes=new_routes)
        # Update obj val
        self.evaluator.evaluate_objfun(new_sol)
        return new_sol

    def _perform_debug_actions(self):
        if self.debug_options.verbose:
            best_val = f'{self.best_solution.objfun_val:.2f}' if self.best_solution and self.best_solution.objfun_val is not None else 'N/A'
            current_val = f'{self._current_solution.objfun_val:.2f}' if self._current_solution and self._current_solution.objfun_val is not None else 'N/A'
            print(f"Iteration {self._iters}: Best Cost = {best_val}, Current Cost = {current_val}")

        if self.debug_options.log_history:
            self.history.append((
                self._iters, 
                self.best_solution.objfun_val if self.best_solution else 0,
                self._current_solution.objfun_val if self._current_solution else 0
            ))
