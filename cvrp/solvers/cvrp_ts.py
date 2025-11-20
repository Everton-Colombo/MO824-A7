import random
import time
from collections import deque
from typing import Literal, List, Tuple, Dict, Any
from dataclasses import dataclass, field

from ..cvrp_instance import CvrpInstance
from ..cvrp_evaluator import CvrpEvaluator
from ..cvrp_solution import CvrpSolution
from .abc_solver import CVRP_Solver, TerminationCriteria, DebugOptions

@dataclass
class TSStrategy:
    search_strategy: Literal['first', 'best'] = 'best'
    neighborhoods: Dict[str, bool] = field(default_factory=lambda: {'relocate': True, 'swap': True})
    
    # Diversification parameters
    enable_diversification: bool = False
    diversification_patience: int = 100
    diversification_multiplier: float = 1.5
    max_tenure_multiplier: float = 5.0
    restart_when_max_reached: bool = True
    
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
                elif self.strategy.restart_when_max_reached:
                    self._current_solution = self._constructive_heuristic()
                    self.evaluator.evaluate_objfun(self._current_solution)
                    self.tenure = self.initial_tenure
                    self.tabu_list = deque(maxlen=self.tenure)
                    if self.debug_options.verbose:
                        print(f"Diversification max tenure reached at iteration {self._iters}. Restarting from constructive heuristic.")

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
        # Track best admissible move (non-tabu or aspiration)
        best_admissible_delta = float('inf')
        best_admissible_move = None 
        best_admissible_type = None
        
        # Track best tabu move (fallback if no admissible move exists)
        best_tabu_delta = float('inf')
        best_tabu_move = None
        best_tabu_type = None
        
        current_obj = self.evaluator.evaluate_objfun(solution)
        best_obj = self.evaluator.evaluate_objfun(self.best_solution)
        
        # Determine order of neighborhoods to explore
        neighborhoods_to_explore = []
        if self.strategy.neighborhoods.get('relocate', False):
            neighborhoods_to_explore.append('relocate')
        if self.strategy.neighborhoods.get('swap', False):
            neighborhoods_to_explore.append('swap')
            
        if self.strategy.search_strategy == 'first':
            random.shuffle(neighborhoods_to_explore)
            
        for neighborhood in neighborhoods_to_explore:
            if neighborhood == 'relocate':
                adm_delta, adm_move, tabu_delta, tabu_move = self._explore_relocate(solution, current_obj, best_obj)
                
                # Update best admissible
                if adm_move is not None:
                    if adm_delta < best_admissible_delta:
                        best_admissible_delta = adm_delta
                        best_admissible_move = adm_move
                        best_admissible_type = 'relocate'
                        
                        if self.strategy.search_strategy == 'first' and adm_delta < -0.0001:
                            return self._apply_move(solution, best_admissible_move, 'relocate')
                
                # Update best tabu
                if tabu_move is not None:
                    if tabu_delta < best_tabu_delta:
                        best_tabu_delta = tabu_delta
                        best_tabu_move = tabu_move
                        best_tabu_type = 'relocate'

            elif neighborhood == 'swap':
                adm_delta, adm_move, tabu_delta, tabu_move = self._explore_swap(solution, current_obj, best_obj)
                
                # Update best admissible
                if adm_move is not None:
                    if adm_delta < best_admissible_delta:
                        best_admissible_delta = adm_delta
                        best_admissible_move = adm_move
                        best_admissible_type = 'swap'
                        
                        if self.strategy.search_strategy == 'first' and adm_delta < -0.0001:
                            return self._apply_move(solution, best_admissible_move, 'swap')
                
                # Update best tabu
                if tabu_move is not None:
                    if tabu_delta < best_tabu_delta:
                        best_tabu_delta = tabu_delta
                        best_tabu_move = tabu_move
                        best_tabu_type = 'swap'

        # Prefer admissible move
        if best_admissible_move:
            return self._apply_move(solution, best_admissible_move, best_admissible_type)
        
        # Fallback to tabu move if no admissible move found (prevent getting stuck)
        if best_tabu_move:
            return self._apply_move(solution, best_tabu_move, best_tabu_type)
        
        if self.debug_options.verbose:
            print(f"Warning: No valid moves found at iteration {self._iters}!")
            
        return solution

    def _explore_relocate(self, solution: CvrpSolution, current_obj: float, best_obj: float) -> Tuple[float, Any, float, Any]:
        best_admissible_delta = float('inf')
        best_admissible_move = None
        
        best_tabu_delta = float('inf')
        best_tabu_move = None
        
        for r1_idx, r1 in enumerate(solution.routes):
            for p1_idx, customer in enumerate(r1):
                for r2_idx, r2 in enumerate(solution.routes):
                    for p2_idx in range(len(r2) + 1):
                        if r1_idx == r2_idx:
                            if p1_idx == p2_idx or p1_idx + 1 == p2_idx:
                                continue
                        
                        if not self.evaluator.check_capacity(solution, r1_idx, p1_idx, r2_idx):
                            continue
                            
                        delta = self.evaluator.evaluate_relocate_delta(solution, r1_idx, p1_idx, r2_idx, p2_idx)
                        
                        is_tabu = customer in self.tabu_list
                        is_aspiration = (current_obj + delta < best_obj)
                        
                        if not is_tabu or is_aspiration:
                            if delta < best_admissible_delta:
                                best_admissible_delta = delta
                                best_admissible_move = (r1_idx, p1_idx, r2_idx, p2_idx, customer)
                                
                                if self.strategy.search_strategy == 'first' and delta < -0.0001:
                                    return best_admissible_delta, best_admissible_move, float('inf'), None
                        else:
                            if delta < best_tabu_delta:
                                best_tabu_delta = delta
                                best_tabu_move = (r1_idx, p1_idx, r2_idx, p2_idx, customer)
                                
        return best_admissible_delta, best_admissible_move, best_tabu_delta, best_tabu_move

    def _explore_swap(self, solution: CvrpSolution, current_obj: float, best_obj: float) -> Tuple[float, Any, float, Any]:
        best_admissible_delta = float('inf')
        best_admissible_move = None
        
        best_tabu_delta = float('inf')
        best_tabu_move = None
        
        # Iterate all pairs of customers
        # To avoid duplicates (A,B) vs (B,A), we can enforce ordering or just iterate carefully
        # We iterate through routes and positions
        
        routes = solution.routes
        for r1_idx, r1 in enumerate(routes):
            for p1_idx, customer1 in enumerate(r1):
                
                # Start inner loop from current position to avoid duplicates and self-swap
                # If same route, start from p1_idx + 1
                # If next routes, start from 0
                
                start_r2 = r1_idx
                
                for r2_idx in range(start_r2, len(routes)):
                    r2 = routes[r2_idx]
                    start_p2 = p1_idx + 1 if r1_idx == r2_idx else 0
                    
                    for p2_idx in range(start_p2, len(r2)):
                        customer2 = r2[p2_idx]
                        
                        if not self.evaluator.check_swap_capacity(solution, r1_idx, p1_idx, r2_idx, p2_idx):
                            continue
                            
                        delta = self.evaluator.evaluate_swap_delta(solution, r1_idx, p1_idx, r2_idx, p2_idx)
                        
                        # Tabu check: if EITHER is tabu, move is tabu
                        is_tabu = (customer1 in self.tabu_list) or (customer2 in self.tabu_list)
                        is_aspiration = (current_obj + delta < best_obj)
                        
                        if not is_tabu or is_aspiration:
                            if delta < best_admissible_delta:
                                best_admissible_delta = delta
                                best_admissible_move = (r1_idx, p1_idx, r2_idx, p2_idx, customer1, customer2)
                                
                                if self.strategy.search_strategy == 'first' and delta < -0.0001:
                                    return best_admissible_delta, best_admissible_move, float('inf'), None
                        else:
                            if delta < best_tabu_delta:
                                best_tabu_delta = delta
                                best_tabu_move = (r1_idx, p1_idx, r2_idx, p2_idx, customer1, customer2)
                                    
        return best_admissible_delta, best_admissible_move, best_tabu_delta, best_tabu_move

    def _apply_move(self, solution: CvrpSolution, move, move_type: str) -> CvrpSolution:
        if move_type == 'relocate':
            r1_idx, p1_idx, r2_idx, p2_idx, customer = move
            
            new_routes = [r[:] for r in solution.routes]
            
            node = new_routes[r1_idx][p1_idx]
            del new_routes[r1_idx][p1_idx]
            
            target_p = p2_idx
            if r1_idx == r2_idx and p1_idx < p2_idx:
                target_p -= 1
                
            new_routes[r2_idx].insert(target_p, node)
            new_routes = [r for r in new_routes if r]
            
            self.tabu_list.append(customer)
            
            new_sol = CvrpSolution(routes=new_routes)
            self.evaluator.evaluate_objfun(new_sol)
            return new_sol
            
        elif move_type == 'swap':
            r1_idx, p1_idx, r2_idx, p2_idx, c1, c2 = move
            
            new_routes = [r[:] for r in solution.routes]
            
            # Swap
            new_routes[r1_idx][p1_idx] = c2
            new_routes[r2_idx][p2_idx] = c1
            
            # Add both to tabu list
            self.tabu_list.append(c1)
            self.tabu_list.append(c2)
            
            new_sol = CvrpSolution(routes=new_routes)
            self.evaluator.evaluate_objfun(new_sol)
            return new_sol
            
        return solution

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
