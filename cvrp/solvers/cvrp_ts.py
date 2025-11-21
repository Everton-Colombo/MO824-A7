import random
import time
from collections import deque
from typing import Literal, List, Tuple, Dict, Any
from dataclasses import dataclass, field

from ..cvrp_instance import CvrpInstance
from ..cvrp_evaluator import CvrpEvaluator
from ..cvrp_solution import CvrpSolution
from .abc_solver import CVRP_Solver, TerminationCriteria, DebugOptions
from .constructive_heuristics.base_heuristics import (
    BaseConstructiveHeuristic, 
    ConstructiveHeuristicType,
    SavingsHeuristic, 
    CheapestInsertionHeuristic, 
    RouteFirstClusterSecondHeuristic
)

@dataclass
class TSStrategy:
    constructive_heuristic: ConstructiveHeuristicType = ConstructiveHeuristicType.SAVINGS
    search_strategy: Literal['first', 'best'] = 'best'
    
    neighborhoods: Dict[str, bool] = field(default_factory=lambda: {
        'relocate': True, 
        'swap': True, 
        '2opt': True,
        'oropt': True,
        'cross_exchange': True
    })
    
    enable_diversification: bool = False
    diversification_patience: int = 100
    diversification_multiplier: float = 1.5
    max_tenure_multiplier: float = 5.0
    restart_when_max_reached: bool = True
    
    enable_intensification: bool = False
    intensification_patience: int = 1000

class CvrpTS(CVRP_Solver):
    def __init__(self, instance: CvrpInstance, tenure: int = 7, 
                 strategy: TSStrategy = TSStrategy(),
                 termination_criteria: TerminationCriteria = TerminationCriteria(),
                 debug_options: DebugOptions = DebugOptions()):
        
        super().__init__(instance, termination_criteria, debug_options)
        self.strategy = strategy
        self.initial_tenure = tenure
        self.tenure = tenure
        self.tabu_list = deque(maxlen=tenure)
        
        if debug_options.log_history:
            self.history: List[tuple] = []

    def solve(self) -> CvrpSolution:
        self._reset_execution_state()
        
        self.tenure = self.initial_tenure
        self.tabu_list = deque(maxlen=self.tenure)
        
        if self.debug_options.verbose:
            print(f"Generating initial solution using {self.strategy.constructive_heuristic.value}...")
            
        self.best_solution = self._constructive_heuristic(self.strategy.constructive_heuristic)
        self._current_solution = self.best_solution.copy()
        
        self.evaluator.evaluate_objfun(self.best_solution)
        self.evaluator.evaluate_objfun(self._current_solution)
        
        while not self._check_termination():
            self._perform_debug_actions()
            
            if (self.strategy.enable_intensification and 
                self._iters_wo_improvement > 0 and 
                self._iters_wo_improvement % self.strategy.intensification_patience == 0):
                
                self._current_solution = self.best_solution.copy()
                self.tenure = self.initial_tenure
                self.tabu_list = deque(maxlen=self.tenure)
                if self.debug_options.verbose:
                    print(f"Intensification (Restart) applied at iteration {self._iters}.")

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
                    if self.debug_options.verbose:
                        print(f"Max tenure reached. Applying PERTURBATION (Shake) on Best Solution...")
                    
                    self._current_solution = self._perturb_solution(self.best_solution, strength=40)
                    
                    self.tenure = self.initial_tenure
                    self.tabu_list = deque(maxlen=self.tenure)

            self._current_solution = self._neighborhood_move(self._current_solution)
            
            self._update_execution_state()
            
            if self._iters_wo_improvement == 0 and self.tenure != self.initial_tenure:
                self.tenure = self.initial_tenure
                self.tabu_list = deque(self.tabu_list, maxlen=self.tenure)
                if self.debug_options.verbose:
                    print(f"Improvement found. Resetting tenure.")
            
        self.execution_time = time.time() - self._start_time
        return self.best_solution

    def _get_heuristic_builder(self, instance, evaluator, strategy: ConstructiveHeuristicType) -> BaseConstructiveHeuristic:
        mapping = {
            ConstructiveHeuristicType.SAVINGS: SavingsHeuristic,
            ConstructiveHeuristicType.CHEAPEST_INSERTION: CheapestInsertionHeuristic,
            ConstructiveHeuristicType.ROUTE_FIRST_CLUSTER_SECOND: RouteFirstClusterSecondHeuristic,
        }
        HeuristicClass = mapping.get(strategy)
        if HeuristicClass is None: raise ValueError(f"Unknown: {strategy}")
        return HeuristicClass(instance, evaluator)

    def _constructive_heuristic(self, strategy_type: ConstructiveHeuristicType, randomize: bool = False) -> CvrpSolution:
        builder = self._get_heuristic_builder(self.instance, self.evaluator, strategy_type)
        return builder.construct(randomize=randomize)

    def _explore_neighborhood(self, solution: CvrpSolution, current_obj: float, best_obj: float, is_first: bool) -> Tuple[float, Any, str]:
        best_delta = float('inf')
        best_move = None
        best_move_type = None

        routes = solution.routes
        route_indices = list(range(len(routes)))
        random.shuffle(route_indices)
        
        neighborhoods_to_check = []
        if self.strategy.neighborhoods.get('relocate', False): neighborhoods_to_check.append('relocate')
        if self.strategy.neighborhoods.get('swap', False): neighborhoods_to_check.append('swap')
        if self.strategy.neighborhoods.get('2opt', False): neighborhoods_to_check.append('2opt')
        if self.strategy.neighborhoods.get('oropt', False): neighborhoods_to_check.append('oropt')
        if self.strategy.neighborhoods.get('cross_exchange', False): neighborhoods_to_check.append('cross')

        if is_first:
            random.shuffle(neighborhoods_to_check)

        for neighborhood in neighborhoods_to_check:
            
            if neighborhood == 'cross':
                max_len = 2 
                
                for r1_idx in route_indices:
                    r1 = routes[r1_idx]
                    for r2_idx in route_indices:
                        if r1_idx >= r2_idx: continue
                        
                        r2 = routes[r2_idx]
                        
                        for l1 in range(1, max_len + 1):
                            if l1 > len(r1): break
                            
                            for l2 in range(1, max_len + 1):
                                if l2 > len(r2): break
                                
                                p1_indices = list(range(len(r1) - l1 + 1))
                                random.shuffle(p1_indices)
                                
                                for p1_idx in p1_indices:
                                    p2_indices = list(range(len(r2) - l2 + 1))
                                    random.shuffle(p2_indices)
                                    
                                    for p2_idx in p2_indices:
                                        
                                        if not self.evaluator.check_cross_exchange_capacity(solution, r1_idx, p1_idx, l1, r2_idx, p2_idx, l2):
                                            continue
                                            
                                        delta = self.evaluator.evaluate_cross_exchange_delta(solution, r1_idx, p1_idx, l1, r2_idx, p2_idx, l2)
                                        
                                        n1 = r1[p1_idx]
                                        n2 = r2[p2_idx]
                                        is_tabu = (n1 in self.tabu_list) or (n2 in self.tabu_list)
                                        is_aspiration = (current_obj + delta < best_obj - 1e-6)
                                        
                                        if not is_tabu or is_aspiration:
                                            move_data = (r1_idx, p1_idx, l1, r2_idx, p2_idx, l2)
                                            
                                            if is_first and delta < -1e-6:
                                                return delta, move_data, 'cross'
                                            if delta < best_delta:
                                                best_delta = delta
                                                best_move = move_data
                                                best_move_type = 'cross'
                continue

            if neighborhood == 'oropt':
                chain_lengths = [2, 3]
                random.shuffle(chain_lengths)
                
                for k in chain_lengths:
                    for r1_idx in route_indices:
                        r1 = routes[r1_idx]
                        if len(r1) < k: continue
                        
                        p1_indices = list(range(len(r1) - k + 1))
                        random.shuffle(p1_indices)
                        
                        for p1_idx in p1_indices:
                            chain_nodes = r1[p1_idx : p1_idx + k]
                            
                            for r2_idx in route_indices:
                                r2 = routes[r2_idx]
                                
                                start_p2 = 0
                                
                                p2_indices = list(range(len(r2) + 1))
                                random.shuffle(p2_indices)
                                
                                for p2_idx in p2_indices:
                                    if r1_idx == r2_idx:
                                        if p2_idx >= p1_idx and p2_idx <= p1_idx + k:
                                            continue
                                    
                                    if not self.evaluator.check_oropt_capacity(solution, r1_idx, p1_idx, k, r2_idx):
                                        continue
                                    
                                    delta = self.evaluator.evaluate_oropt_delta(solution, r1_idx, p1_idx, k, r2_idx, p2_idx)
                                    
                                    is_tabu = any(node in self.tabu_list for node in chain_nodes)
                                    is_aspiration = (current_obj + delta < best_obj - 1e-6)
                                    
                                    if not is_tabu or is_aspiration:
                                        move_data = (r1_idx, p1_idx, r2_idx, p2_idx, k, chain_nodes)
                                        
                                        if is_first and delta < -1e-6:
                                            return delta, move_data, 'oropt'
                                        
                                        if delta < best_delta:
                                            best_delta = delta
                                            best_move = move_data
                                            best_move_type = 'oropt'
                continue

            for r1_idx in route_indices:
                r1 = routes[r1_idx]
                p1_indices = list(range(len(r1)))
                random.shuffle(p1_indices)

                for p1_idx in p1_indices:
                    customer1 = r1[p1_idx]

                    for r2_idx in route_indices:
                        r2 = routes[r2_idx]
                        start_p2 = p1_idx + 1 if r1_idx == r2_idx else 0
                        p2_limit = len(r2) if neighborhood != 'relocate' else len(r2) + 1
                        p2_indices = list(range(start_p2, p2_limit))
                        random.shuffle(p2_indices)

                        for p2_idx in p2_indices:
                            move_data = None
                            delta = float('inf')

                            if neighborhood == 'relocate':
                                if r1_idx == r2_idx and (p1_idx == p2_idx or p1_idx + 1 == p2_idx): continue
                                if not self.evaluator.check_capacity(solution, r1_idx, p1_idx, r2_idx): continue
                                delta = self.evaluator.evaluate_relocate_delta(solution, r1_idx, p1_idx, r2_idx, p2_idx)
                                move_data = (r1_idx, p1_idx, r2_idx, p2_idx, customer1)
                                
                            elif neighborhood == 'swap':
                                if p2_idx >= len(r2): continue
                                customer2 = r2[p2_idx]
                                if not self.evaluator.check_swap_capacity(solution, r1_idx, p1_idx, r2_idx, p2_idx): continue
                                delta = self.evaluator.evaluate_swap_delta(solution, r1_idx, p1_idx, r2_idx, p2_idx)
                                move_data = (r1_idx, p1_idx, r2_idx, p2_idx, customer1, customer2)

                            elif neighborhood == '2opt':
                                if r1_idx != r2_idx or len(r1) < 4: continue
                                if p1_idx >= p2_idx - 1: continue 
                                delta = self.evaluator.evaluate_2opt_delta(solution, r1_idx, p1_idx, p2_idx)
                                move_data = (r1_idx, p1_idx, p2_idx)
                                
                            if move_data is None: continue
                            
                            is_tabu = customer1 in self.tabu_list
                            is_aspiration = (current_obj + delta < best_obj - 1e-6)

                            if not is_tabu or is_aspiration:
                                if is_first and delta < -1e-6:
                                    return delta, move_data, neighborhood
                                if delta < best_delta:
                                    best_delta = delta
                                    best_move = move_data
                                    best_move_type = neighborhood

        return best_delta, best_move, best_move_type

    def _neighborhood_move(self, solution: CvrpSolution) -> CvrpSolution:
        is_first = self.strategy.search_strategy == 'first'
        current_obj = self.evaluator.evaluate_objfun(solution)
        best_obj = self.best_solution.objfun_val if self.best_solution else current_obj

        best_delta, best_move, best_move_type = self._explore_neighborhood(
            solution, current_obj, best_obj, is_first
        )

        if best_move is not None:
            return self._apply_move(solution, best_move, best_move_type)
        
        if self.debug_options.verbose:
            print(f"Warning: No valid moves found at iteration {self._iters}!")
            
        return solution
    
    def _apply_move(self, solution: CvrpSolution, move: tuple, move_type: str) -> CvrpSolution:
        new_routes = [r[:] for r in solution.routes]
        
        if move_type == 'relocate':
            r1, p1, r2, p2, customer = move
            del new_routes[r1][p1]
            target_p = p2
            if r1 == r2 and p1 < p2: target_p -= 1
            new_routes[r2].insert(target_p, customer)
            self.tabu_list.append(customer)
            
        elif move_type == 'swap':
            r1, p1, r2, p2, c1, c2 = move
            new_routes[r1][p1] = c2
            new_routes[r2][p2] = c1
            self.tabu_list.append(c1)
            self.tabu_list.append(c2)
            
        elif move_type == '2opt':
            r_idx, i, j = move
            route = new_routes[r_idx]
            new_routes[r_idx][i:j+1] = route[i:j+1][::-1]
            self.tabu_list.append(route[i])
            self.tabu_list.append(route[j])

        elif move_type == 'oropt':
            r1, p1, r2, p2, k, chain_nodes = move
            
            del new_routes[r1][p1 : p1 + k]
            
            target_p = p2
            if r1 == r2 and p1 < p2:
                target_p -= k
            
            for node in reversed(chain_nodes):
                new_routes[r2].insert(target_p, node)
                
            for node in chain_nodes:
                self.tabu_list.append(node)
        elif move_type == 'cross':
            r1_idx, p1_idx, len1, r2_idx, p2_idx, len2 = move
            
            seg1 = new_routes[r1_idx][p1_idx : p1_idx + len1]
            seg2 = new_routes[r2_idx][p2_idx : p2_idx + len2]
            
            del new_routes[r1_idx][p1_idx : p1_idx + len1]
            del new_routes[r2_idx][p2_idx : p2_idx + len2]
            
            new_routes[r1_idx][p1_idx:p1_idx] = seg2
            new_routes[r2_idx][p2_idx:p2_idx] = seg1
            
            if seg1: self.tabu_list.append(seg1[0])
            if seg2: self.tabu_list.append(seg2[0])

        new_routes = [r for r in new_routes if r]
        new_sol = CvrpSolution(routes=new_routes)
        self.evaluator.evaluate_objfun(new_sol)
        return new_sol

    def _perturb_solution(self, solution: CvrpSolution, strength: int = 20) -> CvrpSolution:
        """
        Aplica uma série de movimentos aleatórios (Swap/Relocate) para escapar de ótimos locais.
        strength: número de movimentos aleatórios a aplicar.
        """
        current = solution.copy()
        routes_indices = list(range(len(current.routes)))
        
        moves_done = 0
        attempts = 0
        
        while moves_done < strength and attempts < strength * 5:
            attempts += 1
            
            move_type = random.choice(['relocate', 'swap'])
            
            r1_idx = random.choice(routes_indices)
            r2_idx = random.choice(routes_indices)
            
            if not current.routes[r1_idx]: continue
            if move_type == 'swap' and not current.routes[r2_idx]: continue
            
            p1_idx = random.randint(0, len(current.routes[r1_idx]) - 1)
            
            if move_type == 'relocate':
                
                if r1_idx == r2_idx and (p1_idx == p2_idx or p1_idx + 1 == p2_idx): continue
                if not self.evaluator.check_capacity(current, r1_idx, p1_idx, r2_idx): continue
                
                node = current.routes[r1_idx][p1_idx]
                del current.routes[r1_idx][p1_idx]
                target_p = p2_idx
                if r1_idx == r2_idx and p1_idx < p2_idx: target_p -= 1
                current.routes[r2_idx].insert(target_p, node)
                moves_done += 1
                
            elif move_type == 'swap':
                if not current.routes[r2_idx]: continue
                p2_idx = random.randint(0, len(current.routes[r2_idx]) - 1)
                
                if r1_idx == r2_idx and p1_idx == p2_idx: continue
                if not self.evaluator.check_swap_capacity(current, r1_idx, p1_idx, r2_idx, p2_idx): continue
                
                n1 = current.routes[r1_idx][p1_idx]
                n2 = current.routes[r2_idx][p2_idx]
                current.routes[r1_idx][p1_idx] = n2
                current.routes[r2_idx][p2_idx] = n1
                moves_done += 1

        current.routes = [r for r in current.routes if r]
        self.evaluator.evaluate_objfun(current)
        
        if self.debug_options.verbose:
            print(f"Perturbation applied: {moves_done} moves. New Cost: {current.objfun_val:.2f}")
            
        return current

    def _perform_debug_actions(self):
        current_time = time.time() - self._start_time

        if self.debug_options.verbose:
            best_val = f'{self.best_solution.objfun_val:.2f}' if self.best_solution and self.best_solution.objfun_val is not None else 'N/A'
            current_val = f'{self._current_solution.objfun_val:.2f}' if self._current_solution and self._current_solution.objfun_val is not None else 'N/A'
            print(f"Iteration {self._iters}: Best Cost = {best_val}, Current Cost = {current_val}")

        if self.debug_options.log_history:
            self.history.append((
                self._iters, 
                self.best_solution.objfun_val if self.best_solution else 0,
                self._current_solution.objfun_val if self._current_solution else 0,
                current_time
            ))