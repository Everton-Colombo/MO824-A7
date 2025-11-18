from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Literal

from ..cvrp_instance import CvrpInstance
from ..cvrp_evaluator import CvrpEvaluator
from ..cvrp_solution import CvrpSolution

@dataclass
class TerminationCriteria:
    max_iterations: int = None
    max_no_improvement: int = None
    max_time_secs: float = None
    target_value: float = None 

@dataclass
class DebugOptions:
    verbose: bool = False
    log_history: bool = False

class CVRP_Solver(ABC):
    def __init__(self, instance: CvrpInstance, termination_criteria: TerminationCriteria, debug_options: DebugOptions = DebugOptions()):
        self.instance = instance
        self.evaluator = CvrpEvaluator(instance)
        
        self.termination_criteria = termination_criteria
        self.debug_options = debug_options

        # Execution state properties
        self._iters = 0
        self._iters_wo_improvement = 0
        self._start_time = None
        self.execution_time = 0
        self.stop_reason: Literal["max_iterations", "max_no_improvement", "max_time_secs", "target_value"] = None
        
        self._current_solution: CvrpSolution = None
        self.best_solution: CvrpSolution = None
    
    @abstractmethod
    def solve(self) -> CvrpSolution:
        pass
    
    def _reset_execution_state(self):
        self._iters = 0
        self._iters_wo_improvement = 0
        self._start_time = time.time()
        self.execution_time = 0
        self.stop_reason = None
        self._current_solution = None
        self.best_solution = None
    
    def _check_termination(self) -> bool:
        """Check if any termination criteria is met."""
        
        if self.termination_criteria.max_iterations is not None and self._iters >= self.termination_criteria.max_iterations:
            self.stop_reason = "max_iterations"
            return True
        if self.termination_criteria.max_no_improvement is not None and self._iters_wo_improvement >= self.termination_criteria.max_no_improvement:
            self.stop_reason = "max_no_improvement"
            return True
        if self.termination_criteria.max_time_secs is not None and (time.time() - self._start_time) >= self.termination_criteria.max_time_secs:
            self.stop_reason = "max_time_secs"
            return True
        if self.termination_criteria.target_value is not None and self.best_solution is not None and \
                self.evaluator.evaluate_objfun(self.best_solution) <= self.termination_criteria.target_value: # Note: <= for minimization
            self.stop_reason = "target_value"
            return True
        return False
    
    def _update_execution_state(self):
        """Update iteration counters and best solution found."""
        
        self._iters += 1
        self.execution_time = time.time() - self._start_time
        
        current_val = self.evaluator.evaluate_objfun(self._current_solution)
        best_val = self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else float('inf')

        if self.best_solution is None or current_val < best_val: # Minimization
            self.best_solution = self._current_solution.copy() # Make a copy!
            self._iters_wo_improvement = 0
        else:
            self._iters_wo_improvement += 1
    
    @abstractmethod
    def _perform_debug_actions(self):
        """Perform any debug actions based on debug options."""
        pass
