from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CvrpSolution:
    routes: List[List[int]]
    _objfun_val: Optional[float] = field(default=None, init=True, repr=False)
    
    @property
    def objfun_val(self) -> Optional[float]:
        return self._objfun_val
    
    @objfun_val.setter
    def objfun_val(self, value: Optional[float]):
        self._objfun_val = value

    def __str__(self):
        obj_val = self.objfun_val
        val_str = f"{obj_val:.2f}" if obj_val is not None else "None"
        return f"CvrpSolution(cost={val_str}, num_routes={len(self.routes)})"
    
    def copy(self) -> 'CvrpSolution':
        new_routes = [route[:] for route in self.routes]
        return CvrpSolution(routes=new_routes, _objfun_val=self._objfun_val)
