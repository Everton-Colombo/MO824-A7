from dataclasses import dataclass
from typing import List, Tuple
import math

@dataclass
class CvrpInstance:
    """
    Container for a CVRP instance.
    """
    name: str
    dimension: int
    capacity: int
    coordinates: List[Tuple[int, int]]
    demands: List[int]
    depot: int

    @property
    def distance_matrix(self) -> List[List[float]]:
        if not hasattr(self, '_distance_matrix'):
            self._distance_matrix = self._compute_distance_matrix()
        return self._distance_matrix

    def _compute_distance_matrix(self) -> List[List[float]]:
        n = self.dimension
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.coordinates[i]
                    x2, y2 = self.coordinates[j]
                    dist[i][j] = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return dist

    @classmethod
    def from_file(cls, filename: str) -> 'CvrpInstance':
        """
        Reads a CVRP instance from a file (TSPLIB format).
        """
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f]

        name = ""
        dimension = 0
        capacity = 0
        coordinates = []
        demands = []
        depot = 0
        
        section = ""
        
        for line in lines:
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                section = "COORD"
                continue
            elif line.startswith("DEMAND_SECTION"):
                section = "DEMAND"
                continue
            elif line.startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue
            elif line.startswith("EOF"):
                break
            
            if section == "COORD":
                parts = line.split()
                if len(parts) >= 3:
                    # Node ID is usually 1-based, but we'll store in 0-based list
                    # We assume nodes are listed in order 1..N
                    coordinates.append((int(parts[1]), int(parts[2])))
            elif section == "DEMAND":
                parts = line.split()
                if len(parts) >= 2:
                    demands.append(int(parts[1]))
            elif section == "DEPOT":
                val = int(line)
                if val != -1:
                    depot = val - 1 # Convert to 0-based index

        if len(coordinates) != dimension or len(demands) != dimension:
             # Fallback or error handling if parsing failed or format is slightly different
             pass

        return cls(
            name=name,
            dimension=dimension,
            capacity=capacity,
            coordinates=coordinates,
            demands=demands,
            depot=depot
        )
