from src.sierpinsky import Point
from scipy.spatial import KDTree

from numba import njit
from numba.typed import List as TypedList
import numpy as np


@njit
def _update_temperature_numba(temp, alpha, dt, nn_indices):
    new_temperature = temp.copy()
    for i in range(len(temp)):
        neighbors = nn_indices[i]
        if len(neighbors) > 0:
            sum_temp = 0.0
            for j in neighbors:
                sum_temp += temp[j]
            neighbor_temp = sum_temp / len(neighbors)
        else:
            neighbor_temp = temp[i]
        new_temperature[i] += alpha * (neighbor_temp - temp[i]) * dt
    return new_temperature

def convert_to_numba_list(pylist):
    typed_list = TypedList()
    for sub in pylist:
        sublist = TypedList()
        for val in sub:
            sublist.append(val)
        typed_list.append(sublist)
    return typed_list

class ThermalSimulation:
    def __init__(self, points: list[Point], alpha: float = 0.01, use_numba: bool = True): 
        self.tree = KDTree([(p.x, p.y) for p in points])
        self.points = points
        
        self.nn_indices = [self.tree.query_ball_point((p.x, p.y), r=0.1, workers=-1) for p in points]
        if use_numba:
            self.nn_indices = convert_to_numba_list(self.nn_indices)
        self.temperature = np.full(len(points), 298.15)
        self.alpha = alpha
        
        self.use_numba = use_numba
    
    def place_source(self, point: Point, temp: float, radius: float) -> None:
        heated_indices = self.tree.query_ball_point((point.x, point.y), r=radius, workers=-1)
        for idx in heated_indices:
            self.temperature[idx] = temp
    
    def update_temperature(self, dt: float) -> None:
        if self.use_numba:
            
            self.temperature = _update_temperature_numba(self.temperature, self.alpha, dt, self.nn_indices)
            return
        
        new_temperature = np.copy(self.temperature)
        for i, neighbors in enumerate(self.nn_indices):
            if neighbors:
                neighbor_temp = np.mean([self.temperature[j] for j in neighbors])
            else:
                neighbor_temp = self.temperature[i]
            new_temperature[i] += self.alpha * (neighbor_temp - self.temperature[i]) * dt
        self.temperature = new_temperature

    def simulate(self, t_max: float, dt: float, save_series: bool = False) -> None:
        steps = int(t_max / dt)
        if save_series:
            temps = np.zeros((steps, len(self.points)))
        for i in range(steps):
            self.update_temperature(dt)
            if save_series:
                temps[i] = self.temperature
        
        if save_series:
            return temps
        else:
            return self.temperature
