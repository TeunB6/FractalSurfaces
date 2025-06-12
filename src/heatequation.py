from src.sierpinsky import Point
from scipy.spatial import KDTree

from numba import njit
from numba.typed import List as TypedList
from tqdm import tqdm
import numpy as np


UREF = 298.15  # Reference temperature in Kelvin (25 degrees Celsius)

@njit
def _update_temperature_numba(temp: np.ndarray, alpha: float, K_rad: float,
                              dt: float, nn_indices: TypedList, source: TypedList) -> np.ndarray:
    """temperature update function using numba for performance.

    Args:
        temp (np.ndarray): the temperature of each point
        alpha (float): heating coefficient
        dt (float): time step
        nn_indices (TypedList): list of lists containing indices of neighboring points
        source (TypedList): list of source temperatures
        is_cooling (bool, optional): Include radiative cooling. Defaults to False.

    Returns:
        TypedList: updated temperature of each point
    """    
    new_temperature = temp.copy()
    for i in range(len(temp)):
        
        # handle sources
        source_temp = source[i]
        if source_temp != 0:
            new_temperature[i] = source_temp
            continue
        
        # handle neighbors (heating)
        neighbors = nn_indices[i]
        if len(neighbors) > 0:
            sum_temp = 0.0
            for j in neighbors:
                sum_temp += temp[j]
            neighbor_temp = sum_temp / len(neighbors)
        else:
            neighbor_temp = temp[i]
        new_temperature[i] += alpha * (neighbor_temp - temp[i]) * dt
        
        # radiative cooling
        if K_rad is not None:
            new_temperature[i] -= K_rad * (temp[i]**4 - UREF**4) * dt 
    return new_temperature

def convert_to_numba_list(pylist: list) -> TypedList:
    """Convert a nested Python list to a Numba typed list for performance."""
    typed_list = TypedList()
    for sub in pylist:
        if isinstance(sub, list):
            sublist = TypedList()
            for val in sub:
                sublist.append(val)
            typed_list.append(sublist)
        else:
            typed_list.append(sub)
    return typed_list


class ThermalSimulation:
    def __init__(self, points: list[Point], alpha: float = 0.01, K_rad: float = None,
                 use_numba: bool = True) -> None: 
        self.tree = KDTree([(p.x, p.y) for p in points])
        self.points = points
        
        self.nn_indices = [self.tree.query_ball_point((p.x, p.y), r=0.1, workers=-1) for p in points]
        if use_numba:
            self.nn_indices = convert_to_numba_list(self.nn_indices)
        self.temperature = np.full(len(points), UREF)
        self.alpha = alpha
        
        self.K_rad = K_rad
        
        self.use_numba = use_numba
        
        self.source = [0] * len(self.points)
        if use_numba:
            self.source_numba = None
    
    def place_source(self, point: Point, temp: float, radius: float) -> None:
        """Place a heat source at a given point with a specified temperature and radius.

        Args:
            point (Point): the location of the heat source
            temp (float): the temperature of the heat source
            radius (float): the radius of the heat source's influence
        """        
        heated_indices = self.tree.query_ball_point((point.x, point.y), r=radius, workers=-1)
        for idx in heated_indices:
            self.source[idx] += temp
            self.temperature[idx] += temp
        
        if self.use_numba:
            self.source_numba = convert_to_numba_list(self.source)
    
    def update_temperature(self, dt: float) -> None:
        """Update the temperature of each point based on neighboring points and sources."""	
        if self.use_numba:
            self.temperature = _update_temperature_numba(self.temperature, self.alpha, self.K_rad,
                                                         dt, self.nn_indices, self.source_numba)
            return
        
        new_temperature = np.copy(self.temperature)
        for i, neighbors in enumerate(self.nn_indices):
            if neighbors:
                sum_temp = 0.0
                for j in neighbors:
                    sum_temp += self.temperature[j]
                neighbor_temp = sum_temp / len(neighbors)
            else:
                neighbor_temp = self.temperature[i]
            new_temperature[i] += self.alpha * (neighbor_temp - self.temperature[i]) * dt
        self.temperature = new_temperature

    def simulate(self, t_max: float, dt: float, save_series: bool = False) -> np.ndarray:
        """simulate the thermal diffusion process over a specified time period.

        Args:
            t_max (float): time to simulate up to
            dt (float): time step
            save_series (bool, optional): return the temperatures or not. Defaults to False.

        Returns:
            np.ndarray: the temperature of each point at each time step if save_series is True, otherwise the final temperature.
        """        
        steps = int(t_max / dt)
        if save_series:
            temps = np.zeros((steps, len(self.points)))
        for i in tqdm(range(steps), desc="Simulation Steps", total=steps, leave=False):
            self.update_temperature(dt)
            if save_series:
                temps[i] = self.temperature
        
        if save_series:
            return temps
        else:
            return self.temperature
