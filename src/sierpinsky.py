import random
import math
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    @classmethod
    def random(self, x_min: float = 0, x_max: float = 1, y_min: float = 0, y_max: float = 1):
        return Point(random.uniform(x_min, x_max),
                     random.uniform(y_min, y_max))

    @classmethod
    def random_polygon(self, corners: list['Point']):
        bounds = [min(p.x for p in corners), max(p.x for p in corners),
                  min(p.y for p in corners), max(p.y for p in corners)]
        
        while True:
            guess = Point.random(*bounds)
            
            for c in corners:
                if guess.is_inside_polygon(corners):
                    return guess
    
    # # https://wrfranklin.org/Research/Short_Notes/pnpoly.html
    # def is_inside_polygon(self, corners: list['Point']):
    #     # Currenlty there are some issues with this
    #     for c1, c2 in zip(corners[1:], corners[:-1]):
    #         if (c1.y > self.y) != (c2.y > self.y):
    #             if (self.x < (c2.x - c1.x) * (self.y - c1.y) / (c2.y - c1.y) + c1.x):
    #                 return False
    #     return True
    
    def is_inside_polygon(self, corners: list['Point']):
        corners = ccw_order(corners)
        n = len(corners)
        inside = True
        for i in range(n):
            c1 = corners[i]
            c2 = corners[(i + 1) % n]  # Next vertex (wraps around)
            # Check if the point is on the "inside" side of every edge
            cross_product = (c2.x - c1.x) * (self.y - c1.y) - (c2.y - c1.y) * (self.x - c1.x)
            if cross_product < 0:  # Assuming counter-clockwise winding
                inside = False
                break
        return inside

def ccw_order(points):
    cx = sum(p.x for p in points) / len(points)
    cy = sum(p.y for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p.y - cy, p.x - cx))
    
class SierpinskyTriangle:
    def __init__(self, corners: list[Point]):
        self.corners = ccw_order(corners.copy())
        self.points = corners.copy()
        self.start = Point((self.corners[0].x + self.corners[1].x)/2, (self.corners[0].y + self.corners[1].y)/2)

        self.last_new = [self.start]
        self.N = 0
            
    def generate_triangle(self) -> None:  
        new_points = []      
        for p1 in self.last_new:
            for p2 in self.corners:
                new_points.append(Point(
                    (p1.x + p2.x) / 2,
                    (p1.y + p2.y) / 2
                ))
        self.points += new_points
        self.N += 1
        self.last_new = new_points.copy()
        
    def plot(self, title: str) -> plt.Figure:
        fig, ax = plt.subplots()
        xx = [p.x for p in self.points]
        yy = [p.y for p in self.points]
        
        ax.scatter([p.x for p in self.corners], [p.y for p in self.corners], s=40, color='blue', marker='o', label='Corners')
        ax.scatter(self.start.x, self.start.y, s=40, color='orange', marker='o', label='Start')

        ax.scatter(xx, yy, s=1, color='black', marker='^')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.set_xlim(min(xx) - 0.1, max(xx) + 0.1)
        ax.set_ylim(min(yy) - 0.1, max(yy) + 0.1)

        ax.grid(True)
        ax.set_aspect('equal')
        return fig
            
        
        
    