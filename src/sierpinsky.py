import random

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __truediv__(self, scalar: float):
        return Point(self.x / scalar, self.y / scalar)
    
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
    
    # https://wrfranklin.org/Research/Short_Notes/pnpoly.html
    def is_inside_polygon(self, corners: list['Point']):
        for c1, c2 in zip(corners, reversed(corners)):
            if (c1.y > self.y) != (c2.y > self.y):
                if (self.x < (c2.x - c1.x) * (self.y - c1.y) / (c2.y - c1.y) + c1.x):
                    return False
        return True
    
class SierpinskyTriangle:
    def __init__(self, corners: list[Point]):
        self.corners = corners.copy()
        self.points = corners.copy()
        self.last_new = [Point.random_polygon(self.corners)]
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
        self.last_new = new_points
    