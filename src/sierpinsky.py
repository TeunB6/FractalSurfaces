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
    def random(self, min: float = -5, max: float = 5):
        return Point(random.uniform(min, max), random.uniform(min, max))

class SierpinskyTriangle:
    def __init__(self, a: tuple, b: tuple, c: tuple, max_iter: int = 1000):
        self.a = a
        self.b = b
        self.c = c
        self.max_iter = max_iter
    
    def __iter__(self):
        start = Point.random()
        for _ in range(self.max_iter):
            r = random.random()
            if r < 1/3:
                start = Point((start.x + self.a[0]) / 2, (start.y + self.a[1]) / 2)
            elif r < 2/3:
                start = Point((start.x + self.b[0]) / 2, (start.y + self.b[1]) / 2)
            else:
                start = Point((start.x + self.c[0]) / 2, (start.y + self.c[1]) / 2)
            yield start