"""
Implements the following parent classes:
    Point: A point in n-dimensional space
    PointCloud: A point cloud in n-dimensional space
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt


class Point:
    """A point in n-dimensional space

    Attributes:
        coordinates (np.array): An array of floats representing the coordinates
            of the point
        dimension (int): The number of dimensions of the point

        If the point is 1 to 3-dimensional, the x, y and z coordinates can be
        accessed directly as attributes of the Point object
    """

    def __init__(self, coordinates=np.array([0, 0, 0])) -> None:
        self.coordinates = np.array(coordinates)
        self.dimension = len(coordinates)
        if self.dimension <= 3:
            if self.dimension >= 1:
                self.x = coordinates[0]
            if self.dimension >= 2:
                self.y = coordinates[1]
            if self.dimension >= 3:
                self.z = coordinates[2]

    def __getitem__(self, key: int) -> float | int:
        """Get the coordinate at the specified index
        """
        # Typecheck key for int
        if not isinstance(key, int):
            raise TypeError("Index must be an integer")

        # Check if index is within range
        if key > self.dimension - 1:
            raise IndexError("Index out of range")

        # Return coordinate
        return self.coordinates[key]
    
    def __add__(self, other: Point) -> Point:
        """Implements dimension-wise addition of Point objects"""
        if self.dimension != other.dimension:
            raise ValueError("Points must have the same dimension")
        
        new_coordinates = np.array([
            self.coordinates[i] + other.coordinates[i]
            for i in range(self.dimension)
        ])

        return Point(new_coordinates)

    def __sub__(self, other: Point) -> Point:
        """Implements dimension-wise addition of Point objects"""
        if self.dimension != other.dimension:
            raise ValueError("Points must have the same dimension")
        
        new_coordinates = np.array([
            self.coordinates[i] - other.coordinates[i]
            for i in range(self.dimension)
        ])

        return Point(new_coordinates)
    
    def __truediv__(self, other: float | int) -> Point:
        """Implements scalar true division of Point objects"""
        if not isinstance(other, (float, int)):
            raise TypeError("Can only divide by a scalar")

        new_coordinates = np.array([
            self.coordinates[i] / other
            for i in range(self.dimension)
        ])

        return Point(new_coordinates)

    def __str__(self) -> str:
        return f"({', '.join([str(coord) for coord in self.coordinates])})"

    def __repr__(self) -> str:
        return f"Point({self.coordinates})"

    # TODO: Implement rotation methods

class Origin(Point):
    """A point at the origin of n-dimensional space"""

    def __init__(self, dimension=3) -> None:
        super().__init__(np.zeros(dimension))


class PointCloud:
    """A point cloud in n-dimensional space

    Attributes:
        points (np.array): An array of Point3D objects representing the points
            in the point cloud
        dimension (int): The number of dimensions of the point cloud
        point_count (int): The number of points in the point cloud
    """

    def __init__(
        self,
        points,
        dimension: int = 3,
        point_count: int = 21,
        normalise: bool = True
    ) -> None:

        # Ensure point coordinates are correctly formatted then convert to
        # Point objects
        reformatted_points = np.array(points).reshape(point_count, dimension)
        self.points = np.array([Point(point) for point in reformatted_points])

        self.dimension = dimension
        self.point_count = point_count

        if normalise:
            self.normalise()

    @classmethod
    def from_mphands(
        cls,
        hand_landmarks,
        dimension: int = 3,
        point_count: int = 21,
        normalise: bool = True
    ) -> None:
        """Create a PointCloud object from a mediapipe landmarks object
        """
        return cls(
            points=[(lm.x, lm.y, lm.z) for lm in hand_landmarks],
            dimension=dimension,
            point_count=point_count,
            normalise=normalise
        )

    def normalise(self) -> None:
        """Normalise the points relative to the first point in the point cloud
        and scale the points so that the furthest point is at a distance of 1
        """
        origin = self.points[0]
        max_distance = max(
            [distance(point, origin) for point in self.points]
        )
        self.points = np.array([
            (point - origin) / max_distance for point in self.points
        ])
    
    def plot(self) -> None:
        """Plot the point cloud in 3D space
        """
        if self.dimension != 3:
            raise ValueError("Can only plot 3-dimensional point clouds")
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for coord in self.points:
            x, y, z = coord
            ax.scatter(x, y, z, marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
        


def distance(self, point1: Point, point2: Point = Origin()) -> float:
        """Calculate the distance between two Point objects
        """
        return math.sqrt(sum(
            [(point1[i] - point2[i])**2 for i in range(point1.dimension)]
        ))