# box-iteration
This project aims to solve the problem that rectangles do not intersect in a certain space.

First, the project simulates the attributes in the space and each rectangle, and then iterates according to the similarity of attributes.

The principle of guiding iteration is that the attributes of the center point of the rectangle are similar to those of the discrete points in the space, and the overlap between multiple rectangles is reduced. In other words, after each iteration, the rectangle attribute is more similar to the spatial point attribute x, and the overlapping area between multiple rectangles is less.


The concrete iterative method uses the idea of mean IFT to search the optimal value in the interior of the rectangle, but it often leads to local optimization.


Dependency package:
pip install -r requriements.txt
