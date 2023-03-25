# Newton's Fractal

Newton's Fractal is a fractal structure created by applying Newton's method to approximate the roots of functions.

For each pixel, we convert its coordinate to a complex number and apply newtons method to get a new point p[n+1] = p[n] - f(p[n])  / f'(p[n]) where f(x) is the polynomial whose roots we're approximating and f'(x) is its derivative. We iteratively apply the function to the initial point some number of times (16 is usually plenty) and color the original pixel based on which root the iterated sequence approached.

Typically, a point quickly approaches the root that it is nearest to. However, on the borders between the regions, the behavior of the points becomes chaotic and produces an interesting fractal structure.