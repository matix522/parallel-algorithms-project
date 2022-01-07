#!/usr/bin/env python
import numpy as np

global_stars_count = 4

stars_count = global_stars_count

np.random.seed(2137)
stars = np.array([[1, 3, 3, 3],
                [2, -3.0, 1.5, 0.0],
                [3, 2.0, 2.0, 2.0],
                [4, -1.0, 1.0, -2.0],
                ])
G = 6.67408e-11

def serial(stars):
    my_star_count = stars.shape[0]

    acceleration = np.zeros(shape=(stars_count,3))

    for i in range(my_star_count):
        for j in range(my_star_count):
            if i == j: 
                continue
            acceleration[i,:] += newtown(stars[i,:], stars[j,:])
    return G * acceleration

def newtown(star1, star2):
    m1, x1, y1, z1 = star1
    m2, x2, y2, z2 = star2

    dist = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2 + (z1 - z2)** 2) + 1e-20

    m_dist = m2 / (dist ** 3)

    ax = m_dist * (x1 - x2)
    ay = m_dist * (y1 - y2)
    az = m_dist * (z1 - z2)

    return ax, ay, az


print(serial(stars))