#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

global_stars_count = 1400

comm = MPI.COMM_WORLD
thread_count = comm.Get_size()
thread_id = comm.Get_rank()

start_star = (thread_id * global_stars_count) // thread_count
end_star = ((thread_id+1) * global_stars_count) // thread_count

stars_count = end_star - start_star

buff_size = int(np.ceil(global_stars_count / thread_count))

stars = np.random.rand(stars_count,4)

G = 6.67408e-11

def parallel(stars):
    my_star_count = stars.shape[0]

    prev_thread = thread_count - 1 if thread_id == 0 else thread_id - 1
    next_thread = 0 if thread_id == thread_count -1 else thread_id + 1

    acceleration = np.zeros(shape=(stars_count,3), dtype=np.float32)

    other_stars = np.zeros(shape=(buff_size, 4), dtype=np.float32)
    other_stars[:my_star_count,:] = stars.copy()

    for _i in range(thread_count - 1):
        print(f"{thread_id} start")
        comm.Send([other_stars, MPI.FLOAT], dest=next_thread)
        print(f"{thread_id} sent")
        comm.Recv([other_stars, MPI.FLOAT], source=prev_thread)
        print(f"{thread_id} recvd")
        for i in range(my_star_count):
            for j in range(other_stars.shape[0]):
                acceleration[i,:] += newtown(stars[i,:], other_stars[j,:])

    for i in range(my_star_count):
        for j in range(my_star_count):
            if i == j: 
                continue
            acceleration[i,:] += newtown(stars[i,:], stars[j,:])
    
    if thread_id != 0:
        comm.Send([acceleration, MPI.FLOAT], dest=0)
        return None

    total_acceleration = np.empty((global_stars_count,3), dtype=np.float32)
    total_acceleration[start_star:end_star,:] = acceleration

    other_acceleration = np.empty((buff_size, 3), dtype=np.float32)
    for i in range(1, thread_count): 
        other_start_star = (i * global_stars_count) // thread_count
        other_end_star = ((i+1) * global_stars_count) // thread_count
        other_size = other_end_star-other_start_star
        comm.Recv([other_acceleration[:other_size, :], MPI.FLOAT], source=i)
        total_acceleration[other_start_star:other_end_star,:] = other_acceleration[:other_size, :]


    return G * total_acceleration


def newtown(star1, star2):
    m1, x1, y1, z1 = star1
    m2, x2, y2, z2 = star2

    dist = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2 + (z1 - z2)** 2) + 1e-20

    m_dist = m2 / (dist ** 3)

    ax = m_dist * (x1 - x2)
    ay = m_dist * (y1 - y2)
    az = m_dist * (z1 - z2)

    return ax, ay, az


result = parallel(stars)
if result is not None:
    print(result)