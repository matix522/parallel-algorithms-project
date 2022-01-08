#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

global_stars_count = 4

comm = MPI.COMM_WORLD
thread_count = comm.Get_size()
thread_id = comm.Get_rank()

start_star = (thread_id * global_stars_count) // thread_count
end_star = ((thread_id+1) * global_stars_count) // thread_count

stars_count = end_star - start_star

np.random.seed(2137)
stars = np.array([[1, 3, 3, 3],
                [2, -3.0, 1.5, 0.0],
                [3, 2.0, 2.0, 2.0],
                [4, -1.0, 1.0, -2.0],
                ])
                
stars = stars[start_star:end_star]

buff_size = int(np.ceil(global_stars_count / thread_count))

G = 6.67408e-11

def parallel(stars):
    my_star_count = stars.shape[0]

    prev_thread = thread_count - 1 if thread_id == 0 else thread_id - 1
    next_thread = 0 if thread_id == thread_count -1 else thread_id + 1

    acceleration = np.zeros(shape=(stars_count,3), dtype=np.float32)

    other_stars = np.zeros(shape=(buff_size, 4), dtype=np.float32)
    other_stars[:my_star_count,:] = stars.copy()
    other_acceleration = np.zeros(shape=(buff_size, 3), dtype=np.float32)

    for _i in range(int(np.ceil(thread_count / 2)) - 1):
        comm.Send([other_stars, MPI.FLOAT], dest=next_thread)
        comm.Send([other_acceleration, MPI.FLOAT], dest=next_thread)
        comm.Recv([other_stars, MPI.FLOAT], source=prev_thread)
        comm.Recv([other_acceleration, MPI.FLOAT], source=prev_thread)
        for i in range(my_star_count):
            for j in range(other_stars.shape[0]):
                acceleration_i, acceleration_j = newtown_2(stars[i,:], other_stars[j,:])
                acceleration[i,:] += acceleration_i
                other_acceleration[j,:] += acceleration_j

    if thread_count % 2 == 0:
        comm.Send([other_stars, MPI.FLOAT], dest=next_thread)
        comm.Recv([other_stars, MPI.FLOAT], source=prev_thread)
        for i in range(my_star_count):
            for j in range(other_stars.shape[0]):
                acceleration_i, _ = newtown_2(stars[i,:], other_stars[j,:])
                acceleration[i,:] += acceleration_i

    other_acceleration_dest = thread_id - (int(np.ceil(thread_count / 2)) - 1)
    if other_acceleration_dest < 0:
        other_acceleration_dest += thread_count
    other_acceleration_source = thread_id + (int(np.ceil(thread_count / 2)) - 1)
    if other_acceleration_source >= thread_count:
        other_acceleration_source -= thread_count
    print(thread_id, other_acceleration_dest, other_acceleration_source)
    # if thread_id == 0:
    # print(thread_id, other_acceleration)
    comm.Send([other_acceleration, MPI.FLOAT], dest=other_acceleration_dest)
    comm.Recv([other_acceleration, MPI.FLOAT], source=other_acceleration_source)

    for i in range(my_star_count):
        for j in range(my_star_count):
            if i == j: 
                continue
            acceleration_i, _ = newtown_2(stars[i,:], stars[j,:])
            acceleration[i,:] += acceleration_i
    acceleration[:,:] += other_acceleration[:my_star_count, :]
    
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


def newtown_2(star1, star2):
    m1, x1, y1, z1 = star1
    m2, x2, y2, z2 = star2

    if m1 == 0 or m2 == 0:
        return [0,0,0], [0,0,0]

    dist = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2 + (z1 - z2)** 2) + 1e-20

    dist_3 = dist ** 3

    ax = (x1 - x2) / dist_3
    ay = (y1 - y2) / dist_3
    az = (z1 - z2) / dist_3
    a = np.array([ax, ay, az])

    return m2 * a, - m1 * a


result = parallel(stars)
if result is not None:
    print(result)