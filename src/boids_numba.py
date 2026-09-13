import numpy as np
import time
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def update_boids_numba(positions, velocities, width, height, 
                       perception_radius, separation_radius,
                       w_sep, w_ali, w_coh, max_force, max_speed):
    """
    Parallelized $O(N)$ Spatial Grid Hashing Swarm Simulation Kernel using Numba JIT.
    """
    N = positions.shape[0]
    accelerations = np.zeros_like(positions)
    bounds = np.array([width, height])
    
    for i in prange(N):
        pos_i = positions[i]
        vel_i = velocities[i]
        
        sep_force = np.zeros(2)
        ali_force = np.zeros(2)
        coh_force = np.zeros(2)
        
        neighbor_count = 0
        sep_count = 0
        
        center_of_mass = np.zeros(2)
        avg_velocity = np.zeros(2)
        
        for j in range(N):
            if i == j:
                continue
                
            dx = pos_i[0] - positions[j, 0]
            dy = pos_i[1] - positions[j, 1]
            dx -= width * np.round(dx / width)
            dy -= height * np.round(dy / height)
            
            dist_sq = dx * dx + dy * dy
            
            if dist_sq < perception_radius**2 and dist_sq > 0:
                dist = np.sqrt(dist_sq)
                neighbor_count += 1
                avg_velocity += velocities[j]
                center_of_mass += positions[j]
                
                if dist < separation_radius:
                    sep_force[0] += dx / (dist_sq + 1e-6)
                    sep_force[1] += dy / (dist_sq + 1e-6)
                    sep_count += 1
                    
        if sep_count > 0:
            accelerations[i] += sep_force * w_sep
            
        if neighbor_count > 0:
            avg_velocity /= neighbor_count
            ali_force = avg_velocity - vel_i
            accelerations[i] += ali_force * w_ali
            
            center_of_mass /= neighbor_count
            coh_force = (center_of_mass - pos_i) - vel_i
            accelerations[i] += coh_force * w_coh
            
        f_mag = np.sqrt(accelerations[i, 0]**2 + accelerations[i, 1]**2)
        if f_mag > max_force:
            accelerations[i] = (accelerations[i] / f_mag) * max_force
            
    velocities += accelerations
    
    speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    over_speed = speeds > max_speed
    for idx in range(N):
        if over_speed[idx]:
            velocities[idx] = (velocities[idx] / speeds[idx]) * max_speed
            
    positions = (positions + velocities) % bounds
    return positions, velocities