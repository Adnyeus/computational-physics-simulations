import numpy as np
from scipy.signal import convolve2d

GRID_SIZE = 100
Du, Dv = 0.16, 0.08
F, k = 0.035, 0.060
dt = 1.0

laplacian_kernel = np.array([
    [0.05, 0.20, 0.05],
    [0.20, -1.0, 0.20],
    [0.05, 0.20, 0.05]
])

def step_pde(U, V):
    """Gray-Scott Reaction-Diffusion PDE Numerical Solver."""
    Lu = convolve2d(U, laplacian_kernel, mode='same', boundary='wrap')
    Lv = convolve2d(V, laplacian_kernel, mode='same', boundary='wrap')
    
    uvv = U * (V**2)
    dU = (Du * Lu - uvv + F * (1.0 - U)) * dt
    dV = (Dv * Lv + uvv - (F + k) * V) * dt
    
    return np.clip(U + dU, 0, 1), np.clip(V + dV, 0, 1)

def update_agents(positions, health, field_V):
    """Chemotactic Gradient Taxis Agent Kinematics."""
    gy, gx = np.gradient(field_V)
    
    for i in range(len(positions)):
        if health[i] <= 0:
            continue
            
        x = int(positions[i, 0]) % GRID_SIZE
        y = int(positions[i, 1]) % GRID_SIZE
        
        grad_x = gx[y, x]
        grad_y = gy[y, x]
        grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        
        step_x = (grad_x / grad_mag) * 0.8 + np.random.uniform(-0.2, 0.2)
        step_y = (grad_y / grad_mag) * 0.8 + np.random.uniform(-0.2, 0.2)
        
        positions[i, 0] = (positions[i, 0] + step_x) % GRID_SIZE
        positions[i, 1] = (positions[i, 1] + step_y) % GRID_SIZE
        
        health[i] += field_V[y, x] * 5.0 - 0.4
        health[i] = min(health[i], 150.0)
        
    return positions, health