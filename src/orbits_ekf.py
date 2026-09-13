import numpy as np

GM = 1.0

def acceleration(pos):
    r = np.linalg.norm(pos)
    return -GM * pos / (r**3)

def adaptive_verlet_step(pos, vel, dt_min=0.001, dt_max=0.05, eta=0.01):
    """Adaptive Symplectic Velocity Verlet Integrator Step."""
    r = np.linalg.norm(pos)
    v = np.linalg.norm(vel)
    
    dt = eta * (r / (v + 1e-6))
    dt = np.clip(dt, dt_min, dt_max)
    
    a = acceleration(pos)
    vel_half = vel + 0.5 * dt * a
    pos_next = pos + dt * vel_half
    
    a_next = acceleration(pos_next)
    vel_next = vel_half + 0.5 * dt * a_next
    
    return pos_next, vel_next, dt

def get_state_jacobian(pos, dt):
    """Dynamic State Transition Matrix (F_k) Jacobian Computation."""
    x, y = pos[0], pos[1]
    r = np.sqrt(x**2 + y**2)
    r5 = r**5
    
    jax_a_p = np.array([
        [-GM/r**3 + 3*GM*x**2/r5,  3*GM*x*y/r5],
        [ 3*GM*x*y/r5,            -GM/r**3 + 3*GM*y**2/r5]
    ])
    
    F = np.eye(4)
    F[0:2, 2:4] = np.eye(2) * dt
    F[2:4, 0:2] = jax_a_p * dt
    return F

def get_measurement_jacobian(pos):
    """Polar Measurement Matrix (H_k) Jacobian Computation."""
    x, y = pos[0], pos[1]
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    
    H = np.zeros((2, 4))
    H[0, 0] = x / r
    H[0, 1] = y / r
    H[1, 0] = -y / r2
    H[1, 1] = x / r2
    return H