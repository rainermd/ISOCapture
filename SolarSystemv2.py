import rebound as rb
import numpy as np

"""
Set-Up/Parameters
-Integrator: WHFast, needed for quick simulations. May consider moving to higher accuracy IAS15. Set time step to large for quicker
and small for higher accuracy
-Widget: Uncomment in case you like visualization (will slow down)
-N_active: Only major gravitating bodies, i.e. Sun, Jupiter, Saturn (may include Neptune later)
-G=1: Natural units, everything in AU, years. May want to switch to SI
-r_max: Radius of the sphere we incorporate with particles
-No: Number of particles, note speed scales with No^2
-T: Lifetime of simulation
-date: Time of initialization, may include new dates
"""
sim=rb.Simulation()
sim.integrator="whfast"
sim.dt=1e-3
#sim.widget(port=1234, size=(800,800))
sim.N_active=3
sim.G=1 
r_max=10 
No=1000000 
T=0.5 #lifetime of simulation
date="2025-01-22 18:00" #arbitrary date, may need to include time dependence

"""
Initialize solar system
"""
sim.add("Sun", date=date, hash="Sun")
sim.particles[0].r=0.00465047
sim.add("Jupiter",date=date, hash="Jupiter")
sim.particles[1].r=0.00046732617
sim.add("Saturn",date=date, hash="Saturn")
sim.particles[2].r=0.000389256877
sim.move_to_com()
sim.configure_box(3e10)

"""
Collision
-Type: tree, scales computation time as Noln(No), may test other types
-Resolve: Custom, removes test particles if they collide with main objects, and does not include test-test particle interaction
"""
sim.collision = "tree" #collision method
def collision_solver(sim_pointer, collision):
    sim = sim_pointer.contents
    if sim.particles[collision.p1]==sim.particles["Sun"] or sim.particles[collision.p1]==sim.particles["Jupiter"] or sim.particles[collision.p1]==sim.particles["Saturn"]:
        return 2
    if sim.particles[collision.p2]==sim.particles["Sun"] or sim.particles[collision.p2]==sim.particles["Jupiter"] or sim.particles[collision.p2]==sim.particles["Saturn"]:
        return 1
    else:
        return 0

sim.collision_resolve = collision_solver

"""
Random distribution
-rProb: Returns scaling as r^2, up to r_max, since cross sectional area scales that way
-angles (theta0, phi0): We assume isotropic, may change?
-v:Define the velocity range in G=1 units, see https://arxiv.org/pdf/2103.03289, a log-normal distribution!
-Impact: Define v to be perpendicular to CoM vector, i.e. as an impact paramater
"""
def rProb(n):
    cdf_inverse = lambda u: (r_max**3 * u) ** (1 / 3)
    uniform_samples = np.random.uniform(0, 1, n)
    return cdf_inverse(uniform_samples)
    
r0 = rProb(No) 
theta0 = np.arccos(1 - 2 * np.random.uniform(0, 1, No))
phi0 = np.random.random(No) * 2 * np.pi 

x0 = r0 * np.cos(phi0) * np.sin(theta0)
y0 = r0 * np.sin(phi0) * np.sin(theta0)
z0 = r0 * np.cos(theta0)

v_mean = 4.921174692  
v_var = 0.8265983871  

q = np.random.uniform(0,2*np.pi,No)  
v = np.random.lognormal(mean=v_mean, sigma=v_var, size=No) 
v0x, v0y, v0z = np.zeros(No), np.zeros(No), np.zeros(No)

for i in range(No):
    pos_vec = np.array([x0[i], y0[i], z0[i]])
    norm_pos_vec = np.linalg.norm(pos_vec)
    
    if np.allclose(norm_pos_vec, 0):
        raise ValueError("Position vector must be non-zero")
    
    # Normalize position vector
    pos_unit = pos_vec / norm_pos_vec

    # Find any vector not parallel to pos_unit
    arbitrary_vec = np.array([1, 0, 0]) if not np.isclose(pos_unit[0], 1) else np.array([0, 1, 0])

    # Compute perpendicular vector using Gram-Schmidt process
    perp_vec1 = arbitrary_vec - np.dot(arbitrary_vec, pos_unit) * pos_unit
    perp_vec1 /= np.linalg.norm(perp_vec1)

    # Compute a second perpendicular vector
    perp_vec2 = np.cross(pos_unit, perp_vec1)

    # Combine components with random phase
    v_perpendicular = v[i] * (np.cos(q[i]) * perp_vec1 + np.sin(q[i]) * perp_vec2)
    
    # Assign velocities
    v0x[i], v0y[i], v0z[i] = v_perpendicular
    
particles = [
    rb.Particle(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, r=1.6e-8)
    for x, y, z, vx, vy, vz in zip(x0, y0, z0, v0x, v0y, v0z)
]
sim.add(particles)

"""
Simulate
-Integrate backwards first, delete particles that are already bound.
    SINCE capture is so unlikely, it may be more efficient to integrate forwards first and THEN integrate backwards
-Count # of bound particles
"""
sim.integrate(T)
NoCollide=(No+3-sim.N)/No
boundp0=np.array([p.e < 1 for p in sim.particles[3:]])
for i in range(sim.N-3):
    if not boundp0[i]:
        sim.remove(i+3, keep_sorted=False)

sim.integrate(-T)
boundn=np.array([p.e < 1 for p in sim.particles[3:]])
for i in range(sim.N-3):
    if boundn[i]:
        sim.remove(i+3, keep_sorted=False)

"""
Analysis
-ec: Eccentricity
    Should do for multiple orbital parameters, and also for initial conditions.
-Bound fraction
-Collide Fraction
"""
N=sim.N-3
bound_fraction=N/No

sim.integrate(T)

print(f"Bound fraction: {bound_fraction}, Fraction Collided: {NoCollide}")
for i in range(N):
    print(sim.particles[i+3].e)
