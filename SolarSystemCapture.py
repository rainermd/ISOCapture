import rebound as rb
import numpy as np
from scipy.stats import rv_continuous

"""
Set-Up/Parameters
"""
sim=rb.Simulation()
sim.integrator="whfast"
sim.dt=1e-3
sim.widget(port=1234, size=(800,800))
#sim.softening=1e-6
sim.N_active=3
sim.G=1 #natural units
r_max=5 #radius of interest
No=20000 # Number of particles
T=1 #lifetime of simulation
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
"""
def rProb(n):
    cdf_inverse = lambda u: (r_max**3 * u) ** (1 / 3)
    uniform_samples = np.random.uniform(0, 1, n)
    return cdf_inverse(uniform_samples)

r0 = rProb(No)  #Set sphere of influence to be up to around Neptune (<30 AU)
theta0 = np.arccos(1 - 2 * np.random.uniform(0, 1, No))
phi0 = np.random.random(No) * 2 * np.pi  # Random azimuthal angles

# Convert to Cartesian coordinates (relative to com)

x0 = r0 * np.cos(phi0) * np.sin(theta0)
y0 = r0 * np.sin(phi0) * np.sin(theta0)
z0 = r0 * np.cos(theta0)

# Define the velocity range in G=1 units, based off of astronomer estimates, MAY CHANGE
v_min = 0  # Minimum velocity
v_max = 10   # Maximum velocity

# Generate velocities
q = np.random.uniform(0,2*np.pi,No)  # Direction parameter, MAY WANT TO TWEAK DISTRUBUTION, assume random here
v = np.random.uniform(v_min, v_max, No)  # Velocity magnitudes, MAY WANT TO TWEAK DISTRIBUTION, we assume uniform here
v0x, v0y, v0z = np.zeros(No), np.zeros(No), np.zeros(No)

# Allocate arrays for velocities
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
"""
sim.integrate(-T)

boundn = np.array([p.e <= 1 for p in sim.particles[3:]])
for i in range(sim.N-3):
    if boundn[i]:
        sim.remove(i+3, keep_sorted=False)

NoUnbound = sim.N
# Integrate forward and calculate bound fraction
sim.integrate(T)
boundp = np.array([p.e <= 1 for p in sim.particles[3:]])

"""
Analysis
"""
ec=[]
for i in range(sim.N-3):
    if sim.particles[i+3].e<=1:
        ec.append(sim.particles[i+3].e)
bound_fraction = np.sum(boundp) / NoUnbound

NoCollide=(NoUnbound-sim.N)/NoUnbound

print(f"Bound fraction: {bound_fraction}, Fraction Collided: {NoCollide}")
for i in range(len(ec)):
    print(ec[i])
