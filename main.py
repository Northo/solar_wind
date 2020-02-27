# Model the Earth as a sphere with a magnetic dipole embeddedat its centre (remember that the direction of the magnetic dipoleis tilted with regard to the ecliptic).
# Plot the magnetic field inat  least  2  of  the  3  planes  through  the  origin:  the xy and xz planes.
# Choose the x axis from the Sun toward the Earth and the z axis perpendicular to the ecliptic.

"""
About units:
------------
We use here dimensionless quantities. They are defined as, where we use the primed version
r = Rj * r', Rj = 6.4E6 m, Earth's radius.
v = v0 * v', v0 = 2.5E5 m/s, typical speed of solar wind.
From this, we have implicitly defined
t = t0 * t', t0 = Rj / v0 = 25 s.

This means that to convert to real units from this program, simply multiply
with the apropriate constant, Rj, v0, or t0.
Example:
-------.
If simulate 10 time units here, that corresponds to
t = t0 * 100 = 250 s.
From an execution of the program, we see that a particle
having velocity 1, travels 10 units of space during 10 units of time.
r = Rj * 1 = 6.4E6 m,
which is what we could excpect.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import RK45

plt.style.use('bmh')

newParams = {'figure.figsize'  : (12, 6),  # Figure size
             'figure.dpi'      : 200,      # figure resolution
             'axes.titlesize'  : 30,       # fontsize of title
             'axes.labelsize'  : 11,       # fontsize of axes labels
             'axes.linewidth'  : 2,        # width of the figure box lines
             'lines.linewidth' : 1,        # width of the plotted lines
             'savefig.dpi'     : 200,      # resolution of a figured saved using plt.savefig(filename)
             'ytick.labelsize' : 11,       # fontsize of tick labels on y axis
             'xtick.labelsize' : 11,       # fontsize of tick labels on x axis
             'legend.fontsize' : 12,       # fontsize of labels in legend
             'legend.frameon'  : True,     # activate frame on lengend?
            }
plt.rcParams.update(newParams) # Set new plotting parameters

########## Setup ##########
# Meta setup
PLOT_FIELD = True
SAVE_FIG = True
FIG_DIR = "media/"

# Physics setup
END_TIME = 10
ecliptic_tilt = 23.4  # degrees
ecliptic_tilt = np.radians(ecliptic_tilt)

# Pairs of initial position and initial velocity to check (z0, vz0)
vals = [
    (7, 0.4),
    (5, 0.4),
    (0, 1),
    (-4, 0.4),
]


########## Function definitions ##########
# Nothing should be executed in this section

def B(x, y, z):
    """B-field in r=(x,y,z)"""
    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 0.01)  # Avoid singularity
    B_0 = 100  # q*B_0*Rj^3*V0

    m_x = np.sin(ecliptic_tilt)
    m_y = 0
    m_z = -np.cos(ecliptic_tilt)
    inner = (x*m_x + y*m_y + z*m_z)

    B_x = 3*x*inner/r**5 - m_x/r**3
    B_y = 3*y*inner/r**5 - m_y/r**3
    B_z = 3*z*inner/r**5 - m_z/r**3

    return np.array([B_x, B_y, B_z])*B_0


def acceleration(t, y):
    # y = [x, y, z, vx, vy, vz]
    # Ignore mass, as we have absorbed this in B
    cross = np.cross(y[3:], B(*y[:3]))
    return np.concatenate((y[3:], cross))


def get_path(x=-20, y=0, z=0, vx=0.1, vy=0, vz=0,
             max_step=10000, return_iter=False, return_vel=False):
    # Allocate memory for maximum possible steps
    y_list = np.empty([max_step, 6])
    t_list = np.empty(max_step)
    t_list[0] = 0
    y_list[0] = [x, y, z, vx, vy, vz]
    solver = RK45(acceleration, 0, y_list[0], END_TIME, max_step=max_step)
    i = 1

    while solver.status == "running":
        solver.step()
        y_list[i] = solver.y
        t_list[i] = solver.t
        i += 1

    # We probably did not use all the memory we allocated
    # so slice out the part we used
    x = y_list[:i, 0]
    y = y_list[:i, 1]
    z = y_list[:i, 2]
    vx = y_list[:i, 3]
    vy = y_list[:i, 4]
    vz = y_list[:i, 5]

    if return_vel:
        return x, y, z, vx, vy, vz

    if return_iter:
        return x, y, z, i-1

    return x, y, z,

def plot_field(x1, x2, Bx1, Bx2, name, savefig=False):
    """Plots field in the x1x2-plane.
    Parameters:
       x1   : 2D array, meshgrid of first axis
       x2   : 2D array, meshgrid of second axis
       Bx1  : 2D array, meshgrid of B-fields first axis
       Bx2  : 2D array, meshgrid of B-fields second axis
       name : string, two letter name of plane, f.eks. "xy"
    """
    plt.streamplot(x1, x2, Bx1, Bx2)
    plt.title(f"${name}$-plane")
    plt.xlabel(f"${name[0]}$")
    plt.ylabel(f"${name[1]}$")
    if SAVE_FIG:
        plt.savefig(f"{FIG_DIR}B_field_{name}_plane.pdf")
        plt.clf()
    else:
        plt.show()

def _get_label(i):
    """Helper function, returns formated string for inital value i"""
    return f"$vz_0: {vals[i][1]}$, $z_0: {vals[i][0]}$"


########## Calculations ##########
# Create space
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
z = np.linspace(-10, 10, 100)

path_x = []
path_y = []
path_z = []
path_vx = []
path_vy = []
path_vz = []
for i, (z_0, vz_0) in enumerate(vals):
    xi, yi, zi, vxi, vyi, vzi = get_path(z=z_0, vx=vz_0, return_vel=True)
    path_x.append(xi)
    path_y.append(yi)
    path_z.append(zi)
    path_vx.append(vxi)
    path_vy.append(vyi)
    path_vz.append(vzi)

# Energy conservation
E = []
for i, val in enumerate(vals):
    E.append(path_vx[i]**2 + path_vy[i]**2 + path_vz[i]**2)  # Ignoring prefactor

########## Plotting ##########
if PLOT_FIELD:
    # xz plane
    xx, zz = np.meshgrid(x, z)
    Bx, By, Bz = B(xx, np.zeros_like(xx), zz)
    plot_field(xx, zz, Bx, Bz, "xz", savefig=SAVE_FIG)
    # xy plane
    xx, yy = np.meshgrid(x, y)
    Bx, By, Bz = B(xx, yy, np.zeros_like(xx))
    plot_field(xx, yy, Bx, By, "xy", savefig=SAVE_FIG)

# Plot trajectories

############
# xz plane #
############
plt.title("XZ plane")
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("z")
plt.xlim(-20, 12)
plt.ylim(-20, 14)

for i in range(len(vals)):  # Plot each path
    plt.plot(path_x[i], path_z[i], label=_get_label(i))
plt.axvline(0, c="gray")  # Plot x-axis (axis of travel)
ax_line = lambda x: -x/np.tan(ecliptic_tilt)  # ecliptic axis slope
plt.plot([-10, 10], [ax_line(-10), ax_line(10)], linestyle="--", color="gray")
earth = plt.Circle((0,0), radius=1)
plt.gca().add_artist(earth)

# B-field
xx, zz = np.meshgrid(x, z)
Bx, By, Bz = B(xx, np.zeros_like(xx), zz)
stream = plt.streamplot(xx, zz, Bx, Bz, color="gray", linewidth=0.5)

plt.legend()
if SAVE_FIG:
    plt.savefig(f"{FIG_DIR}trajectory_xz_plane.pdf")
    plt.clf()
else:
    plt.show()

############
# xy plane #
############
plt.title("XY plane")
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-20, 12)
plt.ylim(-20, 14)

for i in range(len(vals)):  # Plot each path
    plt.plot(path_x[i], path_y[i], label=_get_label(i))
earth = plt.Circle((0,0), radius=1)
plt.gca().add_artist(earth)

plt.legend()
plt.axhline(color="gray")  # Axis of travel

if SAVE_FIG:
    plt.savefig(f"{FIG_DIR}trajectory_xy_plane.pdf")
    plt.clf()
else:
    plt.show()

#####################
# Energy validation #
#####################
plt.title("Energy")
plt.ylabel("Relative error [%]")
plt.xlabel("Step number")
for i in range(len(vals)):
    plt.plot(100*(E[i]-E[i][0])/E[i][0], label=_get_label(i))  # Relative error
plt.legend()

if SAVE_FIG:
    plt.savefig(f"{FIG_DIR}energy.pdf")
    plt.clf()
else:
    plt.show()
