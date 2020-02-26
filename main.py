# Model the Earth as a sphere with a magnetic dipole embeddedat its centre (remember that the direction of the magnetic dipoleis tilted with regard to the ecliptic).
# Plot the magnetic field inat  least  2  of  the  3  planes  through  the  origin:  the xy and xz planes.
# Choose the x axis from the Sun toward the Earth and the z axis perpendicular to the ecliptic.

"""
             ^
             z
(s) -- x --> |  (J) 

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



ecliptic_tilt = 23.4  # degrees
ecliptic_tilt = np.radians(ecliptic_tilt)

PLOT_FIELD = True
SAVE_FIG = True

def B(x,y,z):
    r = np.maximum(np.sqrt(x**2 + y**2 + z**2), 0.01)
    m = 100  # magnetic strength
    m_x = np.sin(ecliptic_tilt) * m
    m_y = 0 * m
    m_z = -np.cos(ecliptic_tilt) * m
    inner = (x*m_x + y*m_y + z*m_z)

    B_x = 3*x*inner/r**5 - m_x/r**3
    B_y = 3*y*inner/r**5 - m_y/r**3
    B_z = 3*z*inner/r**5 - m_z/r**3

    return np.array([B_x, B_y, B_z])

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
z = np.linspace(-2, 2, 100)

if PLOT_FIELD:
    # xy plane
    xx, zz = np.meshgrid(x, z)
    Bx, By, Bz = B(xx, np.zeros_like(xx), zz)
    plt.streamplot(xx, zz, Bx, Bz)
    plt.title("$xz$-plane")
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    if SAVE_FIG:
        plt.savefig("B_field_xz_plane.pdf")
        plt.clf()
    else:
        plt.show()

    # xy plane
    xx, yy = np.meshgrid(x, y)
    Bx, By, Bz = B(xx, yy, np.zeros_like(xx))
    plt.streamplot(xx, yy, Bx, By)
    plt.title("$xy$-plane")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    if SAVE_FIG:
        plt.savefig("B_field_xy_plane.pdf")
        plt.clf()
    else:
        plt.show()

### Rk45 stuff ###

def f(t, y):
    # y = [x, y, z, vx, vy, vz]
    cross = np.cross(y[3:], B(*y[:3]))
    return np.concatenate((y[3:], cross))

def get_path(x=-20, y=0, z=0, vx=0.1, vy=0, vz=0,
             max_step=10000, return_iter=False, return_vel=False):
    max_step = 10000
    y_list = np.empty([max_step, 6])
    t_list = np.empty(max_step)
    t_list[0] = 0
    y_list[0] = [x, y, z, vx, vy, vz]
    solver = RK45(f, 0, y_list[0], 100, max_step=max_step)
    i = 1

    while solver.status == "running":
        solver.step()
        y_list[i] = solver.y
        t_list[i] = solver.t
        i += 1
    i_end = i

    x = y_list[:i_end, 0]
    y = y_list[:i_end, 1]
    z = y_list[:i_end, 2]
    vx = y_list[:i_end, 3]
    vy = y_list[:i_end, 4]
    vz = y_list[:i_end, 5]

    if return_vel:
        return x, y, z, vx, vy, vz

    if return_iter:
        return x, y, z, i_end-1

    return x, y, z,

x = []
y = []
z = []
vals = [(7, 0.4), (5, 0.4), (0, 1), (-4, 0.4)]
for i, (z_0, vz_0) in enumerate(vals):
    xi, yi, zi = get_path(z=z_0, vx=vz_0)
    x.append(xi)
    y.append(yi)
    z.append(zi)


# xz plane ####
plt.title("XZ plane")
for i in range(len(vals)):
    plt.plot(x[i], z[i], label=f"$vz_0: {vals[i][1]}$, $z_0: {vals[i][0]}$")
plt.axvline(0, c="gray")
ax_line = lambda x: -x/np.tan(ecliptic_tilt)  # ecliptic axis
plt.plot([-10, 10], [ax_line(-10), ax_line(10)], linestyle="--", color="gray")  # ecliptic axis
plt.scatter([0], [0], s=3000, zorder=10)  # earth

# B-field
x_lin = np.linspace(-10, 10, 100)
z_lin = np.linspace(-10, 10, 100)
xx, zz = np.meshgrid(x_lin, z_lin)
Bx, By, Bz = B(xx, np.zeros_like(xx), zz)
stream = plt.streamplot(xx, zz, Bx, Bz, color="gray", linewidth=0.5)

#plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("z")
plt.legend()
plt.xlim(-20, 12)
plt.ylim(-20, 14)
if SAVE_FIG:
    plt.savefig("trajectory_xz_plane.pdf")
    plt.clf()
else:
    plt.show()

# xy plane #####
plt.title("XY plane")
for i in range(len(vals)):
    plt.plot(x[i], y[i], label=f"$vz_0: {vals[i][1]}$, $z_0: {vals[i][0]}$")
plt.scatter([0], [0], s=3000, zorder=10)  # earth

plt.axhline(color="gray")
#plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.xlim(-20, 12)
plt.ylim(-20, 14)
if SAVE_FIG:
    plt.savefig("trajectory_xy_plane.pdf")
    plt.clf()
else:
    plt.show()

## Energy validation
x, y, z, vx, vy, vz = get_path(5, 0.4, return_vel=True)
E = vx**2 + vy**2 + vz**2
plt.title("Energy")
plt.plot(100*(E-E[0])/E[0])
plt.ylabel("Relative error [%]")
plt.xlabel("Step number")
if SAVE_FIG:
    plt.savefig("energy.pdf")
    plt.clf()
else:
    plt.show()
