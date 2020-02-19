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
from scipy.integrate import solve_ivp

plt.style.use('bmh')

newParams = {'figure.figsize'  : (12, 6),  # Figure size
             'figure.dpi'      : 200,      # figure resolution
             'axes.titlesize'  : 20,       # fontsize of title
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

PLOT_FIELD = False

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

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# xx,yy,zz = np.meshgrid(x,y,z)
# Bx,By,Bz = B(xx,yy,zz)
# ax.quiver(xx,yy,zz,Bx,By,Bz, normalize=True, length=0.2)
# plt.show()

if PLOT_FIELD:
    # xy plane
    xx, zz = np.meshgrid(x, z)
    Bx, By, Bz = B(xx, np.zeros_like(xx), zz)
    plt.streamplot(xx, zz, Bx, Bz)
    plt.title("$xz$-plane")
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.show()

    # xy plane
    xx, yy = np.meshgrid(x, y)
    Bx, By, Bz = B(xx, yy, np.zeros_like(xx))
    plt.streamplot(xx, yy, Bx, By)
    plt.title("$xy$-plane")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()


## Simulate particles ##

def F(vx, vy, vz, Bx, By, Bz):
    Fx = vy*Bz - vz*By
    Fy = vz*Bx - vx*Bz
    Fz = vx*By - vy*Bx

    return Fx, Fy, Fz

def RK4(y, h):
    """Time-independent RHS assumed"""
    # y = np.array([x, y, z], [vx, vy, vz])
    def f(y):
        return np.array([
            y[1],
            np.cross(y[1], B(*y[0]))
        ])
    k1 = f(y)
    k2 = f(y + k1/2)
    k3 = f(y + k2/2)
    k4 = f(y + k3)

    y_next = y + (k1 + 2*k2 + 2*k3 + k4)*h/6
    return y_next

x = -20
y = 10
z = 0
vx = 0.1
vy = 0
vz = 0

end_time = 100000
step = 0.005

y_list = np.empty([end_time, 2, 3])
y_list[0] = np.array([[x, y, z], [vx, vy, vz]])

# RK4
for i in range(1, end_time):
    y_list[i] = RK4(y_list[i-1], step)

# x_list, y_list, z_list = [], [], []
# for i in range(end_time):
#     x_list.append(x)
#     y_list.append(y)
#     z_list.append(z)
#     Bx, By, Bz = B(x, y, z)
#     Fx, Fy, Fz = F(vx, vy, vz, Bx, By, Bz)
#     x += vx*step
#     y += vy*step
#     x += vz*step
#     vx += Fx*step  # Yes, we just ignore mass
#     vy += Fy*step
#     vz += Fz*step

x = y_list[:,0,0]
y = y_list[:,0,1]
z = y_list[:,0,2]

plt.plot(x,z)
plt.show()
plt.plot(x,y)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y_list[0, 0, 0], y_list[0, 0, 1], y_list[0, 0, 2], label="Start", marker="x")
ax.scatter(y_list[-1, 0, 0], y_list[-1, 0, 1], y_list[-1, 0, 2], label="End", marker="o")
ax.plot(y_list[:, 0, 0], y_list[:, 0, 1], y_list[:, 0, 2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()

# Plot energy
E = np.sum(y_list[:, 1, :]**2, axis=1)
print(np.max(E), np.min(E))
plt.plot(np.arange(end_time)*step, E)
plt.show()

