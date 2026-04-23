import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def lane_emden_system(xi, y, n):
    """
    Defines the LE equation as a system of first-order ODEs.
    y[0] = theta
    y[1] = d_theta/d_xi (phi)
    """
    theta, phi = y

    # To prevent issues with negative theta when n is fractional
    theta_term = np.power(max(theta, 0), n)

    # d_theta/d_xi = phi
    # d_phi/d_xi = -theta^n - (2/xi)*phi
    d_theta = phi

    if xi == 0:
        d_phi = 0  # Symmetry boundary condition
    else:
        d_phi = -theta_term - (2 / xi) * phi

    return [d_theta, d_phi]


def solve_le(n, xi_max=20, step=0.01):
    # Initial conditions using Taylor expansion near xi=0 to avoid singularity
    xi0 = 1e-4
    theta0 = 1 - (xi0 ** 2) / 6 + (n * xi0 ** 4) / 120
    phi0 = -xi0 / 3 + (n * xi0 ** 3) / 30

    # Event to stop when theta reaches 0 (the surface of the star)
    def surface_event(xi, y, n):
        return y[0]

    surface_event.terminal = True
    surface_event.direction = -1

    sol = solve_ivp(
        lane_emden_system,
        [xi0, xi_max],
        [theta0, phi0],
        args=(n,),
        events=surface_event,
        max_step=step
    )
    return sol.t, sol.y[0]


# Example: Generate profiles for n=1.5 (Convective) and n=3.0 (Radiative)
xi_15, theta_15 = solve_le(1.5)
xi_30, theta_30 = solve_le(3.0)

plt.plot(xi_15 / xi_15[-1], theta_15, label='n=1.5 (Convective)')
plt.plot(xi_30 / xi_30[-1], theta_30, label='n=3.0 (Radiative)')
plt.xlabel('Normalized Radius (r/R)')
plt.ylabel('Dimensionless Density (theta)')
plt.legend()
plt.title('Theoretical Polytropic Profiles')
plt.show()