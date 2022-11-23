import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from State import State

# Configure the system's initial conditions
@jit(nopython=True)
def conf_ini(N, boxsize, temper):
    L = int(np.sqrt(N))

    x = np.zeros(N)
    y = np.zeros(N)

    vx = np.zeros(N)
    vy = np.zeros(N)

    ax = np.zeros(N)
    ay = np.zeros(N)

    pot = np.zeros(N)

    cont = 0
    for i in range(L):
        for j in range(L):
            # distribute the particles in a square net
            x[cont] = (i + 0.5) - L / 2
            y[cont] = (j + 0.5) - L / 2

            # vx[cont] = x[cont]/np.abs(x[cont])
            # vy[cont] = y[cont]/np.abs(y[cont])

            # picks a speed direction uniformly
            phi = np.random.uniform(0, 2 * np.pi)
            vx[cont] = np.cos(phi)
            vy[cont] = np.sin(phi)

            cont += 1

    # set positions to inside the box
    x = x / L * boxsize
    y = y / L * boxsize

    # normalize the speeds according to the equipartition theorem
    prov = np.sqrt((2.0 - 2.0 / N) * temper)

    vx = vx * prov
    vy = vy * prov

    # Nullifies the total momentum
    prov = np.sum(vx)
    vx = vx - prov / N
    prov = np.sum(vy)
    vy = vy - prov / N

    return State(x, y, vx, vy, ax, ay, pot)


@jit(nopython=True)
def copy_state(state):
    return State(
        state.x.copy(),
        state.y.copy(),
        state.vx.copy(),
        state.vy.copy(),
        state.ax.copy(),
        state.ay.copy(),
        state.pot.copy(),
    )


# Add random perturbation to the positions
@jit(nopython=True)
def perturbate(state, part_size, amount, boxsize, N):
    new_state = copy_state(state)
    for i in range(N):
        phi = np.random.uniform(0, 2 * np.pi)

        new_state.x[i] = calc_coord(
            new_state.x[i] + np.cos(phi) * part_size * amount, boxsize
        )

        new_state.y[i] = calc_coord(
            new_state.y[i] + np.sin(phi) * part_size * amount, boxsize
        )

    return new_state


# Calculate the distance between particles i and j
@jit(nopython=True)
def calc_dist(i, j, state, boxsize):
    x = state.x
    y = state.y
    xij = x[i] - x[j]
    yij = y[i] - y[j]

    xij = calc_coord(xij, boxsize)
    yij = calc_coord(yij, boxsize)

    r2 = xij**2 + yij**2

    return r2, xij, yij


# Create the verlet list
@jit(nopython=True)
def verlet_list(state, rv, boxsize):
    x = state.x

    N = len(x)

    nviz = np.zeros(N, np.int64)
    viz = np.empty(0, np.int64)

    # count number of neighbors
    cont = 0

    for i in range(N):
        for j in range(i + 1, N):  # loop over possible neighbors
            r2, _, _ = calc_dist(i, j, state, boxsize)

            if r2 < rv:
                cont += 1
                viz = np.append(viz, [j])

            nviz[i] = cont

    return nviz, viz


# Calculate the potential at a distance r
@jit(nopython=True)
def lennard_jones_pot(r, depth, part_size):
    return 4 * depth * ((part_size / r) ** (12) - (part_size / r ** (6)))


# Calculate the force at a distance r
@jit(nopython=True)
def lennard_jones_force(r, xij, yij, depth, part_size):
    f = depth * (48 * part_size ** (12) / r ** (14) - 24 * part_size ** (6) / r ** (8))
    fx = f * xij
    fy = f * yij
    return fx, fy


# Apply periodic boundary conditions
@jit(nopython=True)
def calc_coord(dist, boxsize):
    return dist - boxsize * np.rint(dist / boxsize)


# Velocity-verlet position step
@jit(nopython=True)
def position_step(rt, vt, at, dt, boxsize):
    r_step = rt + dt * vt + dt**2 * at / 2

    return calc_coord(r_step, boxsize)


# Velocity-verlet velocity step
@jit(nopython=True)
def velocity_step(vt, at, a_step, dt):
    return vt + dt * (at + a_step) / 2


@jit(nopython=True)
def time_step(state, nviz, viz, dt, boxsize, N, rv, rcut, part_size, depth, mass):
    old_state = copy_state(state)

    # verlet list should only be updated when some particle has
    # enough speed to enter/leave another particle's potential
    vmax2 = np.amax(state.vx**2 + state.vy**2)
    update_verl = 2 * vmax2 * dt < rv - rcut

    if update_verl:
        nviz, viz = verlet_list(state, rv, boxsize)

    state.ax = np.zeros(N)
    state.ay = np.zeros(N)

    state.pot = np.zeros(N)

    for i in range(N):
        nviz_i = nviz[i]
        viz_i = viz[nviz[i - 1] : nviz_i]

        # the next variables are the values in the previous time frame
        old_x = old_state.x[i]
        old_y = old_state.y[i]
        old_vx = old_state.vx[i]
        old_vy = old_state.vy[i]
        old_ax = old_state.ax[i]
        old_ay = old_state.ay[i]

        # the next 3 variables will take into account the
        # values at the new time frame t + dt
        axi = state.ax[i]
        ayi = state.ay[i]
        poti = state.pot[i]

        # update position
        state.x[i] = position_step(old_x, old_vx, old_ax, dt, boxsize)
        state.y[i] = position_step(old_y, old_vy, old_ay, dt, boxsize)

        for j in viz_i:
            r2, xij, yij = calc_dist(i, j, state, boxsize)

            if np.sqrt(r2) < 0.75 * part_size:
                raise Exception("Position threshold reached")

            apply_potential = r2 < rcut

            if apply_potential:
                r = np.sqrt(r2)
                potential = lennard_jones_pot(r, depth, part_size)
                fx, fy = lennard_jones_force(r, xij, yij, depth, part_size)

                axij = fx / mass
                ayij = fy / mass

                axi += axij
                ayi += ayij

                state.pot[j] += potential
                poti += potential

                state.ax[j] -= axij
                state.ay[j] -= ayij

        state.vx[i] = velocity_step(old_vx, old_ax, axi, dt)
        state.vy[i] = velocity_step(old_vy, old_ay, ayi, dt)

        state.ax[i] = axi
        state.ay[i] = ayi

        state.pot[i] = poti

    return state, nviz, viz


@jit(nopython=True)
def calc_pos_deviation(ref_state, pert_state, N):
    sum = 0
    for i in range(N):
        pert_pos = np.sqrt(pert_state.x[i] ** 2 + pert_state.y[i] ** 2)
        ref_pos = np.sqrt(ref_state.x[i] ** 2 + ref_state.y[i] ** 2)

        sum += (pert_pos - ref_pos) ** 2

    avg_deviation = sum / N

    return avg_deviation


@jit(nopython=True)
def calc_energy(state, N, mass):
    kin_energy = 0
    pot_energy = 0
    total = 0

    for i in range(N):
        v2 = state.vx[i] ** 2 + state.vy[i] ** 2
        kin_energy += mass * v2 / 2

        pot_energy += state.pot[i]

    pot_energy = pot_energy / 2

    total = kin_energy + pot_energy
    return kin_energy, pot_energy, total


def plot_dev_over_time(dev_over_time, pert_amount):
    fig, ax = plt.subplots(1, 1)

    ax.set_yscale("log")

    for index, amount in enumerate(pert_amount):
        ax.plot(dev_over_time[index], label=f"{amount} σ")

    ax.set_ylabel("log(Δr²)")
    ax.set_xlabel("Number of iterations")

    ax.set_title("Avg position deviation over time")
    ax.legend(title="Initial perturbation")

    plt.show()


def plot_energy_over_time(kin, pot, total):
    energy_std = np.std(total)

    fig, ax = plt.subplots(1, 1)

    ax.plot(kin, label="Kinetic")
    ax.plot(pot, label="Potential")
    ax.plot(total, label=f"Total (STD: {energy_std:.2e})")

    ax.set_ylabel("Energy")
    ax.set_xlabel("Number of iterations")

    ax.set_title("Energy evolution over time")
    ax.legend()

    plt.show()
