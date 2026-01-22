import numpy as np
import scipy

def simulate_lorenz(num_steps = 50000, burn_in=10000):
    def lorenz_dot(xyz, *, s=10, r=28, b=2.667):
        """
        Parameters
        ----------
        xyz : array-like, shape (3,)
           Point of interest in three-dimensional space.
        s, r, b : float
           Parameters defining the Lorenz attractor.

        Returns
        -------
        xyz_dot : array, shape (3,)
           Values of the Lorenz attractor's partial derivatives at *xyz*.
        """
        x, y, z = xyz
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])

    DT = 0.01

    time_points = np.array([i*DT for i in range(burn_in,num_steps+burn_in)])

    current = np.array([0.1, 0.1, 0.1])
    integrated = scipy.integrate.solve_ivp(lambda t,y:lorenz_dot(y), y0=current, t_span=(0, time_points[-1]), t_eval=time_points, max_step=1e-2, rtol=1e-3)

    result = np.zeros((num_steps, 3))
    result[:,0] = integrated.y[0,:]
    result[:,1] = integrated.y[1,:]
    result[:,2] = integrated.y[2,:]
    #result[:,3] = np.array([i*DT for i in range(num_steps)])

    return result

def add_white_noise(data, signal_to_noise_ratio):
    noise = np.random.normal(0, 1, data.shape)
    signal_power = np.mean(np.square(data - np.mean(data, axis=0)))
    noise_power = np.mean(np.square(noise - np.mean(noise, axis=0)))
    desired_noise_power = signal_power * 10**(-signal_to_noise_ratio / 10)
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    return data + scaling_factor * noise