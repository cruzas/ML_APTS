import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the Rosenbrock function
def rosenbrock(x):
    return (1 - x[0])**2 + 1 * (x[1] - x[0]**2)**2

# Use SciPy's minimize function with Trust Region Constrained algorithm
initial_guess = [-1, -1]  # Starting point for the optimization
result = minimize(rosenbrock, initial_guess, method='trust-constr')

# Generate points for the contour plot
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2.5, 3, 1000)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

# Redefine the optimization with a callback function to record the path
path = []  # List to store the path

# Redefine the callback function to include the trust region radius
def record_path(x, state):
    # Append the position (x, y) and the trust region radius
    path.append((x, state.tr_radius))

# Redo the optimization with the updated callback function
path.clear()  # Clearing the existing path
# result = minimize(rosenbrock, initial_guess, method='trust-constr', callback=record_path)
result = minimize(rosenbrock, initial_guess, method='trust-constr', jac='2-point', hess=lambda x: np.zeros((2, 2)), callback=record_path)

# Extract position and trust region radius from the path for plotting
positions, radii = zip(*path)
positions = np.array(positions)  # Convert positions to numpy array for easier plotting

# Create the contour plot again
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contour, inline=1, fontsize=8)
plt.title('Rosenbrock Function with Trust Region Path and Radii')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1.5, 1.5)
plt.ylim(-2.5, 2.5)

positions = positions[:10]
radii = radii[:10]

# Overlay the optimization path on the contour plot
plt.plot(positions[:, 0], positions[:, 1], 'r.-', label='TR Path (order 1)', linewidth=3)
# plt.plot(1, 1, 'bo', label='Global Minimum (1, 1)')  # Global minimum point

# Draw circles to represent the trust region at each step
for pos, radius in zip(positions, radii):
    circle = plt.Circle(pos, radius, color='green', fill=False, linestyle='--', linewidth=2)
    plt.gca().add_artist(circle)


# second order TR path
path.clear()  # Clearing the existing path
result = minimize(rosenbrock, initial_guess, method='trust-constr', callback=record_path)
positions, radii = zip(*path)
positions = np.array(positions)  # Convert positions to numpy array for easier plotting
positions = positions[:10]
radii = radii[:10]
plt.plot(positions[:, 0], positions[:, 1], 'b.-', label='TR Path (order 2)', linewidth=3)
plt.plot(1, 1, 'mo', label='Minimum (1, 1)')  # Global minimum point

#here we now increase the text size:
plt.rcParams.update({'font.size': 20})
#now we increase the size of the title:
plt.title('Rosenbrock Function', fontsize=20)



plt.legend()
plt.tight_layout()
# Show the plot
plt.show()