import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class ASTR(torch.optim.Optimizer):
    def __init__(self, params, lr=1, epsilon=1e-2, mu=1/2): # mu = 1/2 for Adagrad
        super(ASTR, self).__init__(params, {})
        self.lr = lr
        self.epsilon = epsilon
        self.state_sum = [torch.zeros_like(p) for p in self.param_groups[0]['params']]
        self.mu = mu
    def step(self):
        for i,param in enumerate(self.param_groups[0]['params']):
            if param.grad is None:
                continue
            grad = param.grad.data
            print(torch.norm(grad))
            self.state_sum[i] += grad.pow(2)
            std = self.state_sum[i].add_(self.epsilon).pow(self.mu)
            param.data -= self.lr * grad / std


            
def function_to_optimize(x, y, a=1, b=10): # Rosenbrock function (minimum at (1, 1))
    return (a - x)**2 + b * (y - x**2)**2

# Initial parameters
params = torch.tensor([0.74, -1.0], requires_grad=True)

# Optimizer
optimizer = ASTR([params], lr=0.1)

# For plotting
x_values, y_values, z_values = [params[0].item()], [params[1].item()], [function_to_optimize(params[0], params[1]).item()]

# Optimization loop
for _ in range(1000):
    optimizer.zero_grad()
    loss = function_to_optimize(params[0], params[1])
    loss.backward()
    optimizer.step()
    
    # Logging values for plotting
    x_values.append(params[0].item())
    y_values.append(params[1].item())
    z_values.append(function_to_optimize(params[0], params[1]).item())
    print(params)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.linspace(-1, 1, 100)
Y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(X, Y)
Z = function_to_optimize(X, Y)
ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.5)
ax.scatter(x_values, y_values, z_values, color='r', s=10, label='Optimizer path')
# plot the minima 
ax.scatter(1, 1, function_to_optimize(1, 1), color='g', s=100, label='Global minimum')
ax.legend()

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title('Optimizer Path on Function Surface')
# plt.show()



fig2 = plt.figure()
plt.plot(z_values)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
