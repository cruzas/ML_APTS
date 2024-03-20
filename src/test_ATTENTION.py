import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class TrustRegionMethod(torch.optim.Optimizer):
    def __init__(self, params, delta_min=0.001, delta_max=1, delta=0.1):
        super(TrustRegionMethod, self).__init__(params,{})
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta = delta
        self.memory = []

    def attention(self, g):
        s = -g / torch.norm(g)
        self.memory.append(s)
        if len(self.memory) > 2: # keep only the current gradient and the previous one
            self.memory.pop(0)
        if len(self.memory) < 2:
            return s*self.delta
        # here we make an attention mechanism
        # print(self.memory[0])
        
        W = self.memory[0] @ self.memory[1].T
        print(W)
        if W < 0:
            s = -W/2 * self.memory[0] + self.memory[1]
            
        
        return (self.delta / torch.norm(s)) * s
        
            
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure(self.param_groups[0]['params'][0])

        assert len(self.param_groups[0]['params']) == 1, "This optimizer only supports a single parameter group."

        ATTENTION = 0
        if ATTENTION == 0:
            s = (self.delta / torch.norm(self.param_groups[0]['params'][0].grad.data)) * -self.param_groups[0]['params'][0].grad.data
        else:
            s = self.attention(self.param_groups[0]['params'][0].grad.data)
        p_old = self.param_groups[0]['params'][0].data.clone()
        old_loss = loss
        self.param_groups[0]['params'][0].data += s
        
        new_loss = closure(self.param_groups[0]['params'][0])
        rho = (old_loss - new_loss) / (old_loss - (old_loss + s @ self.param_groups[0]['params'][0].grad.data))

        if rho < 0.25:
            self.delta = max(0.25 * self.delta, self.delta_min)
            self.param_groups[0]['params'][0].data.copy_(p_old)
        elif rho > 0.75:
            self.delta = min(2 * self.delta, self.delta_max)
            

        return loss





            
def function_to_optimize(x, y, a=1, b=10): # Rosenbrock function (minimum at (1, 1))
    return (a - x)**2 + b * (y - x**2)**2

# Initial parameters
params = torch.tensor([0.74, -1.0], requires_grad=True)

# Optimizer
optimizer = TrustRegionMethod([params])

# For plotting
x_values, y_values, z_values = [params[0].item()], [params[1].item()], [function_to_optimize(params[0], params[1]).item()]

# Optimization loop
for _ in range(100):
    optimizer.zero_grad()
    loss = function_to_optimize(params[0], params[1])
    loss.backward()
    optimizer.step(lambda params: function_to_optimize(params[0], params[1]))
    
    # Logging values for plotting
    x_values.append(params[0].item())
    y_values.append(params[1].item())
    z_values.append(function_to_optimize(params[0], params[1]).item())
    # print(params)

# Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
X = np.linspace(-1, 1, 100)
Y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(X, Y)
Z = function_to_optimize(X, Y)
# ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.5)
# ax.scatter(x_values, y_values, z_values, color='r', s=10, label='Optimizer path')
# # plot the minima 
# ax.scatter(1, 1, function_to_optimize(1, 1), color='g', s=100, label='Global minimum')
# ax.legend()

# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# plt.title('Optimizer Path on Function Surface')
# plt.show()



fig2 = plt.figure()
plt.plot(z_values)
plt.xlabel('Iterations')
plt.ylabel('Loss')


fig3 = plt.figure()
cp = plt.contour(X, Y, Z, colors='blue', alpha=0.5)
plt.scatter(x_values, y_values, color='r', s=10, label='Optimizer path')
plt.plot(x_values, y_values, color='r', linewidth=2) # Path line
plt.scatter(1, 1, color='g', s=100, label='Global minimum') # Global minimum
plt.legend()
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Contour Plot with Optimizer Path')
plt.show()

#print distance between the last point and the global minimum
print(np.linalg.norm(np.array([x_values[-1], y_values[-1]]) - np.array([1, 1])))