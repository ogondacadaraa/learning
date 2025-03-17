import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def y_a(t):
    return (6 * t**3 - 3 * t - 4) / (8 * np.sin(5 * t))

def y_b(t):
    return (3 * t - 2) / (4 * t) - (np.pi / 2) * t

# Generate t values for the plots
t_a = np.linspace(0.1, 0.25, 500)  # More points for smoother curves
t_b = np.linspace(1, 5, 500)      # More points for smoother curves

# Calculate y values
y_values_a = y_a(t_a)
y_values_b = y_b(t_b)

# Create the plots
plt.figure(figsize=(12, 6))  # Adjust figure size for better viewing

# Plot for y_a
plt.subplot(1, 2, 1)
plt.plot(t_a, y_values_a, label='y = (6t³ - 3t - 4) / (8sin(5t))')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Graph of y = (6t³ - 3t - 4) / (8sin(5t))')
plt.grid(True)
plt.legend()

# Plot for y_b
plt.subplot(1, 2, 2)
plt.plot(t_b, y_values_b, label='y = (3t - 2) / (4t) - (π/2)t')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Graph of y = (3t - 2) / (4t) - (π/2)t')
plt.grid(True)
plt.legend()

plt.tight_layout()  # Improve layout
plt.show()