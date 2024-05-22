import numpy as np
import matplotlib.pyplot as plt
from function import compute_function_py

# Bereich für x und y definieren
x = np.linspace(-8, 10, 100)
y = np.linspace(-8, 10, 100)

# Festes z
z = 0.5  

# Funktionswerte berechnen
F = np.zeros((len(x), len(y)))  # leeres Array für die Funktionswerte
for i in range(len(x)):
    for j in range(len(y)):
        F[i, j] = compute_function_py(x[i], y[j], z) - 1.0 / (4.0 * np.pi * np.sqrt(3.0)) * (np.exp(-(pow(x[i] + np.sqrt(2), 2) + pow(y[j], 2) + pow(z + np.sqrt(3.0/2.0), 2))) + np.exp(-(pow(x[i] - 2*np.sqrt(2), 2) + pow(y[j], 2) + pow(z - np.sqrt(3.0/2.0), 2))) + np.exp(-(pow(x[i], 2) + pow(y[j] + 3*np.sqrt(2), 2) + pow(z, 2))) + np.exp(-(pow(x[i], 2) + pow(y[j] - 4*np.sqrt(2), 2) + pow(z, 2))))

# Plot erstellen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, F, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(X,Y,Z)')
plt.show()
