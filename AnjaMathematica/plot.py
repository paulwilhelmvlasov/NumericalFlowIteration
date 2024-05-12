import numpy as np
import matplotlib.pyplot as plt
from function import compute_function_py

# Bereich für x und y definieren
x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.linspace(-2*np.pi, 2*np.pi, 100)

# Festes z
z = 0.5  

# Funktionswerte berechnen
F = np.zeros((len(x), len(y)))  # leeres Array für die Funktionswerte
for i in range(len(x)):
    for j in range(len(y)):
        F[i, j] = compute_function_py(x[i], y[j], z)

# Plot erstellen
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, F, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(X,Y,Z)')
plt.show()
