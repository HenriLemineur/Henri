import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

#Création de la fonction qui décrit le paysage
def f(x, y):
    return 20 * (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

#Définition des différentes constantes
num_alpinistes = 10
iterations_max = 100
no_progress_limit = 10
delta_t = 1
omega = 0.7
c1 = 1.2
c2 = 1.2
no_progress_count = 0

#Création des variables qui vont nous servire
positions_x = np.random.uniform(low=-5.12, high=5.12, size=num_alpinistes)
positions_y = np.random.uniform(low=-5.12, high=5.12, size=num_alpinistes)
best_personal_x = positions_x.copy()
best_personal_y = positions_y.copy()
best_global_x = np.min(positions_x)
best_global_y = np.min(positions_y)

vitesses_x = np.zeros(num_alpinistes)
vitesses_y = np.zeros(num_alpinistes)

#Création de la figure et des différents aspect de celle-ci
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)  
fig.colorbar(contour, shrink=0.5, aspect=5)

sc = ax.scatter([], [], label='Alpinistes', color='red')
sc_best_global = ax.scatter([], [], label='Meilleure position globale', color='blue')

ax.set_title('Positions des alpinistes')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.set_xlim(-5.12, 5.12)
ax.set_ylim(-5.12, 5.12)

iteration_text = ax.text(5.5, -4, '', fontsize=12, color='black')

#Fonction qui sert à créer les différentes images pour l'animation et qui sert donc de boucle
def update(frame):
    global positions_x, positions_y, best_personal_x, best_personal_y, best_global_x, best_global_y, no_progress_count, vitesses_x, vitesses_y

    #On prend r1 et r2 aléatoirement. Et ça change à chaque entrée dans la boucle
    r1 = np.random.uniform(0, 1, num_alpinistes)
    r2 = np.random.uniform(0, 1, num_alpinistes)
    
    #Calcul des vitesses
    vitesses_x = omega * vitesses_x + r1 * c1 * (best_personal_x - positions_x) + r2 * c2 * (best_global_x - positions_x)
    vitesses_y = omega * vitesses_y + r1 * c1 * (best_personal_y - positions_y) + r2 * c2 * (best_global_y - positions_y)

    #Calcul des positions
    positions_x += vitesses_x * delta_t
    positions_y += vitesses_y * delta_t
    
    positions_x = np.clip(positions_x, -5.12, 5.12)
    positions_y = np.clip(positions_y, -5.12, 5.12)
    
    #Boucle pour vérifier si chaque alpinist à amélioré sa positon et si oui lui attribuer la nouvelle
    altitudes = f(positions_x, positions_y)
    for i in range(num_alpinistes):
        if altitudes[i] < f(best_personal_x[i], best_personal_y[i]):
            best_personal_x[i] = positions_x[i]
            best_personal_y[i] = positions_y[i]
    
    #Condition qui vérifie si la position global des alpinistes c'est améliorée. Si oui on attribue la nouvelle valeurs a la position global, sinon on ajoute 1 au compteur
    min_altitude = np.min(altitudes)
    if min_altitude < f(best_global_x, best_global_y):
        best_global_x = positions_x[np.argmin(altitudes)]
        best_global_y = positions_y[np.argmin(altitudes)]
        no_progress_count = 0
    else:
        no_progress_count += 1

    #Pour afficher la bonne valeur d'itération sur l'image
    iteration_text.set_text(f"Itération : {frame+1}")

    
    sc.set_offsets(np.column_stack([positions_x, positions_y]))
    sc_best_global.set_offsets([[best_global_x, best_global_y]])

    #Vérification que les différentes conditions sont respectée pour continuer les itérations ou non
    if no_progress_count >= no_progress_limit:
        print(f"Arrêt après {frame+1} itérations sans progression.")
        ani.event_source.stop()  

    if frame+1 >= iterations_max:  
        print(f"Arrêt après {frame+1} itérations.")
        ani.event_source.stop()  
ani = FuncAnimation(fig, update, frames=iterations_max, interval=50) 


#plt.show()
#A réactiver si l'on ne fait pas tourner la partie d'après

#########################################################################################
#GRAPHIQUE PAYSAGE
#Cette partie ne sert que à afficher une image en 3D du paysage pour mieux se visualiser. 
#########################################################################################

def f(x, y):
    return 20 * (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))


x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
x, y = np.meshgrid(x, y)
z = f(x, y)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


surf = ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Altitude')


fig.colorbar(surf, shrink=0.5, aspect=5)

plt.title('Topographie du paysage montagneux')

#Actuelement plot les deux graphs en même temps. Il faut le désactiver et réactiver celui en ligne 105 pour n'avoir que l'animation
plt.show()

