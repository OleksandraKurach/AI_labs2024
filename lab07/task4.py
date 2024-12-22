import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Завантаження даних із файлу data_clustering.txt
data = np.loadtxt('data_clustering.txt', delimiter=',')  # Передбачається, що числа розділені комами

# Обчислення різниць між координатами (у даному разі залежить від формату даних)
# Якщо це двовимірні дані, їх можна кластеризувати без обчислення різниць
normalized_data = normalize(data)

# Створення моделі кластеризації
affinity_model = AffinityPropagation(random_state=0)
affinity_model.fit(normalized_data)

# Отримання результатів
cluster_centers_indices = affinity_model.cluster_centers_indices_
labels = affinity_model.labels_

# Виведення результатів
n_clusters = len(cluster_centers_indices)
print(f"Кількість кластерів: {n_clusters}")
print("Кластери та їх центри:")
for i, center in enumerate(cluster_centers_indices):
    print(f"Кластер {i+1}: Центр - {normalized_data[center]}")

# Візуалізація результатів
plt.figure(figsize=(10, 6))
colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

for k, col in zip(range(n_clusters), colors):
    class_members = labels == k
    cluster_center = normalized_data[cluster_centers_indices[k]]
    plt.scatter(normalized_data[class_members, 0], normalized_data[class_members, 1], color=col, label=f'Кластер {k+1}')
    plt.scatter(cluster_center[0], cluster_center[1], s=200, color=col, edgecolors='k', marker='X')

plt.title('Affinity Propagation Clustering')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.legend()
plt.grid(True)
plt.show()
