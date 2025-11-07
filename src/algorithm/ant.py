import time
import math
import random
import numpy as np

# 计算两点间距离的函数
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 初始化距离矩阵
def create_distance_matrix(locations):
    n = len(locations)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = distance(locations[i][1], locations[j][1])
    return distance_matrix

# 蚁群算法
def ant_colony_optimization(distance_matrix, n_ants=10, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5):
    start_time = time.time()
    
    n = len(distance_matrix)
    pheromones = np.ones((n, n)) / n
    best_route = None
    best_distance = float('inf')

    for _ in range(n_iterations):
        all_routes = []
        all_distances = []
        
        for _ in range(n_ants):
            route = []
            visited = set()
            current_city = random.randint(0, n - 1)
            route.append(current_city)
            visited.add(current_city)
            
            for _ in range(n - 1):
                probabilities = []
                for next_city in range(n):
                    if next_city not in visited:
                        pheromone = pheromones[current_city][next_city] ** alpha
                        heuristic = (1.0 / distance_matrix[current_city][next_city]) ** beta
                        probabilities.append((pheromone * heuristic, next_city))
                
                total_prob = sum([prob[0] for prob in probabilities])
                probabilities = [(prob[0] / total_prob, prob[1]) for prob in probabilities]
                probabilities.sort(reverse=True)
                
                rand_prob = random.random()
                cumulative_prob = 0.0
                for prob, next_city in probabilities:
                    cumulative_prob += prob
                    if rand_prob <= cumulative_prob:
                        route.append(next_city)
                        visited.add(next_city)
                        current_city = next_city
                        break

            all_routes.append(route)
            total_distance = sum([distance_matrix[route[i - 1]][route[i]] for i in range(n)])
            all_distances.append(total_distance)

            if total_distance < best_distance:
                best_distance = total_distance
                best_route = route

        for i in range(n):
            for j in range(n):
                pheromones[i][j] *= (1.0 - evaporation_rate)

        for route, total_distance in zip(all_routes, all_distances):
            for i in range(n):
                pheromones[route[i - 1]][route[i]] += 1.0 / total_distance
                
    end_time = time.time()
    exetime = end_time - start_time
    return best_route, best_distance, exetime