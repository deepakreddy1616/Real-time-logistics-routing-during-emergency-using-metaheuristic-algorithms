import os
from dotenv import load_dotenv

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:34:36 2025

@author: Deepak Reddy Chelladi
"""
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from math import radians, sin, cos, sqrt, atan2
from IPython.display import clear_output

# Configuration
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
if not OPENWEATHER_API_KEY:
    raise ValueError("⚠️ OPENWEATHER_API_KEY not found! Please create .env file with your API key")

CITY_NAME = "Hyderabad, Telangana, India"
VEHICLE_CAPACITY = 50
NUM_WOLVES = 20
MAX_ITER = 50
EARLY_STOPPING_ROUNDS = 10

# Initialize OpenStreetMap graph
def initialize_map_graph():
    print("Downloading street network data...")
    G = ox.graph_from_place(CITY_NAME, network_type='drive')
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    print(f"Graph with {len(G.nodes())} nodes and {len(G.edges())} edges loaded")
    return G

# Get real-time weather data
def get_weather_data(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'temp': data['main']['temp'],
                'conditions': data['weather'][0]['main'],
                'wind_speed': data['wind']['speed'],
                'rain': data.get('rain', {}).get('1h', 0)
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    return None

# Calculate weather impact factor
def calculate_weather_impact(weather):
    if not weather:
        return 1.0
    impact = 1.0
    if weather['conditions'] in ['Rain', 'Snow', 'Thunderstorm']:
        impact *= 1.1
    if weather['rain'] > 5:
        impact *= 1.15
    if weather['wind_speed'] > 10:
        impact *= 1.05
    return min(max(impact, 0.8), 1.2)

# Get coordinates for nodes
def get_node_coordinates(G, node_ids):
    return [(G.nodes[node]['x'], G.nodes[node]['y']) for node in node_ids]

# Emergency Logistics Problem
class EmergencyRoutingProblem:
    def __init__(self, G, depot, locations, demands, vehicle_capacity):
        self.G = G
        self.depot = depot
        self.locations = locations
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.node_list = [depot] + locations
        self.weather_impacts = self.calculate_all_weather_impacts()
        
    def calculate_all_weather_impacts(self):
        impacts = []
        for node in self.node_list:
            lat, lon = self.G.nodes[node]['y'], self.G.nodes[node]['x']
            weather = get_weather_data(lat, lon)
            impacts.append(calculate_weather_impact(weather))
        return impacts
    
    def get_route_time(self, route):
        total_time = 0
        for i in range(len(route)-1):
            from_node = route[i]
            to_node = route[i+1]
            try:
                path = nx.shortest_path(self.G, from_node, to_node, weight='travel_time')
                edge_times = [self.G.edges[path[i], path[i+1], 0]['travel_time'] 
              for i in range(len(path)-1)]
                weather_impact = self.weather_impacts[self.node_list.index(from_node)]
                total_time += sum(edge_times) * weather_impact
            except (nx.NetworkXNoPath, KeyError):
                return float('inf')
        
        try:
            return_path = nx.shortest_path(self.G, route[-1], self.depot, weight='travel_time')
            edge_times = [self.G.edges[return_path[i], return_pathpath[i+1], 0]['travel_time'] 
              for i in range(len(return_pathpath)-1)] 
            weather_impact = self.weather_impacts[self.node_list.index(route[-1])]
            total_time += sum(edge_times) * weather_impact
        except (nx.NetworkXNoPath, KeyError):
            return float('inf')
        
        return total_time
    
    def fitness(self, route):
        # Ensure route starts at depot (index 0)
        if route[0] != 0:
            return float('inf')
            
        # Convert continuous positions to discrete nodes
        discrete_route = []
        for node in route:
            idx = min(int(round(node)), len(self.node_list)-1)
            idx = max(idx, 0)
            discrete_route.append(self.node_list[idx])
        
        # Check for duplicate nodes (except depot)
        if len(set(discrete_route[1:])) != len(discrete_route)-1:
            return float('inf')
            
        # Calculate total demand
        total_demand = sum(self.demands[self.locations.index(node)] 
                          for node in discrete_route if node in self.locations)
        
        # Calculate total travel time
        total_time = self.get_route_time(discrete_route)
        
        # Penalize capacity violations
        if total_demand > self.vehicle_capacity:
            total_time *= 1.5
            
        return total_time

# Refined Grey Wolf Optimization
class RealTimeGWO:
    def __init__(self, problem, num_wolves, max_iter, early_stopping_rounds=10):
        self.problem = problem
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.alpha_pos = None
        self.alpha_score = float('inf')
        self.beta_pos = None
        self.beta_score = float('inf')
        self.delta_pos = None
        self.delta_score = float('inf')
        self.wolves = self.initialize_wolves()
        self.convergence_curve = []
        
    def initialize_wolves(self):
        wolves = []
        num_nodes = len(self.problem.node_list)
        for _ in range(self.num_wolves):
            # Start with depot (0), then random permutation of other nodes
            route = [0] + list(np.random.permutation(range(1, num_nodes)))
            wolves.append(np.array(route, dtype=float))
        return wolves
    
    def update_position(self, wolf, alpha_pos, beta_pos, delta_pos, a):
        # Ensure all positions have same length
        if alpha_pos is None or beta_pos is None or delta_pos is None:
            return wolf
            
        # Make sure all vectors have same length
        target_length = len(wolf)
        alpha_pos = alpha_pos[:target_length]
        beta_pos = beta_pos[:target_length]
        delta_pos = delta_pos[:target_length]
        
        # Update position calculations
        A1 = 2 * a * np.random.random() - a
        C1 = 2 * np.random.random()
        D_alpha = np.abs(C1 * alpha_pos - wolf)
        X1 = alpha_pos - A1 * D_alpha
        
        A2 = 2 * a * np.random.random() - a
        C2 = 2 * np.random.random()
        D_beta = np.abs(C2 * beta_pos - wolf)
        X2 = beta_pos - A2 * D_beta
        
        A3 = 2 * a * np.random.random() - a
        C3 = 2 * np.random.random()
        D_delta = np.abs(C3 * delta_pos - wolf)
        X3 = delta_pos - A3 * D_delta
        
        new_position = (X1 + X2 + X3) / 3
        return new_position
    
    def optimize(self):
        no_improvement_count = 0
        
        for iter in range(self.max_iter):
            a = 2 - 2 * (iter / self.max_iter)
            
            # Update positions
            for i in range(self.num_wolves):
                self.wolves[i] = self.update_position(
                    self.wolves[i], 
                    self.alpha_pos if self.alpha_pos is not None else self.wolves[i],
                    self.beta_pos if self.beta_pos is not None else self.wolves[i],
                    self.delta_pos if self.delta_pos is not None else self.wolves[i],
                    a
                )
                
                # Ensure depot remains at start
                self.wolves[i][0] = 0
                
                # Clip values to valid node indices
                self.wolves[i] = np.clip(self.wolves[i], 0, len(self.problem.node_list)-1)
            
            # Evaluate fitness
            for i in range(self.num_wolves):
                fitness = self.problem.fitness(self.wolves[i])
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.wolves[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.wolves[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.wolves[i].copy()
            
            # Store best score
            self.convergence_curve.append(self.alpha_score)
            
            # Early stopping
            if iter > 0 and self.convergence_curve[-1] >= self.convergence_curve[-2]:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
                
            if no_improvement_count >= self.early_stopping_rounds:
                break
                
            # Progress update
            clear_output(wait=True)
            print(f"Iteration {iter+1}/{self.max_iter}")
            print(f"Best travel time: {self.alpha_score:.2f} ms")
        
        return self.alpha_pos, self.alpha_score, self.convergence_curve


# Visualization
def plot_results(G, problem, best_route):
    # Convert continuous route to discrete nodes
    discrete_route = []
    for node in best_route:
        idx = min(int(round(node)), len(problem.node_list)-1)
        idx = max(idx, 0)
        discrete_route.append(problem.node_list[idx])
    
    # Get full path with all edges
    full_path = []
    for i in range(len(discrete_route)-1):
        segment = nx.shortest_path(G, discrete_route[i], discrete_route[i+1], weight='travel_time')
        full_path.extend(segment[:-1])
    full_path.append(discrete_route[-1])
    
    # Plot
    fig, ax = ox.plot_graph_route(
        G, full_path, route_linewidth=6, 
        node_size=0, bgcolor='white', 
        orig_dest_size=100)
    
    # Mark depot and locations
    depot_coords = (G.nodes[problem.depot]['x'], G.nodes[problem.depot]['y'])
    location_coords = [(G.nodes[loc]['x'], G.nodes[loc]['y']) for loc in problem.locations]
    
    plt.scatter(depot_coords[0], depot_coords[1], c='green', s=200, 
                marker='s', edgecolor='black', label='Depot', zorder=5)
    for i, (x, y) in enumerate(location_coords):
        plt.scatter(x, y, c='red', s=150, 
                   marker='o', edgecolor='black', 
                   label=f'Location {i+1}' if i == 0 else "", zorder=5)
    
    plt.title(f"Optimized Emergency Route\nTotal Time: {problem.get_route_time(discrete_route):.2f} ms")
    plt.legend()
    plt.show()

# Main execution
def main():
    # Initialize map
    G = initialize_map_graph()
    
    # Select nodes (depot + locations)
    all_nodes = list(G.nodes())
    depot = all_nodes[0]
    locations = all_nodes[1:6]  # First 5 nodes after depot
    
    # Generate random demands
    demands = np.random.randint(1, 10, size=len(locations))
    
    # Create problem instance
    problem = EmergencyRoutingProblem(G, depot, locations, demands, VEHICLE_CAPACITY)
    
    # Run optimization
    gwo = RealTimeGWO(problem, NUM_WOLVES, MAX_ITER, EARLY_STOPPING_ROUNDS)
    best_route, best_score = gwo.optimize()
    
    # Results
    print("\nOptimization Complete!")
    print(f"Best Route Travel Time: {best_score:.2f} ms")
    
    # Visualization
    plot_results(G, problem, best_route)
    
    return best_route, best_score

if __name__ == "__main__":
    best_route, best_time = main()

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import linregress

class GWOPerformanceTester:
    def __init__(self, num_tests=50, max_nodes=30):
        self.metrics = [
            'Test_ID', 'Num_Nodes', 'Travel_Time', 'Congestion_Penalty',
            'Safety_Index', 'Fitness_Score', 'Convergence_Rate',
            'Route_Reliable', 'Execution_Time', 'Capacity_Utilization'
        ]
        self.results = pd.DataFrame(columns=self.metrics)
        self.congestion_data = {}  # Stores historical congestion data
        self.safety_data = {}      # Stores road safety information
        
    def calculate_congestion_penalty(self, route, G):
        """Calculate congestion penalty based on time of day and historical data"""
        penalty = 0
        current_hour = time.localtime().tm_hour
        
        for i in range(len(route)-1):
            edge_data = G.get_edge_data(route[i], route[i+1])
            if edge_data:
                # Base congestion (higher during rush hours 7-9am, 4-7pm)
                base_congestion = 1.2 if (7 <= current_hour <= 9) or (16 <= current_hour <= 19) else 1.0
                # Historical congestion (if available)
                edge_key = f"{route[i]}-{route[i+1]}"
                hist_congestion = self.congestion_data.get(edge_key, 1.0)
                penalty += base_congestion * hist_congestion
        return penalty / len(route) if route else 1.0
    
    def calculate_safety_index(self, route, G):
        """Calculate safety index based on road types and conditions"""
        safety_score = 0
        road_types = {
            'motorway': 0.9, 'trunk': 0.8, 'primary': 0.7,
            'secondary': 0.6, 'tertiary': 0.5, 'residential': 0.4
        }
        
        for i in range(len(route)-1):
            edge_data = G.get_edge_data(route[i], route[i+1])
            if edge_data:
                road_type = edge_data[0].get('highway', 'residential')
                base_safety = road_types.get(road_type, 0.5)
                # Consider weather impact
                weather_impact = self.problem.weather_impacts[self.problem.node_list.index(route[i])]
                safety_score += base_safety * (1/weather_impact)
        return safety_score / len(route) if route else 0
    
    def calculate_convergence_rate(self, convergence_curve):
        """Calculate how quickly the algorithm converges"""
        if len(convergence_curve) < 2:
            return 0
        normalized_curve = (convergence_curve - np.min(convergence_curve)) / \
                         (np.max(convergence_curve) - np.min(convergence_curve))
        x = np.arange(len(normalized_curve))
        slope, _, _, _, _ = linregress(x, normalized_curve)
        return -slope  # More negative = faster convergence
    
    def run_performance_test(self, num_tests=50):
        """Execute the comprehensive performance testing"""
        G = initialize_map_graph()
        all_nodes = list(G.nodes())
        
        # Initialize congestion and safety data (simulated)
        for u, v, data in G.edges(data=True):
            edge_key = f"{u}-{v}"
            self.congestion_data[edge_key] = np.random.uniform(0.8, 1.5)
        
        for test_id in tqdm(range(1, num_tests + 1), desc="Performance Testing"):
            try:
                num_nodes = np.random.randint(5, min(30, len(all_nodes)))
                depot = all_nodes[0]
                locations = all_nodes[1:num_nodes]
                demands = np.random.randint(1, 10, size=len(locations))
                
                self.problem = EmergencyRoutingProblem(G, depot, locations, demands, VEHICLE_CAPACITY)
                
                start_time = time.time()
                gwo = RealTimeGWO(self.problem, NUM_WOLVES, MAX_ITER, EARLY_STOPPING_ROUNDS)
                best_route, best_score, convergence_curve = gwo.optimize()
                exec_time = time.time() - start_time
                
                # Convert route to node IDs
                discrete_route = []
                if best_route is not None:
                    discrete_route = [self.problem.node_list[min(int(round(node)), len(self.problem.node_list)-1)] 
                                    for node in best_route]
                
                # Calculate all metrics
                metrics = {
                    'Test_ID': test_id,
                    'Num_Nodes': num_nodes,
                    'Travel_Time': self.problem.get_route_time(discrete_route) if discrete_route else None,
                    'Congestion_Penalty': self.calculate_congestion_penalty(discrete_route, G) if discrete_route else None,
                    'Safety_Index': self.calculate_safety_index(discrete_route, G) if discrete_route else None,
                    'Fitness_Score': best_score if best_score < float('inf') else None,
                    'Convergence_Rate': self.calculate_convergence_rate(convergence_curve) if convergence_curve else None,
                    'Route_Reliable': int(len(set(discrete_route[1:])) == len(discrete_route)-1) if discrete_route else 0,
                    'Execution_Time': exec_time,
                    'Capacity_Utilization': sum(demands)/VEHICLE_CAPACITY*100 if discrete_route else 0
                }
                
                self.results.loc[test_id] = metrics
                
            except Exception as e:
                print(f"\nTest {test_id} failed: {str(e)}")
                continue
                
        return self.results
    
    def analyze_results(self):
        """Generate comprehensive analysis and visualizations"""
        # Basic statistics
        analysis = {
            'Average_Travel_Time': self.results['Travel_Time'].mean(),
            'Average_Congestion': self.results['Congestion_Penalty'].mean(),
            'Average_Safety': self.results['Safety_Index'].mean(),
            'Success_Rate': self.results['Route_Reliable'].mean() * 100,
            'Avg_Execution_Time': self.results['Execution_Time'].mean(),
            'Avg_Convergence_Rate': self.results['Convergence_Rate'].mean()
        }
        
        # Scalability analysis
        scalability = self.results.groupby('Num_Nodes').agg({
            'Execution_Time': 'mean',
            'Fitness_Score': 'mean',
            'Route_Reliable': 'mean'
        }).reset_index()
        
        # Visualization
        plt.figure(figsize=(18, 12))
        
        # Metric distribution plots
        metrics_to_plot = ['Travel_Time', 'Congestion_Penalty', 'Safety_Index', 
                          'Execution_Time', 'Capacity_Utilization']
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 3, i)
            plt.hist(self.results[metric].dropna(), bins=20, alpha=0.7)
            plt.title(f'{metric} Distribution')
            plt.xlabel(metric)
            plt.ylabel('Frequency')
        
        # Scalability plot
        plt.subplot(2, 3, 6)
        plt.scatter(scalability['Num_Nodes'], scalability['Execution_Time'])
        plt.plot(scalability['Num_Nodes'], scalability['Execution_Time'], 'r-')
        plt.title('Computation Time vs Problem Size')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Execution Time (s)')
        
        plt.tight_layout()
        plt.show()
        
        # Print analysis summary
        print("\n=== Performance Analysis Summary ===")
        for k, v in analysis.items():
            print(f"{k.replace('_', ' ')}: {v:.2f}")
        
        return analysis, scalability

# Usage example:
tester = GWOPerformanceTester()
results = tester.run_performance_test(50)
analysis, scalability = tester.analyze_results()

# Save results
results.to_csv('gwo_performance_metrics.csv', index=False)
print("\nResults saved to 'gwo_performance_metrics.csv'")
