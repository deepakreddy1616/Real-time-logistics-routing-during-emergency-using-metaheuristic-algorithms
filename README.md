# 🚨 Real-Time Emergency Logistics Routing using Grey Wolf Optimization

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

---

## 📌 Project Overview

During natural disasters like earthquakes and floods, **every minute matters**. Efficient delivery of relief supplies—medical equipment, food, rescue gear—can save thousands of lives.

This project implements a **real-time adaptive logistics optimization system** using the **Grey Wolf Optimization (GWO)** metaheuristic algorithm. It dynamically routes relief vehicles while adapting to:

- ✅ **Live weather conditions** (rain, wind, storms)
- ✅ **Real-time traffic updates** 
- ✅ **Road closures and hazards**
- ✅ **Vehicle capacity constraints**
- ✅ **Multiple delivery points**

**Research Paper:** Available in `paper/Project-Paper.pdf`

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Real-time Weather Integration** | Uses OpenWeatherMap API for live weather data |
| **Actual Street Networks** | Optimizes using OpenStreetMap data (Hyderabad) |
| **Dynamic Route Optimization** | Adapts to changing conditions every iteration |
| **Multi-objective Fitness** | Balances travel time, congestion, and safety |
| **Performance Testing** | 50+ automated test scenarios |
| **Capacity Management** | Respects vehicle load limits |

---

## 📊 Performance Results

| Metric | Result |
|--------|--------|
| **Route Reliability** | 97% success rate |
| **Average Execution Time** | 0.13 seconds (25 nodes) |
| **Improvement vs Baseline** | 15-30% better than greedy routing |
| **Convergence Speed** | Early stopping at ~30 iterations |
| **Vehicle Utilization** | 85-95% capacity usage |

---

## 🛠️ Installation & Setup

### Prerequisites

Before starting, you need:
- **Python 3.8+** ([Download here](https://www.python.org/downloads/))
- **Git** ([Download here](https://git-scm.com/download/win))
- **OpenWeatherMap API Key** (Free: https://openweathermap.org/api)

### Step-by-Step Installation

#### 1️⃣ Clone the Repository

