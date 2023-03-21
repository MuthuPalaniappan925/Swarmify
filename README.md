
# Swarmify

This GitHub repository contains an implementation of the Particle Swarm Optimization (PSO) algorithm in Python. A PSO-based Optimization Framework with Real-time Visual Feedback.

PSO is a powerful optimization algorithm that is commonly used in engineering, computer science, and other fields to find the optimal solutions to complex problems.

## Overview

The PSO algorithm is a metaheuristic optimization technique that is based on the behavior of a swarm of particles. Each particle represents a candidate solution to the optimization problem, and its position in the search space is updated iteratively based on its own experience and the experience of its neighboring particles.

The PSO algorithm has several advantages over other optimization techniques, such as its simplicity, efficiency, and ability to find global optima in multi-modal search spaces. It has been successfully applied to a wide range of problems, including feature selection, image segmentation, clustering, and parameter tuning, among others.

## Requirements

- NumPy
- Matplotlib
- Pandas
- Streamlit
- imageio

pip install -r requirements.txt
## Customization

This implementation of the PSO algorithm provides several parameters that you can customize to improve its performance or adapt it to your specific problem. These parameters include:

- swarm_size: the number of particles in the swarm (default: 50)
- c1: the cognitive parameter, which controls the weight of the particle's own experience (default:0.50)
- c2: the social parameter, which controls the weight of the swarm's experience (default: 0.5)
- w: the inertia weight, which controls the particle's tendency to follow its previous velocity (default: 0.7298)
