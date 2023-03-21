##Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import imageio
import pandas as pd
import time

##Defining the Objective-Function
def f(x):
    return (1+(2*x)-(x**2))  #1+2x-x^2

fig, ax = plt.subplots()

# Set the x and y limits
x_min, x_max = -5, 15
y_min, y_max = -50, 100

##Defining the PSO
def PSO(f,n_particles=50,n_iterations=100,c1=2,c2=2,w=0.7):
    #Setting the Bounds
    x_min = -10
    x_max = 10
    particles = np.random.uniform(x_min, x_max, size=(n_particles, 1))
    velocities = np.zeros((n_particles, 1))
    best_positions = particles.copy()
    best_scores = f(best_positions)

    #Initialize the global best position and score
    global_best_position = best_positions[np.argmax(best_scores)]
    global_best_score = np.max(best_scores)

    #Running PSO for n_iterations
    for i in range(n_iterations):
        #Generating r1 and r2
        r1 = np.random.uniform(size=(n_particles, 1))
        r2 = np.random.uniform(size=(n_particles, 1))


        #Update the velocities and positions
        velocities = w * velocities \
                    + c1 * r1 * (best_positions - particles) \
                    + c2 * r2 * (global_best_position - particles)

        particles += velocities

        #Check if any particles have gone out of bounds
        particles = np.clip(particles, x_min, x_max)

        #Update the personal learning components
        scores = f(particles)
        improved_indices = scores > best_scores

        best_positions[improved_indices] = particles[improved_indices]
        best_scores[improved_indices] = scores[improved_indices]

        #Update the social learning components
        if np.max(best_scores) < global_best_score:
            global_best_score = np.max(best_scores)
            global_best_position = best_positions[np.argmax(best_scores)]

        st.set_option('deprecation.showPyplotGlobalUse', False)
        #Plot the particles
        ax.clear()
        # Plot the particles and best positions
        ax.plot(particles, f(particles), 'bo', label='particles')
        ax.plot(best_positions, f(best_positions), 'ro', label='best positions')
        ax.plot(global_best_position, global_best_score, 'go', label='global best')

        # Set the x and y limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Set the x and y labels
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')

        # Set the title
        ax.set_title(f'Iteration {i+1}/{n_iterations}')

        # Add the legend
        ax.legend()

        # Save the figure as a PNG file
        filename = f'iteration_{i+1}.png'
        plt.savefig(filename)
        
    images = []
    for i in range(n_iterations):
        filename = f'iteration_{i+1}.png'
        images.append(imageio.imread(filename))
    imageio.mimsave('PSO_Animation_Output.gif', images, fps=10)

    return global_best_position, global_best_score

# Define the streamlit app
def app():
    # Set the page title and layout
    #st.set_page_config(page_title="PSO App", page_layout="wide")
    count = 0
    # Define the sidebar
    st.sidebar.header("PSO Control Parameters")
    count+=1
    n_particles = st.sidebar.slider("Number of particles", min_value=10, max_value=100, value=50,key = count)
    count+=1
    n_iterations = st.sidebar.slider("Number of iterations", min_value=10, max_value=1000, value=100,key = count)
    count+=1
    c1 = st.sidebar.slider("C1 - acceleration coefficient", min_value=0.0, max_value=2.0, value=0.5,key = count)
    count+=1
    c2 = st.sidebar.slider("C2 - acceleration coefficient", min_value=0.0, max_value=2.0, value=0.5,key = count)
    count+=1
    w = st.sidebar.slider("Inertia weight", min_value=0.0, max_value=1.0, value=0.9,key = count)
    count+=1

    if st.sidebar.button('Perform PSO Technique'):
        gbp,gbs = PSO(f,n_particles,n_iterations,c1,c2,w)
        st.success(f"Global Best Position: {gbp}")
        st.success(f"Global Best Fitness: {gbs}")
        st.image('PSO_Animation_Output.gif',use_column_width=True)
    # Define the main content
    st.title("Particle Swarm Optimization")
    st.info("Summary of the Control Parameters")
    parameters = {'Parameter': ['Number of particles', 'Number of iterations', 'C1', 'C2', 'Inertia weight'],
              'Value': [n_particles, n_iterations, c1, c2, w]}

    # create a pandas dataframe from the dictionary
    df = pd.DataFrame(parameters)
    # create a table to display the control parameters
    st.table(df)

    

if __name__ == '__main__':
    app()
