
# Neuroevolution Simulation

A Python script that uses the **Neuroevolution of Augmenting Topologies (NEAT)** algorithm to evolve a series of dots to hit a target on a 2D screen.

The purpose of this project was to visually demonstrate an evolutionary AI algorithm that does **NOT** rely on backpropagation. Rather, it uses properties of inheritance, genetic recombination, and fitness to provide an analog to biological evolution.

The simulation has a variety of settings that can be tweaked to model different evolutionary scenarios. To name a few: **mutation rate**, **elitism**, **generations**, and **population size**. Each agent has a 'brain' that is a **3-layer perceptron**. The weights that make up this perceptron control the organisms' movement. Genetic recombination is modeled through the combining and mutation of one agent's weights with another to produce offspring.

After every generation, the agents with the highest fitness (defined by how close they get to the target and how long the path they took was) are kept, and their weights are combined to create the next generation. This evolution is visualized with **PyGame**.

---

### Code Components

- **Classes**:
  - `Target`: Represents a target, its position, and optional movement behavior.
  - `Agent`: Represents an agent with properties like position, velocity, orientation, and a neural network.

- **Functions**:
  - `dist`: Computes the Euclidean distance between two points.
  - `evolve`: Handles the genetic algorithm, including selection, crossover, and mutation.
  - `simulate`: Simulates one generation of agent movement and target interaction.
  - `run`: Orchestrates the simulation and evolution process, visualizing results in a `tkinter` window.

---

### How to Use

1. **Dependencies**:
   - Ensure you have Python installed with the following libraries:
     - `numpy`
     - `matplotlib`
     - `tkinter`

2. **Run the Simulation**:
   - Execute the script to start the simulation. The `run` function initializes the environment, agents, and targets, and then evolves the agents over the specified number of generations.

3. **Adjust Settings**:
   - Modify the `settings` dictionary to customize simulation parameters like:
     - **Population size**
     - **Mutation rate**
     - **Arena size**
     - **Neural network structure**

---
