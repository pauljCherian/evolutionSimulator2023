from __future__ import division, print_function
from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines

import numpy as np
import operator

from time import sleep
from math import atan2, cos, degrees, floor, radians, sin, sqrt, pi
from random import randint, random, sample, uniform, seed

import tkinter as tk
from tkinter import Canvas, Label, Tk

#--- CONSTANTS ----------------------------------------------------------------+

settings = {}
seed(26)
# EVOLUTION SETTINGS
settings['pop_size'] = 30       # number of organisms
settings['target_num'] = 1      # number of food particles
settings['gens'] =100           # number of generations
settings['elitism'] = 0.2      # elitism (selection bias)
settings['mutate'] = 0.1       # mutation rate

# SIMULATION SETTINGS
settings['gen_time'] = 4.2      # generation length         (seconds)
settings['dt'] = 0.03           # simulation time step      (dt)
settings['dTheta_max'] = 180    # max rotational speed      (degrees per second)
settings['v_max'] = 200         # max velocity              (units per second)
settings['dv_max'] = 100      # max acceleration (+/-)    (units per second^2)
settings['x_min'] = 10    # arena western border
settings['x_max'] = 790      # arena eastern border
settings['y_min'] = 10       # arena southern border
settings['y_max'] = 790     # arena northern border
settings['canvas_dimension'] = 800


settings['plot'] = True        # plot the simulation?
settings['moving_target'] = False
settings['target_v'] = 6
# ORGANISM NEURAL NET SETTINGS
settings['inodes'] = 1          # number of input nodes
settings['hnodes'] = 5          # number of hidden nodes
settings['onodes'] = 2          # number of output nodes

#--- CLASSES ------------------------------------------------------------------+

class Target:
    def __init__(self, settings, setManual=False, coord=None):
        self.x = uniform(settings['x_min'] + 0.1*settings['x_max'], settings['x_max'])
        self.y = uniform(settings['y_min'] + 0.1*settings['y_max'], settings['y_max']*0.5)

        self.angle = uniform(0, 360)
        if setManual:
            self.x = coord[0]
            self.y = coord[1]
    def move(self, settings, v, move):
        if move:
            dx = v * cos(radians(self.angle)) 
            dy = v * sin(radians(self.angle)) 

            if (settings['x_min'] + 30 < self.x + dx < settings['x_max']- 30):
                if (settings['y_min'] + 30 < self.y + dy < settings['y_max'] * 0.75):
                    self.x += dx
                    self.y += dy
                else:
                    self.angle *= -1
                    self.angle = self.angle % 360

                    dx = v * cos(radians(self.angle)) 
                    dy = v * sin(radians(self.angle))
                    self.x += dx
                    self.y += dy

            else:
                self.angle *= -1 + 180
                self.angle = self.angle % 360
                dx = v * cos(radians(self.angle)) 
                dy = v * sin(radians(self.angle))
                self.x += dx
                self.y += dy
                

        
            
class Agent:
    def __init__(self, settings, wih=None, who=None, name=None):
        self.x = (settings['x_max'] - settings['x_min'])/2
        self.y = settings['y_max'] * 0.9

        self.theta = uniform(-180, 180)                 # orientation   [-180, 180]
        self.v = uniform(0,settings['v_max'])   # velocity      [0, v_max]
        self.dv = uniform(-settings['dv_max'], settings['dv_max'])   # dv
        self.wih = wih
        self.who = who
        self.name = name
        self.total_dist = 0.1
        self.currentDistToTarget = 0

    def getNearestTargetDistTheta(self, targets):
        closest_dist = float('inf')
        closest_target = None
        for target in targets:
            d = dist(self.x, self.y, target.x, target.y)
            if d < closest_dist:
                closest_dist = d
                closest_target = target

        thetaToTarget = degrees(atan2((closest_target.y-self.y), (closest_target.x-self.x))) # [-pi,pi] converted to [-180,180]
        self.currentDistToTarget = closest_dist
        return closest_target, closest_dist, thetaToTarget

    def think(self, targets):
        nearestTarget, distToTarget, thetaToTarget = self.getNearestTargetDistTheta(targets)

        # SIMPLE MLP
        normThetaToTarget = thetaToTarget/180
        
        af = lambda x: np.tanh(x)               # activation function

        h1 = af(np.dot(self.wih, normThetaToTarget))  # hidden layer
        out = af(np.dot(self.who, h1))          # output layer

        # UPDATE dv AND dr WITH MLP RESPONSE
        self.nn_dv = float(out[0])   # [-1, 1]  (accelerate=1, deaccelerate=-1)
        self.nn_dTheta = float(out[1])   # [-1, 1]  (left=1, right=-1)
        
    # UPDATE ORIENTATION
    def update_theta(self, settings):
        self.theta = self.nn_dTheta * 180 * settings['dTheta_max'] * settings['dt']

        self.theta = self.theta % 360

    # UPDATE VELOCITY
    def update_vel(self, settings):
        self.v += self.nn_dv * settings['dv_max'] * settings['dt']
        if self.v < 0:
            self.v = 0
        if self.v > settings['v_max']:
            self.v = settings['v_max']

    # UPDATE POSITION
    def update_pos(self, settings):
        dx = self.v * cos(radians(self.theta)) * settings['dt']
        dy = self.v * sin(radians(self.theta)) * settings['dt']
        self.x += dx
        self.y += dy
        self.total_dist += ((dx**2)+(dy**2))**(0.5)
       
        

    def calculateFitness(self, targets):
        nearestTarget, distToTarget, thetaToTarget = self.getNearestTargetDistTheta(targets)
        dist_fit = 1 / (distToTarget ** 2)
        # angle_fit = 1 / (1 - (thetaToTarget / radians(self.r)) ** 2)
        tot_dist_covered_fit = 1 / ((self.total_dist) ** 2)
        self.fitness = dist_fit + 0.15* tot_dist_covered_fit

#--- FUNCTIONS ----------------------------------------------------------------+

def dist(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)

def evolve(settings, agents_old, targets, gen):
    elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_agents = settings['pop_size'] - elitism_num

    for agent in agents_old:
        agent.calculateFitness(targets)

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for agent in agents_old:
        if agent.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = agent.fitness
        if agent.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = agent.fitness
        stats['SUM'] += agent.fitness
        stats['COUNT'] += 1
    stats['AVG'] = stats['SUM'] / stats['COUNT']

    #--- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    agents_sorted = sorted(agents_old,
                           key=operator.attrgetter('fitness'),
                           reverse=True)
    agents_new = []
    for i in range(0, elitism_num):
        agents_new.append(Agent(settings,
                                wih=agents_sorted[i].wih,
                                who=agents_sorted[i].who,
                                name=agents_sorted[i].name
                                ))
    #--- GENERATE NEW ORGANISMS ---------------------------+
    for w in range(0, new_agents):
        # SELECTION (TRUNCATION SELECTION)
        canidates = range(0, elitism_num)
        random_index = sample(canidates, 2)
        agent_1 = agents_sorted[random_index[0]]
        agent_2 = agents_sorted[random_index[1]]
        # CROSSOVER
        crossover_weight = random()
        wih_new = (crossover_weight*agent_1.wih)+((1-crossover_weight)*agent_2.wih)
        who_new = (crossover_weight*agent_1.who)+((1-crossover_weight)*agent_2.who)

        # MUTATION
        mutate = random()
        if mutate <= settings['mutate']:
            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0,1)
            # MUTATE: WIH WEIGHTS
            if mat_pick == 0:
                index_row = randint(0,settings['hnodes']-1)
                wih_new[index_row] = wih_new[index_row] * uniform(0.9, 1.1)
                if wih_new[index_row] >  1: wih_new[index_row] = 1
                if wih_new[index_row] < -1: wih_new[index_row] = -1
            # MUTATE: WHO WEIGHTS
            if mat_pick == 1:
                index_row = randint(0,settings['onodes']-1)
                index_col = randint(0,settings['hnodes']-1)
                who_new[index_row][index_col] = who_new[index_row][index_col] * uniform(0.9, 1.1)
                if who_new[index_row][index_col] >  1: who_new[index_row][index_col] = 1
                if who_new[index_row][index_col] < -1: who_new[index_row][index_col] = -1
        agents_new.append(Agent(settings, wih=wih_new, who=who_new, name='gen['+str(gen)+']-agent['+str(w)+']'))
    return agents_new, stats

def plot_agent(agent, canvas):

    d_scaler = 40
    d = settings['canvas_dimension']/d_scaler
    canvas.create_oval(agent.x,
                       agent.y, 
                       agent.x + d, 
                       agent.y + d, 
                       fill='blue')
    dx = d_scaler * cos(radians(agent.theta)) * 0.4
    dy = d_scaler * sin(radians(agent.theta)) * 0.4
    canvas.create_line(agent.x + d/2, agent.y + d/2,agent.x + d/2 + dx, agent.y + d/2+dy, arrow=tk.LAST, width=2, )
    # canvas.create_text(agent.x, agent.y, text='{}'.format(int(agent.currentDistToTarget)))
    

def plot_target(target, canvas):

    d = settings['canvas_dimension']/30
    canvas.create_rectangle(target.x,
                       target.y,
                       target.x +d,
                       target.y+d,
                       fill='green')
    
def plot_frame(settings, agents, targets, gen, time, canvas,):
    
    canvas.delete('all')
    # canvas.create_rectangle(settings['x_min'], settings['y_min'], settings['x_max'], settings['y_max'])
    canvas.create_text(750, 50, text='Gen: {}\nTime: {}'.format(gen, time))
    for agent in agents:
        plot_agent(agent, canvas)
    for target in targets:
        plot_target(target, canvas)
    

def simulate(settings, agents, targets, gen, canvas):
    total_time_steps = int(settings['gen_time'] / settings['dt'])
    
    #--- CYCLE THROUGH EACH TIME STEP ---------------------+
    for t_step in range(0, total_time_steps, 1):
        
         # PLOT SIMULATION FRAME
        if settings['plot']==True:
            win.update()
            plot_frame(settings, agents, targets, gen, t_step, canvas)
            sleep(0.001)
            win.update()
        # UPDATE FITNESS FUNCTION
        for agent in agents:
            agent.calculateFitness(targets)
        # GET ORGANISM RESPONSE
        for agent in agents:
            agent.think(targets)
        # UPDATE ORGANISMS POSITION AND VELOCITY
        for agent in agents:
            agent.update_theta(settings)
            agent.update_vel(settings)
            agent.update_pos(settings)
        for target in targets:
            target.move(settings, settings['target_v'], settings['moving_target'])

    return agents

def run(settings):
    global win
    win= Tk()
    win.title('Target Practice Evolution')
    global canvas
    canvas = Canvas(win, width= settings['canvas_dimension'], height=settings['canvas_dimension'])

    canvas.pack()
    
    #--- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    targets = []
    for i in range(0,settings['target_num']):
        targets.append(Target(settings))
    #--- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    agents = []
    for i in range(0,settings['pop_size']):
        wih_init = np.random.uniform(-1, 1, (settings['hnodes'], settings['inodes']))     # mlp weights (input -> hidden)
        who_init = np.random.uniform(-1, 1, (settings['onodes'], settings['hnodes']))     # mlp weights (hidden -> output)
        agents.append(Agent(settings, wih_init, who_init, name='gen[0]-agent['+str(i)+']'))
    #--- SIMULATION ---------------------------------------+
    last_agents = simulate(settings, agents, targets, 0, canvas)
    #--- EVOLUTION ----------------------------------------+
    for gen in range(1, settings['gens']+1):
        new_agents, stats = evolve(settings, last_agents, targets, gen)
        last_agents = simulate(settings, new_agents, targets, gen, canvas)
        print('> GEN:',gen,'BEST:',stats['BEST'],'AVG:',stats['AVG'],'WORST:',stats['WORST'])
    win.mainloop()
    return stats

run(settings)