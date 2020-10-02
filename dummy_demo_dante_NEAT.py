# framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import neat


n_hidden_neurons = 10

experiment_name = 'multi_demo_neat'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  enemies=[7,8],
                  multiplemode="yes",
                  playermode="ai",
                  # player_controller=player_controller(0),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons.
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
# npop = 100
# gens = 30
# mutation = 0.2
# last_best = 0

np.random.seed(420)

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# normalizes
def norm(x,pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


#### NEAT PART

def eval_genome(genomes,config):
    nets = []
    ge = []
    for genome_id, genome in genomes:
        print(genome)
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        input = np.random.uniform(dom_l, dom_u, 20)
        output = net.activate(input)
        action = []
        for out in output:
            action.append(round(out))
        print("OUTPUT", output, "\nAction",action)
        f,p,e,t = env.play(pcont=action)
        genome.fitness = f


# Setting up NEAT
def run(config_file):
    # Coupling with config file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # setting up population
    pop = neat.Population(config)

    # Setting up statistics
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # giving fitness function and amount of generations
    winner = pop.run(eval_genome,2)

    print('\nBest man:\n{!s}'.format(winner))



# Locating config file
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "config-feedforward.txt")
    run(config_file)

#
# # loads file with the best solution for testing
# if run_mode =='test':
#
#     bsol = np.loadtxt(experiment_name+'/best.txt')
#     print( '\n RUNNING SAVED BEST SOLUTION \n')
#     env.update_parameter('speed','normal')
#     evaluate([bsol])
#
#     sys.exit(0)
#
#
# # initializes population loading old solutions or generating new ones
#
# if not os.path.exists(experiment_name+'/evoman_solstate'):
#
#     print( '\nNEW EVOLUTION\n')
#
#     pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
#     fit_pop = evaluate(pop)
#     best = np.argmax(fit_pop)
#     mean = np.mean(fit_pop)
#     std = np.std(fit_pop)
#     ini_g = 0
#     solutions = [pop, fit_pop]
#     env.update_solutions(solutions)
#
# else:
#
#     print( '\nCONTINUING EVOLUTION\n')
#
#     env.load_state()
#     pop = env.solutions[0]
#     fit_pop = env.solutions[1]
#
#     best = np.argmax(fit_pop)
#     mean = np.mean(fit_pop)
#     std = np.std(fit_pop)
#
#     # finds last generation number
#     file_aux  = open(experiment_name+'/gen.txt','r')
#     ini_g = int(file_aux.readline())
#     file_aux.close()
#
#
#
#
# # saves results for first pop
# file_aux  = open(experiment_name+'/results.txt','a')
# file_aux.write('\n\ngen best mean std')
# print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
# file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
# file_aux.close()
#
#
# # evolution
#
# last_sol = fit_pop[best]
# notimproved = 0
#
# for i in range(ini_g+1, gens):
#
#     offspring = crossover(pop)  # crossover
#     fit_offspring = evaluate(offspring)   # evaluation
#     pop = np.vstack((pop,offspring))
#     fit_pop = np.append(fit_pop,fit_offspring)
#
#     best = np.argmax(fit_pop) #best solution in generation
#     fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
#     best_sol = fit_pop[best]
#
#     # selection
#     fit_pop_cp = fit_pop
#     fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
#     probs = (fit_pop_norm)/(fit_pop_norm).sum()
#     chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
#     chosen = np.append(chosen[1:],best)
#     pop = pop[chosen]
#     fit_pop = fit_pop[chosen]
#
#
#     # searching new areas
#
#     if best_sol <= last_sol:
#         notimproved += 1
#     else:
#         last_sol = best_sol
#         notimproved = 0
#
#     if notimproved >= 15:
#
#         file_aux  = open(experiment_name+'/results.txt','a')
#         file_aux.write('\ndoomsday')
#         file_aux.close()
#
#         pop, fit_pop = doomsday(pop,fit_pop)
#         notimproved = 0
#
#     best = np.argmax(fit_pop)
#     std  =  np.std(fit_pop)
#     mean = np.mean(fit_pop)
#
#
#     # saves results
#     file_aux  = open(experiment_name+'/results.txt','a')
#     print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
#     file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
#     file_aux.close()
#
#     # saves generation number
#     file_aux  = open(experiment_name+'/gen.txt','w')
#     file_aux.write(str(i))
#     file_aux.close()
#
#     # saves file with the best solution
#     np.savetxt(experiment_name+'/best.txt',pop[best])
#
#     # saves simulation state
#     solutions = [pop, fit_pop]
#     env.update_solutions(solutions)
#     env.save_state()
#
#
#
#
# fim = time.time() # prints total execution time for experiment
# print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
#
#
# file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
# file.close()
#
#
# env.state_to_log() # checks environment state
