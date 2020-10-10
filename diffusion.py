import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from scipy.stats import rankdata

import numpy as np
import random
import math
import time
import glob, os
import bisect
import heapq


dom_u = 1
dom_l = -1
bosses = [3,4,6,7]
n_bosses = len(bosses)
n_best = 9
n_weights = 265
n_hidden_neurons = 10
mutation_rate = 0.2
gens = 1000
npop = 40
gens_first = 15

def simulation(env,x):
    """Runs simulation"""
    f,p,e,t = env.play(pcont=x)
    return f

# normalizes
def norm(x, pfit_pop):

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

def scramble_pop(pop, fit_pop, fraction):
    """this function deletes a worst fraction of the population
    and replaces them with random ones, so as to keep diversity up"""

    ranked_worst = np.argsort(fit_pop)

    for rank in ranked_worst[:int(fraction * npop)]:
        for gene in range(0, len(pop[rank])-1):
            pop[rank][gene] = np.random.uniform(dom_l, dom_u, 1)

    return pop

def death_match(pop, fit_pop):
    """this function determines which individuals of the population get replaced"""

    # make some lists to keep track of individuals
    copy_pop = pop
    copy_fit_pop = fit_pop
    survivors_pop = []
    survivors_fitness = []
    death_match_pop = []
    death_match_fitness = []

    # we do 20 rounds of 5 randomly sampled individuals from the remaining pop
    for _ in range(int(npop / 5)):
        for _ in range(5):

            # choose the competitors
            if copy_pop.shape[0] > 1:
                index = np.random.randint(0, copy_pop.shape[0]-1, 1)[0]
            else:
                index = 0

            death_match_pop.append(copy_pop[index])
            death_match_fitness.append(copy_fit_pop[index])

            # delete the chosen ones from the population, so they can't be sampled again
            copy_pop = np.delete(copy_pop, [index], 0)
            copy_fit_pop = np.delete(copy_fit_pop, [index], 0)

        # now do the death match and add the best to the survivors
        survivor_index = np.argmax(death_match_fitness)
        survivors_pop.append(death_match_pop[survivor_index])
        survivors_fitness.append(death_match_fitness[survivor_index])

    # we return the 20 survivors and their fitness values as np arrays
    return np.array(survivors_pop), np.array(survivors_fitness)

def parent_selection(pop, fit_pop, rounds):
    """this function will select which 5 parents will mate"""

    # get the list of worst to best of the population
    worst_to_best = np.argsort(fit_pop)

    # select the parents based on which round, first 2 parents are sampled from top 40%
    p1 = pop[worst_to_best[pop.shape[0] - rounds - 1]]
    p2 = pop[worst_to_best[pop.shape[0] - rounds - 2]]

    # last 3 parents are randomly chosen
    p3, p4, p5 = pop[np.random.randint(0, pop.shape[0]-1, 3)]

    return np.array([p1, p2, p3, p4, p5])


def recombination1(parents):
    """recombines 5 parents into 4 offspring"""

    # pick 5 random numbers that add up to 1
    random_values = np.random.dirichlet(np.ones(5),size=1)[0]

    # those random values will serve as weights for the genes 2 offspring get (whole arithmetic recombination)
    offspring1 = random_values[0] * parents[0] + random_values[1] * parents[1] + random_values[2] * parents[2] + random_values[3] * parents[3] + \
        random_values[4] * parents[4]

    # repeat for offspring 2
    random_values = np.random.dirichlet(np.ones(5),size=1)[0]
    offspring2 = random_values[0] * parents[0] + random_values[1] * parents[1] + random_values[2] * parents[2] + random_values[3] * parents[3] + \
        random_values[4] * parents[4]

    # the other 2 offspring will come from 4-point crossover
    random_points = np.sort(np.random.randint(1, parents[0].shape[0]-2, 4))

    # to make it so that it won't always be p1 who gives the first portion of DNA etc, we shuffle the parents
    np.random.shuffle(parents)

    # add the genes together
    offspring3 = np.concatenate((parents[0][0:random_points[0]], parents[1][random_points[0]:random_points[1]], parents[2][random_points[1]:random_points[2]],\
        parents[3][random_points[2]:random_points[3]], parents[4][random_points[3]:]))

    # repeat for offspring 4
    random_points = np.sort(np.random.randint(1, parents[0].shape[0]-2, 4))
    np.random.shuffle(parents)
    offspring4 = np.concatenate((parents[0][0:random_points[0]], parents[1][random_points[0]:random_points[1]], parents[2][random_points[1]:random_points[2]],\
        parents[3][random_points[2]:random_points[3]], parents[4][random_points[3]:]))

    # return the offspring
    return np.concatenate(([offspring1], [offspring2], [offspring3], [offspring4]))

def mutate(offspring):
    """this function will mutate the offspring"""

    # get the children and their genes
    offspring = offspring
    for child in offspring:

        # don't mutate every child, make it 50% of the offspring
        if np.random.uniform(0,0.4,1) < mutation_rate:
            for gene in range(0, len(child)-1):

                # pick a random number between 0-1, mutate if < mutation rate
                if np.random.uniform(0,1,1) < mutation_rate:

                    # change the gene by a small number from a very narrow normal distribution
                    child[gene] += np.random.normal(0, 0.2, 1)

                # make sure the genes don't get values outside of the limits
                if child[gene] > dom_u:
                    child[gene] = dom_u
                if child[gene] < dom_l:
                    child[gene] = dom_l

    return offspring


def sample_insertion(pops):
    """
    Takes dictionary of all best bosses weights, randomly takes one from each boss and inserts it randomly into another boss list.
    Returns the changed dictionary.
    """

    # creates list with 1 random sample from each cell and deletes the sample from the dictionary
    sample_list = [0] * n_bosses
    for i in range(n_bosses):
        random_sample = random.randint(0, n_best-1)
        sample_list[i] = pops[bosses[i]][random_sample]
        pops[bosses[i]] = np.delete(pops[bosses[i]], random_sample, axis=0)

    # takes a random indivual from the sample list and inserts it in a different cell
    for j in range(n_bosses):
        random_insertion = j
        while random_insertion == j:
            random_insertion = random.randint(0, n_bosses-1)
        random_location = random.randint(0, n_best-2)
        pops[bosses[j]] = np.insert(pops[bosses[j]], random_location ,sample_list[random_insertion], axis=0)
    return pops

def create_grid(pops):
    """
    Takes a dictionary with weights of each indivual separated by boss, creates grid with weights of each indivual maintaining boss cells.
    Returns either a 2x2, 2x3 or 2x4 cells grid depending on the number of bosses.
    """
    # Creates random grid order in which the grids will be built
    grid_order = random.sample(bosses, n_bosses)

    # reshapes list in dictionary to form a grid block
    for i in range(n_bosses):
        pops[bosses[i]] = pops[bosses[i]].reshape((int(math.sqrt(n_best)), int(math.sqrt(n_best)), n_weights))
        # pops[i] = pops[i].reshape((int(math.sqrt(n_best)), int(math.sqrt(n_best))))

    # Creates two vertical rows of grid blocks depending on the number of cells
    grid1 = np.hstack((pops[grid_order[0]], pops[grid_order[1]]))
    if (n_bosses / 2) > 2:
        for j in range(2, int(n_bosses / 2)):
            grid1 = np.hstack((grid1, pops[grid_order[j]]))
        grid2 = np.hstack((pops[grid_order[int(n_bosses / 2)]], pops[grid_order[int(n_bosses / 2) + 1]]))
        for k in range(int(((n_bosses / 2) + 2)), n_bosses):
            grid2 = np.hstack((grid2, pops[grid_order[k]]))
    else:
        grid2 = np.hstack((pops[grid_order[2]], pops[grid_order[3]]))


    #  adds both vertical blocks together horizontally
    grid = np.vstack((grid1, grid2))

    return grid

def init_evaluate(pop):
    """
    Evaluates all the weights from pop and inserts fitness values into new grid. Calculates fitness for each of the listed bosses.
    Returns grid with the average fitness (over all selected bosses) - the standard devation as one value.
    """
    shape = pop.shape
    pop_fit = np.zeros((shape[0], shape[1]))
    print(pop_fit.shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pop_fit[i][j] = simulation(env, pop[i][j])
    return pop_fit

def positions():
    """
    Creates dictionary containing lists of positions for each cell. Returns this dictionary.
    """
    grid_pos = {}
    num = 0
    while num < int(n_bosses / 2):
        for i in range(int(math.sqrt(n_best))):
            for j in range(int(math.sqrt(n_best))):
                if num in grid_pos.keys():
                    grid_pos[num].append((i, j + int(num * math.sqrt(n_best))))
                else:
                    grid_pos[num] = [(i, j + int(num * math.sqrt(n_best)))]
        num += 1

    num2 = 0
    while num < int(n_bosses):
        for i in range(int(math.sqrt(n_best))):
            for j in range(int(math.sqrt(n_best))):
                if num in grid_pos.keys():
                    grid_pos[num].append((i + int(math.sqrt(n_best)), j + int(num2 * math.sqrt(n_best))))
                else:
                    grid_pos[num] = [(i + int(math.sqrt(n_best)), j + int(num2 * math.sqrt(n_best)))]
        num += 1
        num2 += 1

    return grid_pos

def neighbourhood(x, y):
    """
    Sets neighbourhood area of range 1 around the selected indivual (9 total) like a toroidal grid
    """
    if x == int(math.sqrt(n_best)*2) - 1 and y == int(math.sqrt(n_best)*(n_bosses / 2)) - 1:
        neighbours = [(x,y), (0,y), (x, 0), (0, 0), (x-1, y), (x, y-1), (x-1, y-1), (0, y-1), (x-1, 0)]
    elif x == int(math.sqrt(n_best)*2) - 1:
        neighbours = [(x,y), (0,y), (x, y+1), (0, y+1), (x-1, y), (x, y-1), (x-1, y-1), (0, y-1), (x-1, y+1)]
    elif y == int(math.sqrt(n_best)*(n_bosses / 2)) - 1:
        neighbours = [(x,y), (x+1,y), (x, 0), (x+1, 0), (x-1, y), (x, y-1), (x-1, y-1), (x+1, y-1), (x-1, 0)]
    else:
        neighbours = [(x,y), (x+1,y), (x, y+1), (x+1, y+1), (x-1, y), (x, y-1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]

    return neighbours


def prob_sum(prob):
    """Calculates the sum of the probabilities"""
    total = sum(prob)
    result = []
    val = 0
    for p in prob:
        val += p
        result.append(val / total)
    return result

def linear_rank_prob(pop_fit, neighbours):
    """Ranks every indivual in the neighbourhood and returns list of probabilities and inverse probabilities"""
    # get all fitness values from the neighbourhood
    neighbourhood_size = len(neighbours)
    neighbour_fit = []
    for i in range(1, neighbourhood_size):
        neighbour_fit.append(pop_fit[neighbours[i][0]][neighbours[i][1]])

    # Rank fitnes values (Worst = 0)
    neighbour_rank = rankdata(neighbour_fit, method="ordinal")
    neighbour_rank = [i - 1 for i in neighbour_rank]
    inv_neighbour_rank = list(map(lambda x: len(neighbour_rank) - x - 1, neighbour_rank))

    # Creates selection probabilities
    neighbour_prob = list(map(lambda x: (2*x)/((neighbourhood_size-1)*(neighbourhood_size-2)), neighbour_rank))
    inv_neighbour_prob  = list(map(lambda x: (2*x)/((neighbourhood_size-1)*(neighbourhood_size-2)), inv_neighbour_rank))

    # Adds up the probabilities used for selection
    neighbour_prob = prob_sum(neighbour_prob)
    inv_neighbour_prob = prob_sum(inv_neighbour_prob)

    return neighbour_prob, inv_neighbour_prob

def recombination(p1, p2, recomb_type):
    """Recombines two parents into one offspring using selected strategy.
    Select either 'WAR', '1P', or 'AVG' for recomb_type (3rd argument)"""

    if recomb_type == 'WAR':

        # whole arithmetic recombination:
        fraction = np.random.uniform(0, 1)
        offspring = fraction * p1 + (1-fraction) * p2

    elif recomb_type == '1P':

        # 1-point recombination
        point = int(np.random.uniform(1, len(p1)-1))
        offspring = np.concatenate((p1[:point], p2[point:]))

    elif recomb_type == 'AVG':

        # gene averaging:
        offspring = (p1 + p2) / 2

    return offspring

def scaled_mutation(offspring, pfitness):
    """Mutates individuals from a population list according to their fitness.
    Low fitness means higher mutation stepsizes, and vice versa.
    Returns a list of mutated individuals."""

    fitness = np.mean(pfitness)

    # linear scaling from 1 to 0
    mutation_factor = 1 - fitness / 100

    for i in range(len(offspring)):

        # genes have a chance to mutate
        if np.random.uniform(0, 1) > 0.313:

            # mutate genes with a gaussian, suppressed by the mutation factor
            mutation_size = mutation_factor * np.random.normal(0, 0.15)

            offspring[i] += mutation_size

             # truncate the genes (weights) at -1 and 1
            if offspring[i] > 1:
                offspring[i] = 1
            elif offspring[i] < -1:
                offspring[i] = -1

    return offspring


def evolution(pop, pop_fit, positions):

    for i in range(n_bosses):
        # Selects one individual from each cell
        random_pos = random.sample(positions[i], 1)[0]
        x = random_pos[0]
        y = random_pos[1]

        # Defines neighbourhood
        neighbours = neighbourhood(x, y)
        neighbour_prob, inv_neighbour_prob = linear_rank_prob(pop_fit, neighbours)

        # Selects parent2 based on linear rank within the neighbourhood
        parent1 = pop[x][y]
        fit_parent1 = pop_fit[x][y]
        selected_mate = bisect.bisect(neighbour_prob, random.random()) + 1
        selected_mate = neighbours[selected_mate] # coordinates
        parent2 = pop[selected_mate[0]][selected_mate[1]]
        fit_parent2 = fit_pop[selected_mate[0]][selected_mate[1]]

        # Creates offspring using recombination of two parents
        offspring = recombination(parent1, parent2, "WAR")

        # Mutates the offspring based on the average fitness of parents
        offspring_mut = scaled_mutation(offspring, [fit_parent1, fit_parent2])

        # Selects indivual for replacement
        kiss_of_death = bisect.bisect(inv_neighbour_prob, random.random()) + 1
        kiss_of_death = neighbours[kiss_of_death] # coordinates

        # Determines fitness for offspring
        offspring_fit = simulation(env, offspring_mut)
        print(f"Fitness: {offspring_fit}")

        # Places offspring in the population and gives it the fitness value.
        pop[kiss_of_death[0]][kiss_of_death[1]] = offspring_mut
        pop_fit[kiss_of_death[0]][kiss_of_death[1]] = offspring_fit

    return pop, pop_fit

positions = positions()

for j in range(1):
    experiment_name = f'diffusion_easy_{j}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    pops = {}

    for i in range(n_bosses):

        # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                          enemies=[bosses[i]],
                          playermode="ai",
                          player_controller=player_controller(n_hidden_neurons),
                          enemymode="static",
                          level=2,
                          speed="fastest")

        # default environment fitness is assumed for experiment

        env.state_to_log() # checks environment state


        ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

        # genetic algorithm params

        run_mode = 'train' # train or test

        # number of weights for multilayer with 10 hidden neurons
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

        # loads file with the best solution for testing
        if run_mode =='test':

            bsol = np.loadtxt(experiment_name+'/best.txt')
            print( '\n RUNNING SAVED BEST SOLUTION \n')
            env.update_parameter('speed','normal')
            file_aux = open(experiment_name+'/gain.csv', 'a')
            file_aux.write("Fitness Phealth Ehealth Time")
            for i in range(5):
                results = np.array(list(map(lambda y: env.play(y), [bsol])))
                file_aux.write('\n'+str(results[0][0])+' '+str(results[0][1])+' '+str(results[0][2])+' '+str(results[0][3]))
            file_aux.close()

        else:
            # initializes population loading old solutions or generating new ones

            print( '\nNEW EVOLUTION\n')
            pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
            fit_pop = evaluate(pop)
            best = np.argmax(fit_pop)
            mean = np.mean(fit_pop)
            std = np.std(fit_pop)
            ini_g = 0
            solutions = [pop, fit_pop]
            env.update_solutions(solutions)


            # evolution
            new_best_counter = 0
            all_time_best = 0
            # last_sols_d = {}
            # notimproved_d = {}

            for j in range(ini_g+1, gens_first):

                rounds = int(npop/5)
                offspring = np.zeros((0, n_vars))
                for k in range(1, rounds+1):

                    # choose parents
                    parents = parent_selection(pop, fit_pop, (k-1)*2)

                    # honey, get the kids
                    offspring_group = recombination1(parents)

                    # add them to the offspring array
                    offspring = np.concatenate((offspring, offspring_group))

                # mutate half the offspring for diversity
                offspring = mutate(offspring)

                # we have the offspring, now we kill 80% of the population
                pop = death_match(pop, fit_pop)[0]

                # mutate the surviving pop as well to increase search space
                pop = mutate(pop)

                # combine the survivors with the offspring to form the new pop
                pop = np.concatenate((pop, offspring))

                # test the pop
                fit_pop = evaluate(pop)

                # get stats
                best = np.argmax(fit_pop)
                std  =  np.std(fit_pop)
                mean = np.mean(fit_pop)

                # if 3 generations in a row don't give a new best solution, replace a fraction of the pop
                if fit_pop[best] > all_time_best:
                    all_time_best = fit_pop[best]
                    new_best_counter = 0

                else:
                    new_best_counter += 1

                if new_best_counter > 3:
                    pop = scramble_pop(pop, fit_pop, 0.3)
                    new_best_counter = 0

                # saves simulation state
                solutions = [pop, fit_pop]
                env.update_solutions(solutions)
                env.save_state()

            best_fit_inds = heapq.nlargest(9, fit_pop)
            best_index = heapq.nlargest(9, range(npop), key=lambda x: fit_pop[x])
            best_pop_inds = list(map(lambda y: pop[y], best_index))
            pops[bosses[i]] = best_pop_inds
            print(f"Completed boss {bosses[i]}")
    print(pops)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=bosses,
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      multiplemode ="yes")

    # default environment fitness is assumed for experiment

    env.state_to_log()

    ###   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

    ini = time.time()

    # genetic algorithm params

    run_mode = 'train'

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    pops = sample_insertion(pops)
    pop = create_grid(pops)
    fit_pop = init_evaluate(pop)
    file_aux = open(experiment_name+'/results.csv', 'a')
    file_aux.write("Generation Best Mean Std")
    file_aux.close()

    for i in range(gens):
        pop, fit = evolution(pop, fit_pop, positions)
        print(f"Gen {i}")
        best_in_gen = np.max(fit)
        mean_in_gen = np.mean(fit)
        std_in_gen = np.std(fit)
        file_aux = open(experiment_name+'/results.csv', 'a')
        file_aux.write('\n'+str(i)+' '+str(best_in_gen)+' '+str(mean_in_gen)+' '+str(std_in_gen))
        file_aux.close()
    best = np.max(fit_pop)
    best_index = np.unravel_index(np.argmax(fit_pop), fit_pop.shape)
    print(f"best fitness: {best}")
    np.savetxt(experiment_name+'/best.txt',pop[best_index[0]][best_index[1]])
