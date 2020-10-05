import numpy as np
import random
import math


class Tuning:
    def __init__(
        self, numberOfParents, numberOfParentsMating,
    ):
        self.numberOfParents = numberOfParents
        self.numberOfParentsMating = numberOfParentsMating

    def initilialize_poplulation(self):
        mutation_rate = np.empty([self.numberOfParents, 1])

        for i in range(self.numberOfParents):
            mutation_rate[i] = round(random.uniform(0, 1), 3)

        return mutation_rate

    def new_parents_selection(self, fitness, population):
        selectedParents = np.empty(
            (self.numberOfParentsMating, population.shape[1])
        )  # create an array to store fittest parents

        # find the top best performing parents
        for parentId in range(self.numberOfParentsMating):
            bestFitnessId = np.where(fitness == np.max(fitness))
            bestFitnessId = bestFitnessId[0][0]
            selectedParents[parentId, :] = population[bestFitnessId, :]

            # Set to -1 so it is not chosen again as best parent
            fitness[bestFitnessId] = -1
        return selectedParents

    def crossover_uniform(self, parents, childrenSize):

        crossoverPointIndex = np.arange(
            0, np.uint8(childrenSize[1]), 1, dtype=np.uint8
        )  # get all the index

        crossoverPointIndex1 = np.random.randint(
            0, np.uint8(childrenSize[1]), np.uint8(childrenSize[1] / 2)
        )  # select half  of the indexes randomly

        crossoverPointIndex2 = np.array(
            list(set(crossoverPointIndex) - set(crossoverPointIndex1))
        )  # select leftover indexes

        children = np.empty(childrenSize)

        """
        Create child by choosing parameters from two parents selected using new_parent_selection function. The parameter values
        will be picked from the indexes, which were randomly selected above. 
        """
        for i in range(childrenSize[0]):

            # find parent 1 index
            parent1_index = i % parents.shape[0]
            # find parent 2 index
            parent2_index = (i + 1) % parents.shape[0]

            # insert parameters based on random selected indexes in parent 1
            children[i, crossoverPointIndex1] = parents[
                parent1_index, crossoverPointIndex1
            ]
            # insert parameters based on random selected indexes in parent 1
            children[i, crossoverPointIndex2] = parents[
                parent2_index, crossoverPointIndex2
            ]
        return children

    def mutation(self, crossover):
        # Define minimum and maximum values allowed for each parameter
        minMaxValue = [0.01, 1.0]  # mutation rate

        # Mutation changes a single gene in each offspring randomly.
        mutationValue = 0
        mutationValue = round(random.gauss(0, 0.2), 3)

        # indtroduce mutation by changing one parameter, and set to max or min if it goes out of range
        for idx in range(crossover.shape[0]):
            crossover[idx] = crossover[idx] + mutationValue
            if crossover[idx] > minMaxValue[1]:
                crossover[idx] = minMaxValue[1]
            if crossover[idx] < minMaxValue[0]:
                crossover[idx] = minMaxValue[0]
        return crossover

