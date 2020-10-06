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

    def mutation(self, crossover):
        # Define minimum and maximum values allowed for each parameter
        minMaxValue = [0.01, 1.0]  # mutation rate

        # Mutation changes a single gene in each offspring randomly.
        mutationValue = 0
        mutationValue = round(random.gauss(0, 0.1), 5)

        # indtroduce mutation by changing one parameter, and set to max or min if it goes out of range
        for idx in range(crossover.shape[0]):
            crossover[idx] = crossover[idx] + mutationValue
            if crossover[idx] > minMaxValue[1]:
                crossover[idx] = minMaxValue[1]
            if crossover[idx] < minMaxValue[0]:
                crossover[idx] = minMaxValue[0]
        return crossover


population = np.empty([10, 1])

parents = np.array([[0.53], [0.459], [0.181], [0.967], [0.509]])
print(parents)
children = np.array([[0.50453], [0.43353], [0.15553], [0.94153], [0.48353]])
print(children)
population[0 : parents.shape[0], :] = parents  # fittest parents
population[parents.shape[0] :, :] = children  # children

print(population)
