# Author: Yonis Ismail

# Module: Biocomputation

# TItle: Simple genetic algorithm using roulette wheel selection

#################################################################################

import random


class individual:
    gene = []
    fitness = 0

    def __repr__(self):
        return str(individual.fitness)


population = []
offspring = []

p = 50
n = 10


# Adding total fitnesses.
def fitnessfunction(pieceofgene):
    fitness = 0
    for j in range(0, n):
        if pieceofgene[j] == 1:
            fitness = fitness + 1
    return fitness


# Generating random fitnesses adding them to list and assigning the
# fitness to the gene and appending to population.
for x in range(0, p):
    tempgene = []
    for x in range(0, n):
        random1 = random.randint(0, 1)
        tempgene.append(random1)
    # Computing fitness for this gene.
    val_fit = fitnessfunction(tempgene)
    newind = individual()
    newind.gene = tempgene.copy()
    newind.fitness = val_fit
    population.append(newind)

def roundhousewheelselection(population):
    offspringpopulation = []
    P = len(population)

    for i in range(P):
        parent1 = random.randint(0, P - 1)
        off1 = population[parent1]
        parent2 = random.randint(0, P - 1)
        off2 = population[parent2]
        if off1.fitness > off2.fitness:
            offspringpopulation.append(off1)
        else:
            offspringpopulation.append(off2)

    return offspringpopulation




#call selection function

offspringpopulation = roundhousewheelselection(population)

# Printing gene/fitness and then total fitness.
totalFitness = 0
totalFitnessOffspring = 0
for i in range(0, p):
    totalFitnessOffspring = totalFitnessOffspring + offspringpopulation[i].fitness
    totalFitness = totalFitness + population[i].fitness
    print(offspringpopulation[i].gene)
    print(offspringpopulation[i].fitness)

print("Population fitness:", totalFitnessOffspring)
print("Total fitness:", totalFitness)

######################################################################

