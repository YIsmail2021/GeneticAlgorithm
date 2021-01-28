# author: Yonis Ismail

# Title: Simple genetic algorithm
# Student Number: 18022357
# Tasks: Submit

import math
import random
import matplotlib.pyplot as plt
import numpy as np
P = 50
N = 10

population = []
offspringpopulation = []
Mean = []
Mean3 = []
class individual:
    gene = []
    fitness = 0

def create_gene(N):
    tempgene = []
    for i in range(0,N):
        random1 = random.uniform(-32.0, 32.0)
        tempgene.append(random1)
    return tempgene

def func_fit(pieceofgene):
    # Coursework fitness function
    SumXsquared = 0.0
    SumCoscalculation = 0.0
    for i in pieceofgene:
        SumXsquared += i ** 2.0
        SumCoscalculation += math.cos(2.0 * math.pi * i)
    n = len(pieceofgene)
    fitness = -20.0 * math.exp(-0.2 * math.sqrt(SumXsquared / n)) - math.exp(SumCoscalculation / n)
    return fitness
# Create population
for x in range(0, P):
    tempgene = []
    for x in range(0, N):
        random1 = random.uniform(-32.0, 32.0)#4th dec
        tempgene.append(random1)
    # Computing fitness for this gene.
    val_fit = func_fit(tempgene)
    newind = individual()
    newind.gene = tempgene.copy()
    newind.fitness = val_fit
    population.append(newind)

#Tournament Selection
def tournamentselection(population):
    # pick to random individuals
    offspringpopulation = []
    for i in range(P):
        parent1 = random.randint(0, P - 1)
        off1 = population[parent1]
        parent2 = random.randint(0, P - 1)
        off2 = population[parent2]
        # tournament
        if off1.fitness < off2.fitness:
           offspringpopulation.append(off1)
        else:
            offspringpopulation.append(off2)
    return offspringpopulation

# Crossover the population
def crossover(offspring):

    offspring_crossed = []
    for i in range(0, P, 2):
        newind1 = individual()
        newind2 = individual()

        crosspoint = random.randint(1, N - 1)
        headgene1 = []
        tailgene1 = []
        headgene2 = []
        tailgene2 = []
        for h in range(0, crosspoint):
            headgene1.append(offspring[i].gene[h])
            headgene2.append(offspring[i + 1].gene[h])

        for j in range(crosspoint, len(offspring[0].gene)):
            tailgene1.append(offspring[i].gene[j])
            tailgene2.append(offspring[i + 1].gene[j])
        newind1.gene = headgene1 + tailgene2
        newind1.fitness = func_fit(newind1.gene)
        newind2.gene = headgene2 + tailgene1
        newind2.fitness = func_fit(newind2.gene)
        offspring_crossed.append(newind1)
        offspring_crossed.append(newind2)
    return offspring_crossed


# Mutation of the offspring poppulation
def mutation(offspringpopulation):
    pop = []
    for i in range(0, P):
        newind = individual()
        newind.gene = []
        sign = random.choice([-1,1])
        alter = sign*random.uniform(0,1)
        MUTRATE = 1/N
        for j in range(0, N):
            gene = offspringpopulation[i].gene[j]
            mutprob = random.randint(0,100)
            if mutprob < (100 * MUTRATE):
                if gene + alter < -32.0:
                    gene = -32.0
                elif gene + alter > 32.0:
                    gene = 32.0
                else:
                    gene = gene + alter
            newind.gene.append(gene)
            newind.fitness = func_fit(newind.gene)
        pop.append(newind)
    return pop

def roulettewheelselection(population): # Roulette wheel selection
   offspringpopulation = []
   max_of_fitness = (max([ind.fitness for ind in population]))
   prev_prob = 0.0
   list_shares = []
   for i in population:
       # Changed from i.fitness/sum_of_fitness to max_of_fitness - i.fitness
       current_share = prev_prob + (max_of_fitness - i.fitness)
       list_shares.append(current_share)
       prev_prob = current_share
   array_shares = np.array(list_shares)
   for i in range(0,P):
       random_number = (random.uniform(0,max(list_shares)))
       array_index = np.where(array_shares >= random_number)
       if len(array_index)>0:
           if len(array_index[0])>0:
               offspringpopulation.append(population[array_index[0][0]])
   return offspringpopulation



list_best_each_generation = []


for i in range (0,500): # main loop, total generation
    offspringpopulation = tournamentselection(population) # Call Roulettewheelselection
    population = crossover(offspringpopulation) # Crossover the population
    population = mutation(population)  # Mutate the population
    population = tournamentselection(population)
    # we want to select the better ones to do statistical calculation
    # Obtain Best in gen
    bestingen = (min([ind.fitness for ind in population]))#Changed from MAX to MIN for minimisation
    list_best_each_generation.append(bestingen)
    #Obtain mean
    Mean = []
    Mean.append([ind.fitness for ind in population])
    Mean2 = sum(Mean[0]) / P
    Mean3.append(Mean2)

list_x = range(0, len(Mean3))
plt.xlabel("Number of generation")
plt.ylabel("fitness")
plt.title("Genetic Algorithm")
markers_on_mean = Mean3
plt.plot(list_x, Mean3, label = "Mean fitness")
markers_on_Indfit = bestingen
plt.plot(list_x, list_best_each_generation, label = "Best fitness")
plt.legend()
plt.show()