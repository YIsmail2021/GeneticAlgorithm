
# author: Yonis Ismail

# Title: Simple genetic algorithm

# Tasks: Add mean, Finish  worksheet 3 by monday, work on coursework on Tuesday onwards and finish coursework by 7th.








import random
import matplotlib.pyplot as plt

P = 50

N = 50

population = []

offspringpopulation = []

Mean = []
Mean3 = []
class individual:

    gene = []
    fitness = 0

def create_gene(n):
    tempgene = []

    for i in range(0,n):

        random1 = random.uniform(0,1)

        tempgene.append(random1)

    return tempgene

def func_fit(pieceofgene):
    fitness = 0

    for i in range(0, len(pieceofgene)):

        fitness = fitness + pieceofgene[i]

    return fitness

for x in range(0, P):

    tempgene = []

    for x in range(0, N):

        random1 = random.randint(0, 1)
        tempgene.append(random1)

    # Computing fitness for this gene.

    val_fit = func_fit(tempgene)
    newind = individual()
    newind.gene = tempgene.copy()
    newind.fitness = val_fit
    population.append(newind)

def tournamentselection(population):#haixia: change the name,

    # it is not roulette selection, this is your trounament selection

    # for roulette selection, please refer to assignment sheet



    offspringpopulation = []
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

def crossover(offspring):

    offspring_crossed = []
    for i in range(0, P, 2):

        newind1 = individual()
        newind2 = individual()

        # weirdness of having to use .copy for arrays or
        # it is actually a pointer, not a copy
        crosspoint = random.randint(1, len(offspring[0].gene) - 1)
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
        newind1.fitness = (headgene1 + tailgene2).count(1)
        newind2.gene = headgene2 + tailgene1
        newind2.fitness = (headgene2 + tailgene1).count(1)
        offspring_crossed.append(newind1)
        offspring_crossed.append(newind2)
    return offspring_crossed

# Mutation of the offspring poppulation

def mutation(offspringpopulation):
    pop = []
    for i in range(0, P):

        #newind = offspringpopulation[i]#haixia: if you get the gene from the pop, then,

        # if you append it with newind.gene.append(gene), then, your gene will be growing
        newind = individual()
        newind.gene = []#haixia: this is important: clear it, before you fill it with new elements
        sign = random.choice([-1,1])
        alter = sign*random.uniform(0,1)
        MUTRATE = 1 / N
        for j in range(0, N):
            gene = offspringpopulation[i].gene[j]
            mutprob = random.randint(0,100)
            if mutprob < (100 * MUTRATE):
                if gene + alter < 0:
                    gene = 0
                elif gene + alter > 1:
                    gene = 1
                else:
                    gene = gene + alter
            newind.gene.append(gene)
            newind.fitness = func_fit(newind.gene)#haixia

        pop.append(newind)

    return pop

list_best_each_generation = [] #haixia


list_best_each_generation = [] #haixia

for i in range (0,50): # main loop, total generation



    offspringpopulation = tournamentselection(population) # Call Roulettewheelselection



    population = crossover(offspringpopulation) # Crossover the population



    population = mutation(population)  # Mutate the population



    population = tournamentselection(population) #haixia: after creating new individuals,

    # we want to select the better ones to do statistical calculation



    totalFitness = 0

    totalFitnessOffspring = 0

    bestingen = (max([ind.fitness for ind in population]))

    list_best_each_generation.append(bestingen)  # Changed from MAX to MIN for minimisation

    #for i in range(0, P):

    #    totalFitness = totalFitness + population[i].fitness # Calculate the total fitness of the population

    #print("totalfitness: ", totalFitness) #haixia: after wsheet2, we are not interested in total fitness,

    #instead, get the best fitness from each generation

    Mean = []

    Mean.append([ind.fitness for ind in population])

    Mean2 = sum(Mean[0]) / P

    Mean3.append(Mean2)
print(" The means of each generation: ", Mean3)

print("Best in each generation: ",list_best_each_generation)

list_x = range(0, len(Mean3))
plt.xlabel("Number of generation")
plt.ylabel("fitness")
plt.title("Genetic Algorithm")
#print("Best mean generation:"+str(Mean3))
markers_on_mean = Mean3
plt.plot(list_x, Mean3, label = "Mean fitness")
markers_on_Indfit = bestingen
plt.plot(list_x, list_best_each_generation, label = "Best fitness")
plt.legend()
plt.show()
