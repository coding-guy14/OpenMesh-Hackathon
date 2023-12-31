import random
from typing import List, Tuple
from llm_model import GPT2Model

#region Chromosome and Gene Definition
"""
Represents a chromosome in a genetic algorithm.
Attributes:
    genes (List[str]): The genes of the chromosome.
"""
class PromptChromosome:

    def __init__(self, genes: List[str]):
        self.genes = genes

    def __str__(self):
        return " ".join(self.genes)
#endregion


#region Fitness Function
"""
Evaluates the sentiment prediction of a given prompt using a GPT2 model.
Args:
    model (GPT2Model): The GPT2 model used for sentiment prediction.
    prompt (PromptChromosome): The prompt to be evaluated.
    expected_output (str): The expected sentiment output.
Returns:
    float: The evaluation score. Returns 1 if the sentiment prediction matches the expected output, 0 otherwise.
"""
def evaluate_prompt(model: GPT2Model, prompt: PromptChromosome, expected_output: str) -> float:
    response = model.predict_sentiment(str(prompt))
    sentiment_score = 1 if response == expected_output else 0

    output_length = len(response.split())
    word_diversity = len(set(response.split())) / output_length

    final_score = 0.6 * sentiment_score + 0.2 * output_length/50 + 0.2 * word_diversity
    return final_score
#endregion


#region Crossover Algorithm
"""
Perform crossover operation between two parent chromosomes.
Args:
    parent1 (PromptChromosome): The first parent chromosome.
    parent2 (PromptChromosome): The second parent chromosome.
Returns:
    PromptChromosome: The offspring chromosome generated after crossover.
"""
def crossover(parent1: PromptChromosome, parent2: PromptChromosome) -> PromptChromosome:
    if random.random() < 0.5:
        points = sorted(random.sample(range(len(parent1.genes)), 2))
        child_genes = parent1.genes[:points[0]] + parent2.genes[points[0]:points[1]] + parent1.genes[points[1]:]
    else:
        child_genes = [random.choice([gene1, gene2]) for gene1, gene2 in zip(parent1.genes, parent2.genes)]
    return PromptChromosome(child_genes)

#endregion


#region Mutation Algorithm
"""
Mutates the genes of a chromosome based on the mutation rate.
Args:
    chromosome (PromptChromosome): The chromosome to be mutated.
    mutation_rate (float): The probability of mutation for each gene.

"""
def mutate(chromosome: PromptChromosome, mutation_rate: float) -> None:
    for i in range(len(chromosome.genes)):
        if random.random() < mutation_rate:
            chromosome.genes[i] = random.choice(chromosome.genes[i].split())
#endregion


#region Genetic Algorithm Core
# Genetic Algorithm
class GeneticAlgorithm:
    """
    Initializes a GeneticAlgorithm object.
    Args:
        population_size (int): The size of the population.
        mutation_rate (float): The rate of mutation.
    """
    def __init__(self, population_size: int, mutation_rate: float):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()


    """
    Initializes the population with PromptChromosome objects.
    Returns:
        List[PromptChromosome]: The initialized population.
    """
    def initialize_population(self) -> List[PromptChromosome]:
        return [PromptChromosome(["Analyze sentiment:", "This is a great day."]) for _ in range(self.population_size)]  


    """
    Selects two parents from the population.
    Returns:
        Tuple[PromptChromosome, PromptChromosome]: The selected parents.
    """
    def select_parents(self) -> Tuple[PromptChromosome, PromptChromosome]:
        return random.sample(self.population, 2)

    """
    Runs a generation of the genetic algorithm.
    Args:
        model (GPT2Model): The GPT2 model.
        expected_output (str): The expected output.
    """
    def run_generation(self, model: GPT2Model, expected_output: str):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents()
            child = crossover(parent1, parent2)
            mutate(child, self.mutation_rate)
            new_population.append(child)

        fitness_scores = [evaluate_prompt(model, p, expected_output) for p in new_population]
        # Sort the population based on fitness scores
        sorted_population = [x for _, x in sorted(zip(fitness_scores, new_population), key=lambda pair: pair[0], reverse=True)]
        self.population = sorted_population
#endregion
