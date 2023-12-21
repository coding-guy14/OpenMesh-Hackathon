from llm_model import GPT2Model
import genetic_algorithm
import unittest

"""
A test case for the GeneticAlgorithm class.
This test case includes various test methods to validate the functionality of the GeneticAlgorithm class.
"""
class TestGeneticAlgorithm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize anything that is required for all tests
        cls.model = GPT2Model()

    def test_gpt2_prediction(self):
        prompt = "Analyze sentiment: This is a great day."
        sentiment = self.model.predict_sentiment(prompt)
        self.assertIn(sentiment, ["Positive", "Negative", "Neutral"], "Sentiment not recognized")

    def test_fitness_function(self):
        prompt = genetic_algorithm.PromptChromosome(["Analyze sentiment:", "This is a great day."])
        fitness_score = genetic_algorithm.evaluate_prompt(self.model, prompt, "Positive")
        self.assertIsInstance(fitness_score, int, "Fitness score is not an integer") 

    
    def test_initial_population(self):
        ga = genetic_algorithm.GeneticAlgorithm(population_size=10, mutation_rate=0.1)
        self.assertEqual(len(ga.population), 10, "Initial population size is incorrect")

    def test_crossover(self):
        parent1 = genetic_algorithm.PromptChromosome(["Analyze sentiment:", "This is a great day."])
        parent2 = genetic_algorithm.PromptChromosome(["Analyze sentiment:", "This is a bad day."])
        child = genetic_algorithm.crossover(parent1, parent2)
        self.assertIsInstance(child, genetic_algorithm.PromptChromosome, "Crossover does not produce a PromptChromosome instance")

    def test_mutation(self):
        chromosome = genetic_algorithm.PromptChromosome(["Analyze sentiment:", "This is a great day."])
        genetic_algorithm.mutate(chromosome, 1.0)  # Force mutation
        self.assertNotEqual(chromosome.genes, ["Analyze sentiment:", "This is a great day."], "Mutation did not alter the chromosome")

    def test_generation_evolution(self):
        ga = genetic_algorithm.GeneticAlgorithm(population_size=10, mutation_rate=0.1)
        ga.run_generation(self.model, "Positive")
        self.assertEqual(len(ga.population), 10, "Population size changed after running a generation")


if __name__ == '__main__':
    unittest.main()
