#region Header
'''
HACKATHON CHALLENGE 4
CHALLENGE 3 - AUTOMATED PROMPT ENGINEERING CHALLENGE
SOHAM DIVEKAR

TASK:
- Develop a genetic algorithm capable of generating prompts for a large language model (LLM) that will yield desired outputs.
- Select a pre-trained open-source LLM and clearly define both the task you want it to perform and the desired outputs for that task.
- Define the components of your genetic algorithm
  - the space of chromosomes as the potential solutions (prompts);
  - the structure of genes forming the components of a prompt (e.g., keywords, sentences); and 
  - the fitness function that evaluates how close the output of the LLM is to the desired output for a given prompt.
- Ensure that the fitness function is computable in a way that automatic evaluation of the fitness of prompts can be performed by integrating the LLM with your genetic algorithm.
- Design effective crossover and mutation algorithms.
- Optimize the structure and parameters of your genetic algorithm in a test environment of your choosing.
- Integrate your solution into HyperCycle's Computation Node software
'''
#endregion 

#region Imports 
from llm_model import GPT2Model
from genetic_algorithm import GeneticAlgorithm
#endregion

#region Test Environment
"""
This function represents the main entry point of the program.
It initializes a GPT2Model and a GeneticAlgorithm object.
Then, it runs a loop for 5 generations, calling the `run_generation` method of the GeneticAlgorithm object
with the GPT2Model and the prompt "Positive".
Finally, it prints the best prompt of each generation.
"""
def main():
    model = GPT2Model()
    ga = GeneticAlgorithm(population_size=10, mutation_rate=0.1)
    
    for generation in range(5):
        ga.run_generation(model, "Positive")
        print(f"Generation {generation}: Best Prompt - {ga.population[0]}")

if __name__ == "__main__":
    main()
#endregion

#region HyperCycle Implementation
# Additional integration code would be needed here to integrate this solution into HyperCycle's Computation Node software.
# This would depend on the specifics of HyperCycle's software architecture and API.
#endregion
