import random

def standard_crossover(parent1_repr, parent2_repr):
    """
    Performs standard one-point crossover on two parent representations.

    This operator selects a random crossover point (not at the edges) and 
    exchanges the tail segments of the two parents to produce two offspring. 
    The crossover point is the same for both parents and ensures at least one 
    gene is inherited from each parent before and after the point.

    Parameters:
        parent1_repr (str or list): The first parent representation.
        parent2_repr (str or list): The second parent representation.
            Both parents must have the same length and type.

    Returns:
        tuple: A pair of offspring representations (offspring1, offspring2), 
        of the same type as the parents.

    Raises:
        ValueError: If parent representations are not the same length.
    """

    if not (isinstance(parent1_repr, list) or isinstance(parent1_repr, str)):
        raise ValueError("Parent 1 representation must be a list or a string")
    if not (isinstance(parent2_repr, list) or isinstance(parent2_repr, str)):
        raise ValueError("Parent 1 representation must be a list or a string")
    if len(parent1_repr) != len(parent2_repr):
        raise ValueError("Parent 1 and Parent 2 representations must be the same length")

    # Choose random crossover point
    xo_point = random.randint(1, len(parent1_repr) - 1)

    offspring1_repr = parent1_repr[:xo_point] + parent2_repr[xo_point:]
    offspring2_repr = parent2_repr[:xo_point] + parent1_repr[xo_point:]

    return offspring1_repr, offspring2_repr

def cycle_crossover(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):
    """
    Performs Cycle Crossover (CX) between two parents

    Cycle Crossover preserves the position of elements by identifying a cycle
    of indices where the values from each parent will be inherited by each offspring.
    The remaining indices are filled with values from the other parent, maintaining valid permutations.

    Args:
        parent1_repr (list of lists): The first parent representation.
        parent2_repr (list of lists): The second parent representation.
            Both parents must have the same length and type.

    Returns:
        tuple: Two offspring permutations resulting from the crossover.
    """
    rows = len(parent1_repr)
    cols = len(parent1_repr[0])

    #flatten
    parent1_flat = [idx for row in parent1_repr for idx in row]
    parent2_flat = [idx for row in parent2_repr for idx in row]

    # Randomly choose a starting index for the cycle
    initial_random_idx = random.randint(0, len(parent1_flat)-1)

    # Initialize the cycle with the starting index
    cycle_idxs = [initial_random_idx]
    current_cycle_idx = initial_random_idx

    # Traverse the cycle by following the mapping from parent2 to parent1
    while True:
        value_parent2 = parent2_flat[current_cycle_idx]
        # Find where this value is in parent1 to get the next index in the cycle
        next_cycle_idx = parent1_flat.index(value_parent2)

        # Closed the cycle -> Break
        if next_cycle_idx == initial_random_idx:
            break

        cycle_idxs.append(next_cycle_idx)
        current_cycle_idx = next_cycle_idx
    
    offspring1_repr = []
    offspring2_repr = []
    
    for idx in range(len(parent1_flat)):
        if idx in cycle_idxs:
            # Keep values from parent1 in offspring1 in the cycle indexes
            offspring1_repr.append(parent1_flat[idx])
            # Keep values from parent2 in offspring2 in the cycle indexes
            offspring2_repr.append(parent2_flat[idx])
        else:
            # Swap elements from parents in non-cycle indexes
            offspring1_repr.append(parent2_flat[idx])
            offspring2_repr.append(parent1_flat[idx])


    # To keep the same type as the parents representation
    offspring1_repr = [offspring1_repr[i*cols:(i+1)*cols] for i in range(rows)]
    offspring2_repr = [offspring2_repr[i*cols:(i+1)*cols] for i in range(rows)]

    return offspring1_repr, offspring2_repr  

def partially_matched_crossover(parent1_repr, parent2_repr):
    """
    Performs Partially Matched Crossover (PMX) between two parents.

    Partially Matched Crossover (PMX) is a crossover technique commonly used in permutation-based problems.
    It ensures that the offspring generated are valid permutations by exchanging portions of the parents' genes.
    PMX works by selecting two random crossover points within the parent representations, and exchanging the genetic material between these points.
    The process uses a matching mechanism to ensure that the offspring does not contain repeated or missing genes.
    This is achieved by creating a mapping of the genes from one parent to the other in the crossover region, and applying this mapping to fill the remaining spots in the offspring.

    Args:
        parent1_repr (list): The representation of the first parent, typically a list of values.
        parent2_repr (list): The representation of the second parent, also a list of values.
            Both parents must have the same length and type (typically, both are lists of integers or genes).
        
    Returns:
        tuple: A pair of offspring generated from the crossover. Both offspring will be valid permutations,
    """

    if len(parent1_repr) != len(parent2_repr):
        raise ValueError("Both parents must have the same number of elements.")

    # Choose two crossover points
    idx1 = random.randint(0, len(parent1_repr) - 1)
    idx2 = random.randint(idx1 + 1, len(parent1_repr))  # Ensure idx2 is greater than idx1
    
    # Initialize offspring representations
    offspring1_repr = [None] * len(parent1_repr)
    offspring2_repr = [None] * len(parent2_repr)

    # Copy the crossover subsequence directly from the parents to the offspring
    offspring1_repr[idx1:idx2] = parent2_repr[idx1:idx2]
    offspring2_repr[idx1:idx2] = parent1_repr[idx1:idx2]
    
    # Create mappings between parents
    mapping1 = {parent1_repr[i]: parent2_repr[i] for i in range(idx1, idx2)}
    mapping2 = {parent2_repr[i]: parent1_repr[i] for i in range(idx1, idx2)}
    
    # Fill the offspring with the remaining values, ensuring a valid permutation
    def fill_offspring(offspring_repr, parent_repr, mapping, idx1, idx2):
        for i in range(len(parent_repr)):
            if i < idx1 or i >= idx2:
                if offspring_repr[i] is None:
                    gene = parent_repr[i]
                    # Correct duplicated values using the mapping
                    while gene in mapping.values():
                        gene = mapping[gene]
                    offspring_repr[i] = gene

    # Fill the offspring with valid values
    fill_offspring(offspring1_repr, parent1_repr, mapping1, idx1, idx2)
    fill_offspring(offspring2_repr, parent2_repr, mapping2, idx1, idx2)
    
    return offspring1_repr, offspring2_repr


