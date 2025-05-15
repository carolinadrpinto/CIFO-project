import random


def cycle_crossover(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):

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

def partially_matched_crossover(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):

    rows = len(parent1_repr)
    cols = len(parent1_repr[0])

    # Flatten
    parent1_flat = [idx for row in parent1_repr for idx in row]
    parent2_flat = [idx for row in parent2_repr for idx in row]

    size=len(parent1_flat)

    if size != len(parent2_flat):
        raise ValueError("Both parents must have the same number of elements.")

    # Choose two crossover points
    idx1 = random.randint(0, size - 1)
    idx2 = idx1

    while abs(idx1-idx2)==0:
        idx2=random.randint(0, size - 1)

    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    
    # Initialize offspring representations
    offspring1_repr = [None] * size
    offspring2_repr = [None] * size

    # Copy the crossover subsequence directly from the parents to the offspring
    offspring1_repr[idx1:idx2] = parent2_flat[idx1:idx2]
    offspring2_repr[idx1:idx2] = parent1_flat[idx1:idx2]

    print(offspring1_repr)
    print(offspring2_repr)

    #setting the replacement that should take place if a value is already in the crossover segment
    parent1_mapping = {parent2_flat[i]: parent1_flat[i] for i in range(idx1, idx2)} 
    parent2_mapping = {parent1_flat[i]: parent2_flat[i] for i in range(idx1, idx2)}

    #function that will check each gene and copy it if it doesn't already exist in the crossover segment or replace it if it exists
    def value_checking(parent_mapping, value):
        while value in parent_mapping:
            value = parent_mapping[value]
        return value

    #loop that goes through each gene
    for i in list(range(0, idx1)) + list(range(idx2, size)):
        value_parent1 = value_checking(parent1_mapping, parent1_flat[i])
        value_parent2 = value_checking(parent2_mapping, parent2_flat[i])
        offspring1_repr[i] = value_parent1
        offspring2_repr[i] = value_parent2

    print(set(offspring1_repr))
    print(set(offspring2_repr))

    # Offsprings back to matrix
    offspring1_repr = [offspring1_repr[i*cols:(i+1)*cols] for i in range(rows)]
    offspring2_repr = [offspring2_repr[i*cols:(i+1)*cols] for i in range(rows)]

    print(f"offspring in the end: {offspring1_repr}")
    print(f" offspreing in the end{offspring2_repr}")


    
    return offspring1_repr, offspring2_repr


