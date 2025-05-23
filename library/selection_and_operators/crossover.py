import random
from copy import deepcopy
from collections import Counter


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

        # Close the cycle- Break
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


    while abs(idx1-idx2)==0 or abs(idx1-idx2)>int(size//2):
        idx2=random.randint(0, size - 1)

    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    
    # Initialize offspring representations
    offspring1_repr = [None] * size
    offspring2_repr = [None] * size

    # Copy the crossover subsequence directly from the parents to the offspring
    offspring1_repr[idx1:idx2] = parent2_flat[idx1:idx2]
    offspring2_repr[idx1:idx2] = parent1_flat[idx1:idx2]


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


    # Offsprings back to matrix
    offspring1_repr = [offspring1_repr[i*cols:(i+1)*cols] for i in range(rows)]
    offspring2_repr = [offspring2_repr[i*cols:(i+1)*cols] for i in range(rows)]

    
    return offspring1_repr, offspring2_repr


def swap_time_slots_crossover(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):
    """
    Performs a column-based crossover between two parent solutions.

    The function selects a random column and swaps that column between the two parents
    to create two offspring. Then it ensures that each offspring maintains
    valid permutations by replacing any duplicated values (introduced by the swap) with 
    the corresponding values that were eliminated initally.

    Args:
    parent1_repr (list of lists): The first parent representation.
    parent2_repr (list of lists): The second parent representation.
        Both parents must have the same length and type.

    Returns:
        Two offspring representations resulting from the crossover.
    """
    def check_same_dimensions(matrix1, matrix2):
        if len(matrix1) != len(matrix2):
            return False  
        for row1, row2 in zip(matrix1, matrix2):
            if len(row1) != len(row2):
                return False  
        return True
    
    if not check_same_dimensions(parent1_repr, parent2_repr):
        raise ValueError("Both parents must have the same dimensions.")

    offspring1_repr = deepcopy(parent1_repr)
    offspring2_repr = deepcopy(parent2_repr)
    nlines, ncols = len(offspring1_repr), len(offspring1_repr[0])

    # Choose a random column index to swap
    col = random.randint(0, ncols - 1)
    # Extract the column from each parent and store in a list
    p1_col = []
    p2_col = []
    for line in range(nlines):
        p1_col.append(parent1_repr[line][col])
        p2_col.append(parent2_repr[line][col])

    # Swap the selected column between the two offspring
    for line in range(nlines):
        offspring1_repr[line][col] = p2_col[line]
        offspring2_repr[line][col] = p1_col[line]

    p2_col_copy=deepcopy(p2_col)
    p1_col_copy=deepcopy(p1_col)
    for value in p2_col:
        if value in p1_col:
            p2_col_copy.remove(value)
            p1_col_copy.remove(value)


    parent1_duplicate_missing=dict(zip(p2_col_copy, p1_col_copy)) #dictionary that makes a correspondence between the two swapped columns for fixing duplicates
    parent2_duplicate_missing=dict(zip(p1_col_copy, p2_col_copy))

    #function to fix the global duplicates 
    def fix_global_duplicates(offspring,parent_duplicate_missing,nlines,ncols,swapped_col_index):
        total_vals = nlines * ncols
        all_vals = set(range(total_vals))

        for i in range(nlines):
            for j in range(ncols):
                val = offspring[i][j]
                if j == swapped_col_index:
                    continue  # don't change the swapped column
                if val in parent_duplicate_missing:
                    offspring[i][j] = parent_duplicate_missing[val] #change the duplicated values outside the swapped column
                

        flat = [val for row in offspring for val in row]



        return offspring

        
    offspring1_repr = fix_global_duplicates(offspring1_repr,parent1_duplicate_missing,nlines, ncols, col) #final representations
    offspring2_repr = fix_global_duplicates(offspring2_repr,parent2_duplicate_missing,nlines, ncols, col)

   
    return offspring1_repr, offspring2_repr



def crossover_KAP(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):

    rows = len(parent1_repr)
    cols = len(parent1_repr[0])

    parent1_converted = [row.index(1) for row in parent1_repr] #convert to list where artists represent indexes and the values represent their time slot
    parent2_converted = [row.index(1) for row in parent2_repr]

    def slots_artists(parent_converted,cols):
        slot_artists = {i: [] for i in range(cols)}
        for artist_index, slot in enumerate(parent_converted):
            slot_artists[slot].append(artist_index)
        return slot_artists
    
    parent1_correspondence=slots_artists(parent1_converted,cols) #group the artists assigned to each time slot for easier manipulation
    parent2_correspondence=slots_artists(parent2_converted,cols)

    random_slot = random.randint(0, cols - 1) #select random slot

    parent1_artists = parent1_correspondence[random_slot] #get the artists that are in the random slot
    parent2_artists = parent2_correspondence[random_slot]

    offspring1_repr = deepcopy(parent1_converted)
    offspring2_repr = deepcopy(parent2_converted)


    #swaps the columns (random slot) between parents
    for artist in parent2_artists:
        offspring1_repr[artist] = random_slot 

    
    for artist in parent1_artists:
        offspring2_repr[artist] = random_slot

    # Cross-mapping to fix the duplicates
    for a1, a2 in zip(parent1_artists, parent2_artists):
        offspring2_repr[a2] = parent2_converted[a1]
        offspring1_repr[a1] = parent1_converted[a2]

    #back to matrix
    def return_to_binary(index, length):
        row = [0] * length
        row[index] = 1
        return row

    offspring1_repr = [return_to_binary(idx, cols) for idx in offspring1_repr]
    offspring2_repr = [return_to_binary(idx, cols) for idx in offspring2_repr]
        

    return offspring1_repr, offspring2_repr  