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


    parent1_duplicate_missing=dict(zip(p2_col_copy, p1_col_copy))
    parent2_duplicate_missing=dict(zip(p1_col_copy, p2_col_copy))

    def fix_global_duplicates(offspring,parent_duplicate_missing,nlines,ncols,swapped_col_index):
        total_vals = nlines * ncols
        all_vals = set(range(total_vals))

        for i in range(nlines):
            for j in range(ncols):
                val = offspring[i][j]
                if j == swapped_col_index:
                    continue  # don't change the swapped column
                if val in parent_duplicate_missing:
                    offspring[i][j] = parent_duplicate_missing[val]
                

        flat = [val for row in offspring for val in row]



        return offspring

        
    offspring1_repr = fix_global_duplicates(offspring1_repr,parent1_duplicate_missing,nlines, ncols, col)
    offspring2_repr = fix_global_duplicates(offspring2_repr,parent2_duplicate_missing,nlines, ncols, col)

   
    return offspring1_repr, offspring2_repr








#crossover 19 de maio

def swap_time_slots_crossover1(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):
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


    parent1_duplicate_missing=dict(zip(p2_col, p1_col))
    parent2_duplicate_missing=dict(zip(p1_col, p2_col))

    def fix_global_duplicates(offspring,parent_duplicate_missing,nlines,ncols,swapped_col_index):
        total_vals = nlines * ncols
        all_vals = set(range(total_vals))

        for i in range(nlines):
            for j in range(ncols):
                val = offspring[i][j]
                if j == swapped_col_index:
                    continue  # don't change the swapped column
                if val in parent_duplicate_missing:
                    offspring[i][j] = parent_duplicate_missing[val]
                

        flat = [val for row in offspring for val in row]

        if set(flat) != set(all_vals):
            raise ValueError("Duplicates and missing values still exist")

        return offspring


        
    offspring1_repr = fix_global_duplicates(offspring1_repr,parent1_duplicate_missing,nlines, ncols, col)
    offspring2_repr = fix_global_duplicates(offspring2_repr,parent2_duplicate_missing,nlines, ncols, col)

   
    return offspring1_repr, offspring2_repr























def swap_time_slots_crossover0(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):
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


    def fix_global_duplicates(offspring,nlines,ncols,swapped_col_index):
        total_vals = nlines * ncols
        all_vals = set(range(total_vals))
        flat = [val for row in offspring for val in row]
        count=Counter(flat)
        missing = list(all_vals - set(flat))

        for i in range(nlines):
            for j in range(ncols):
                val = offspring[i][j]
                if j == swapped_col_index:
                    continue  # don't change the swapped column
                if count[val] > 1:
                    if not missing:
                        raise ValueError("Not enough missing values to resolve duplicates")
                    new_val = missing.pop(0)
                    offspring[i][j] = new_val
                    count[val] -= 1
                    count[new_val] += 1

        # flat = [val for row in offspring for val in row]
        # aux_list=[]
        # for i, value in enumerate(flat):
        #     if value not in aux_list:
        #         aux_list.append(value)
        #     else:
        #         print(f"value: {value}, index: {i}")
        # print(flat)

        return offspring


        
    offspring1_repr = fix_global_duplicates(offspring1_repr,nlines, ncols, col)
    offspring2_repr = fix_global_duplicates(offspring2_repr,nlines, ncols, col)

   
    return offspring1_repr, offspring2_repr





def swap_time_slots_crossover1(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):
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
    

    total_vals = nlines * ncols
    all_vals = set(range(total_vals))

    def fix_global_duplicates(matrix):
        flat = [val for row in matrix for val in row]
        freq = Counter(flat)
        # duplicates = [val for val, count in freq.items() if count > 1]
        missing = list(all_vals - set(flat))

        i_missing = 0
        used = set()
        for row in range(nlines):
            for col in range(ncols):
                val = matrix[row][col]
                if freq[val] > 1 and (row, col) not in used:
                    matrix[row][col] = missing[i_missing]
                    freq[val] -= 1
                    freq[missing[i_missing]] = 1
                    i_missing += 1
                    used.add((row, col))

        return matrix
    
    offspring1_repr = fix_global_duplicates(offspring1_repr)
    offspring2_repr = fix_global_duplicates(offspring2_repr)


    return offspring1_repr, offspring2_repr




def new_crossover(parent1_repr: list[list[int]], parent2_repr: list[list[int]]):
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
        tuple: Two offspring representations resulting from the crossover.
    """
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

    # Resolve duplicates introduced by the swap
    while len(p1_col) != 0:
        n1 = p1_col[0]
        n2 = p2_col[0]
        for line in range(nlines):
            for column in range(ncols):
                if column == col:
                    continue
                else:
                    if offspring1_repr[line][column] == n2:
                        offspring1_repr[line][column] = p1_col[0]
                    if offspring2_repr[line][column] == n1:
                        offspring2_repr[line][column] = p2_col[0]
        p1_col = p1_col[1:]
        p2_col = p2_col[1:]


    all_vals=list(range(35))
    flat1 = [val for row in offspring1_repr for val in row]
    aux_list=[]
    for i, value in enumerate(flat1):
        if value not in aux_list:
            aux_list.append(value)
            print(f"value {value} in index {i} ok")
        else:
            print(f"value: {value}, index: {i}")
    
    print(flat1)
    print(set(flat1))
    print(all_vals)
    # if set(flat1) != all_vals:
    #     raise ValueError("Duplicates and missing values still exist1")
    

    flat2 = [val for row in offspring2_repr for val in row]
    aux_list=[]
    for i, value in enumerate(flat2):
        if value not in aux_list:
            aux_list.append(value)
        else:
            print(f"value: {value}, index: {i}")
    
    print(flat2)
    



    return offspring1_repr, offspring2_repr