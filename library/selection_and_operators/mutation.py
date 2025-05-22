from copy import deepcopy
import random

def swap_stages_slots_mutation(representation: list[list[int]], mut_prob=0.3, max_window_size=None):

    new_representation = deepcopy(representation)

    if random.random() <= mut_prob:
        stage1=random.randint(0, len(representation)-1)
        stage2=stage1
        slot1=random.randint(0,len(representation[0])-1)
        slot2=slot1

        # to guarantee we have different stages and different time slots
        while((stage1==stage2) | (slot1==slot2)):
            stage2=random.randint(0,len(representation)-1)
            slot2=random.randint(0,len(representation[0])-1)
        
        # the new representation will swap the artists in the 2 stages and slots selected
        new_representation[stage1][slot1]=representation[stage2][slot2]
        new_representation[stage2][slot2]=representation[stage1][slot1]
    return new_representation



def inversion_mutation(representation: list[list[int]], mut_prob=0.3, max_window_size=10):

    new_representation = deepcopy(representation)
    if random.random() <= mut_prob:
        # Flatten
        rows=len(representation)
        columns=len(representation[0])
        new_representation=[idx for row in new_representation for idx in row]
        idx_1=random.randint(0, len(new_representation)-1)
        idx_2=idx_1

        # to guarantee that the indexes are different and do not have a distance greater than window_size
        while((abs(idx_1-idx_2)==0)):
            new_rand_int=random.randint(2, max_window_size)
            if (idx_1+new_rand_int<=34):
                idx_2=idx_1+new_rand_int
            else:
                idx_2=idx_1-new_rand_int
        
        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1
        
        reversed_subsequence = list(reversed(new_representation[idx_1:idx_2]))
        new_representation_final = new_representation[:idx_1] + reversed_subsequence + new_representation[idx_2:]

        # Back to a matrix
        new_representation_return = [new_representation_final[i * columns:(i + 1) * columns] for i in range(rows)]


        return new_representation_return
    return new_representation



def shuffle_mutation(representation: list[list[int]], mut_prob=0.3, max_window_size=5):
    new_representation = deepcopy(representation)

    if random.random() <= mut_prob:
    
        stage1=random.randint(0, len(representation)-1)
        stage2=stage1

        while(stage1==stage2):
            stage2=random.randint(0, len(representation)-1)

        idx_1=random.randint(0, len(representation[0])-1)
        idx_2=idx_1

        # 
        # to guarantee that the indexes are different and do not have a distance greater than 10 
        while((abs(idx_1-idx_2)==0)):
            new_rand_int=random.randint(2, max_window_size)
            if (idx_1+new_rand_int<=34):
                idx_2=idx_1+new_rand_int
            else:
                idx_2=idx_1-new_rand_int
        
        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1

        
        shuffled_subsequence1=representation[stage1][idx_1:idx_2]
        random.shuffle(shuffled_subsequence1)
        shuffled_subsequence2=representation[stage2][idx_1:idx_2]
        random.shuffle(shuffled_subsequence2)

        new_representation[stage1]=representation[stage1][:idx_1] + shuffled_subsequence1 + representation[stage1][idx_2:]
        new_representation[stage2]=representation[stage2][:idx_1] + shuffled_subsequence2 + representation[stage2][idx_2:]


    return new_representation





def mutation_KA(representation: list[list[int]], mut_prob=0.3, max_window_size=None):

    new_representation = deepcopy(representation)

    if random.random() <= mut_prob:
        new_representation_converted = [row.index(1) for row in new_representation]
        
        shift_options = [-2, -1, 1, 2]
        shift = random.choice(shift_options)
    
        n = len(new_representation_converted)
        
        representation_with_shift = [new_representation_converted[(i - shift) % n] for i in range(n)]

        for i, pos in enumerate(representation_with_shift):
            new_representation[i] = [0] * len(new_representation[i])
            new_representation[i][pos] = 1

    return new_representation