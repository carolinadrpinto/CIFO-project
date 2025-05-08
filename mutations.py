from copy import deepcopy
import random

def swap_stages_slots_mutation(representation: list[list[int]], mut_prob=0.3):

    new_representation = deepcopy(representation)
    if random.random() <= mut_prob:
        stage1=random.randint(0, len(representation)-1)
        stage2=stage1
        slot1=random.randint(0,len(representation[0])-1)
        slot2=slot1

        # to guarantee we have different stages and different time slots
        while((stage1==stage2) & (slot1==slot2)):
            stage2=random.randint(0,len(representation)-1)
            slot2=random.randint(0,len(representation[0])-1)
        
        # the new representation will swap the artists in the 2 stages and slots selected
        new_representation[stage1][slot1]=representation[stage2][slot2]
        new_representation[stage2][slot2]=representation[stage1][slot1]
    
    new_representation



def inversion_mutation(representation: list[list[int]], mut_prob=0.3, max_idx_distance=10):

    new_representation = deepcopy(representation)
    if random.random() <= mut_prob:
        # Flatten
        rows=len(representation)
        columns=len(representation[0])
        new_representation=[idx for row in new_representation for idx in row]
        idx_1=random.randint(0, len(new_representation)-1)
        idx_2=idx_1

        # to guarantee that the indexes are different and do not have a distance greater than 10 
        while((abs(idx_1-idx_2)==0) | (abs(idx_1-idx_2)>max_idx_distance)):
            idx_2=random.randint(0, len(representation)-1)
        
        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1
        
        reversed_subsequence = list(reversed(new_representation[idx_1:idx_2]))
        new_representation = representation[:idx_1] + reversed_subsequence + representation[idx_2:]

        # Back to a matrix
        new_representation= [new_representation[i * columns:(i + 1) * columns] for i in range(rows)]

    return new_representation




def shuffle_mutation(representation: list[list[int]], mut_prob=0.3, max_window_size=5):
    new_representation = deepcopy(representation)

    if random.random() <= mut_prob:
    
        stage1=random.randint(0, len(representation)-1)
        stage2=stage1

        while(stage1==stage2):
            stage2=random.randint(0, len(representation)-1)

        idx_1=random.randint(0, len(representation)-1)
        idx_2=idx_1

        # 
        while((abs(idx_1-idx_2)==0) | (abs(idx_1-idx_2)>max_window_size)):
            idx_2=random.randint(0, len(representation)-1)
        
        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1
        
        shuffled_subsequence1=representation[stage1][idx_1:idx_2]
        random.shuffe(shuffled_subsequence1)
        shuffled_subsequence2=representation[stage2][idx_1:idx_2]
        random.shuffe(shuffled_subsequence2)

        new_representation[stage1]=representation[stage1][:idx_1] + shuffled_subsequence1 + representation[stage1][idx_2:]
        new_representation[stage2]=representation[stage2][:idx_1] + shuffled_subsequence2 + representation[stage2][idx_2:]


    return new_representation