import random
from library.solution import Solution
from itertools import combinations
from copy import deepcopy
from data.df_load import artists_list, conflicts_matrix, dic_artists

class LUSolution(Solution):
    def __init__(
        # This class receives information about the artists and the conflicts matrix,
        # Also information about the number of time slots and stages 
        self,
        artists: list[tuple[int]] = artists_list,
        conflicts: list[list[float]] = conflicts_matrix,
        time_slots: int = 7,
        stages: int = 5,
        repr = None,
    ):
        self.artists=artists
        self.conflicts=conflicts
        self.time_slots=time_slots
        self.stages=stages

        if repr:
            repr = self._validate_repr(repr) #criar função

        super().__init__(repr=repr)

    # function to print the solution with the name of the artists per slot per stage
    def __str__(self):
        id_to_artist = {v: k for k, v in dic_artists.items()}
        output = []
        for i, stage in enumerate(self.repr):
            row = [f"stage {i+1}: "]
            for artist_id in stage:
                row.append(id_to_artist[artist_id] + ', ')
            output.append(''.join(row).rstrip(', '))
        return '\n'.join(output)



    def _validate_repr(self, repr):
        # Confirm repr is list
        if isinstance(repr, list) and all(not isinstance(elem, list) for elem in repr):
            print("Changing to list of lists")
            repr=[repr[i * self.time_slots:(i + 1) * self.time_slots] for i in range(self.stages)]
        # Make sure repr is a matrix of integers
        if not all(isinstance(idx, int) for row in repr for idx in row):
            raise TypeError('Representation must be a matrix of integers')
        # Validate matrix lenght
        if (len(repr) != (self.stages)) or (len(repr[0]) != (self.time_slots)):
            raise ValueError("The number of stages and time slots does not match the ones provided")
        # Validate matrix content
        if (set(idx for row in repr for idx in row) != set([i for i in range(len(self.artists))])):
            raise ValueError("Matrix contain repeated artists")
        return repr
    
    def random_initial_representation(self):
        # Generate a random initial matrix 5x7 with numbers from 0 to 34
        repr = []
        artists = list(range(len(self.artists)))
        random.shuffle(artists)
        repr = [artists[i * self.time_slots:(i + 1) * self.time_slots] for i in range(self.stages)]
        return repr

    def popularity_score(self):
        closers_scores = []
        for stage in self.repr:
            closer = stage[-1]        # in every stage get the last artist
            closers_scores.append(self.artists[closer][2])   # get the popularity of the artist
        optimal_pop = sorted([x[2] for x in self.artists], reverse=True)[:self.stages] # obtaining the best possible popularity
                                                                                    # across all possible combinations
        pop_score = sum(closers_scores) / sum(optimal_pop)    # divide the popularity of the closers by the optimal popularity
        return pop_score

    def diversity_score(self):
        diversity_sums = []
        for slot in zip(*self.repr):                   # * to transpose the matrix and analyze each slot
            diff_genres = set()                          # create a new set for each slot
            for artist in slot:
                diff_genres.add(self.artists[artist][1])      # a new entry for every unique genre
            diversity_sums.append(len(diff_genres) / self.stages)    # a score between 0 and 1 for each slot
                                                                # len(diff_genres) is how many unique genres are in the slot
        diversity_score = sum(diversity_sums) / self.time_slots    
        return diversity_score

    def conflict_score(self):    
        conflict_sums = []                       # empty list to store conflicts of every slot
        for slot in zip(*self.repr):           # iterate through slots (columns)
            slot_conflicts = []                        # initialize list for every slot
            for artist1, artist2 in combinations(slot, 2):       # all unique pairs
                slot_conflicts.append(self.conflicts[artist1][artist2])  # get score
            conflict_sums.append(sum(slot_conflicts) / len(slot_conflicts))    # divide by number of combinations to get average
        worst_scenario = max(conflict_sums)
        conflict_score = 1 - (sum(conflict_sums)/ worst_scenario)/ self.time_slots    # 1 - (average all slots / worst score in this solution)
        return conflict_score
            

    def fitness(self):
        fitness = (self.popularity_score()+self.diversity_score()+self.conflict_score()) / 3
        return fitness


class LUGASolution(LUSolution):
    def __init__(
        self,
        crossover_function,
        mutation_function,
        artists: list[tuple[int]]=artists_list,
        conflicts: list[list[float]] = conflicts_matrix,
        time_slots: int = 7,
        stages: int = 5,
        repr = None,
    ):

        super().__init__(
            artists=artists,
            conflicts=conflicts,
            time_slots=time_slots,
            stages=stages,
            repr=repr,
            )
        
        self.mutation_function=mutation_function
        self.crossover_function=crossover_function

    # define functions for mutation and crossover inside the class solution
    # the class gets more customizable and we can use different crossover and mutation functions

    def mutation(self, mut_prob=0.3, max_window_size=10):

            new_repr = self.mutation_function(self.repr, mut_prob, max_window_size)

            return LUGASolution(
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                artists=self.artists,
                conflicts=self.conflicts,
                time_slots=self.time_slots,
                stages=self.stages,
                repr=new_repr
            )

    def crossover(self, other_solution):
        offspring1_repr, offspring2_repr = self.crossover_function(self.repr, other_solution.repr)
        return(
            LUGASolution(
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            artists=self.artists,
            conflicts=self.conflicts,
            time_slots=self.time_slots,
            stages=self.stages,
            repr=offspring1_repr,
        ),
        LUGASolution(
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            artists=self.artists,
            conflicts=self.conflicts,
            time_slots=self.time_slots,
            stages=self.stages,
            repr=offspring2_repr,
        )
        )



class LUKAPSolution(Solution):
    def __init__(
        self,
        artists: list[tuple[int]] = artists_list,
        conflicts: list[list[float]] = conflicts_matrix,
        time_slots: int = 7,
        stages: int = 5,
        repr = None,
    ):
        self.artists=artists
        self.conflicts=conflicts
        self.time_slots=time_slots
        self.stages=stages

        if repr:
            repr = self._validate_repr(repr) #criar função

        super().__init__(repr=repr)

    def __str__(self):
        output = []
        for i, stage in enumerate(self.repr, start=1):
            artist_names = [self.artists[artist_idx][-1] for artist_idx in stage]
            output.append(f"Stage {i} lineup: {', '.join(artist_names)}")
        return '\n'.join(output)


    def _validate_repr(self, repr):
        # Confirm repr is list
        if isinstance(repr, list) and all(not isinstance(elem, list) for elem in repr):
            print("Changing to list of lists")
            repr=[repr[i * self.time_slots:(i + 1) * self.time_slots] for i in range(self.stages)]

        # Make sure repr is a matrix of integers
        if not all(isinstance(bit, int) for row in repr for bit in row):
            raise TypeError('Representation must be a matrix of integers')
        
        # Validate matrix lenght
        if (len(repr) != (len(self.artists)) or (len(repr[0]) != (self.time_slots))):
            print(len(repr))
            print(len(self.artists))
            print(len(repr[0]))
            raise ValueError("The number of artists and time slots does not match the ones provided")
        
        # Per column we only have ones equal to the number of stages
        for i in range(self.time_slots):
            sum_rows=0
            for j in range(len(self.artists)):
                sum_rows+=repr[j][i]
            if sum_rows!=self.stages:
                print(sum_rows)
                print(self.stages)
                raise ValueError("There are more/less artists than stages in the same timeslot")
        
        # Per row we can only have one number one, that is the time slot that artist is assigned to
        for j in range(len(self.artists)):
            sum_col=0
            for i in range(self.time_slots):
                sum_col+=repr[j][i]
            if sum_col!=1:
                raise ValueError("The artists is not assigned or is assigned to more than 1 time slot")
            
        return repr
    
    # This representation is a matrix with the rows as the number of artists and the columns as the number of time slots
    # Each artists is assigned one time slot and each time slots has 5 artists
    def random_initial_representation(self):
        repr = [[0 for _ in range(self.time_slots)] for _ in range(len(self.artists))] # initialize matrix of zeros
        column_counts = [0] * self.time_slots

        # put ones in the matrix
        for i in range(len(self.artists)):
            valid_columns = [j for j in range(self.time_slots) if column_counts[j] < 5] # valid columns in which i can put the ones
            rand_col = random.choice(valid_columns) # select a random column from the ones valids
            repr[i][rand_col] = 1 # put the one in the matrix in the position selected
            column_counts[rand_col] += 1 # update the counts to keep trach of the number of ones in each column
          
        return repr

    def popularity_score(self):
        closers_scores = []
        for i in range(len(self.repr)):
            if self.repr[i][self.time_slots-1]==1:
                closers_scores.append(self.artists[i][2])   # get the popularity of the artist that are in the last time slot
        optimal_pop = sorted([x[2] for x in self.artists], reverse=True)[:self.stages] # obtaining the best possible popularity
                                                                                       # across all possible combinations
        pop_score = sum(closers_scores) / sum(optimal_pop)    # divide the popularity of the closers by the optimal popularity
        return pop_score

    def diversity_score(self):
        diversity_sums = []
        for slot in zip(*self.repr):                   # * to transpose the matrix and analyze each slot
            diff_genres = set()
            scheduled_artists = [artist_idx for artist_idx, val in enumerate(slot) if val == 1] # see which artists are in this time slot
            for artist in scheduled_artists:
                diff_genres.add(self.artists[artist][1])      # a new entry for every unique genre
            diversity_sums.append(len(diff_genres) / self.stages)    # a score between 0 and 1 for each slot
                                                                   # len(diff_genres) is how many unique genres are in the slot
        diversity_score = sum(diversity_sums) / self.time_slots    
        return diversity_score

    def conflict_score(self):    
        conflict_sums = []                       # empty list to store conflicts of every slot
        for slot in zip(*self.repr):           # iterate through slots (columns)
            slot_conflicts = []
            scheduled_artists = [artist_idx for artist_idx, val in enumerate(slot) if val == 1] # see which artists are in this time slot
            for artist1, artist2 in combinations(scheduled_artists, 2):       # all unique pairs
                slot_conflicts.append(self.conflicts[artist1][artist2])  # get score
            conflict_sums.append(sum(slot_conflicts) / len(slot_conflicts))    # divide by number of combinations to get average
        worst_scenario = max(conflict_sums)
        conflict_score = 1 - (sum(conflict_sums)/ worst_scenario)/ self.time_slots    # 1 - (average all slots / worst score in this solution)
        return conflict_score
            
    def fitness(self):
        fitness = (self.popularity_score()+self.diversity_score()+self.conflict_score()) / 3
        return fitness
    


class LUKAPGASolution(LUKAPSolution):
    def __init__(
        self,
        crossover_function,
        mutation_function,
        artists: list[tuple[int]]=artists_list,
        conflicts: list[list[float]] = conflicts_matrix,
        time_slots: int = 7,
        stages: int = 5,
        repr = None,
    ):

        super().__init__(
            artists=artists,
            conflicts=conflicts,
            time_slots=time_slots,
            stages=stages,
            repr=repr,
            )
        
        self.mutation_function=mutation_function
        self.crossover_function=crossover_function

    # the mutation function and the crossover function are attributes of the class
    # when they are called they return the representation and then they should be transformed in a Solution
    def mutation(self, mut_prob=0.3, max_window_size=10):

            new_repr = self.mutation_function(self.repr, mut_prob, max_window_size)

            return LUKAPGASolution(
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                artists=self.artists,
                conflicts=self.conflicts,
                time_slots=self.time_slots,
                stages=self.stages,
                repr=new_repr
            )

    def crossover(self, other_solution):
        offspring1_repr, offspring2_repr = self.crossover_function(self.repr, other_solution.repr)
        return(
            LUKAPGASolution(
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            artists=self.artists,
            conflicts=self.conflicts,
            time_slots=self.time_slots,
            stages=self.stages,
            repr=offspring1_repr,
        ),
        LUKAPGASolution(
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            artists=self.artists,
            conflicts=self.conflicts,
            time_slots=self.time_slots,
            stages=self.stages,
            repr=offspring2_repr,
        )
        )


class LUSASolution(LUSolution):
    # the solution for simulated need a function named get random neighbor to get a random neigbor of the current solution
    def get_random_neighbor(self):
        """Random neighbor is obtained by flatten the matrix and swap 2 consecutive artists"""
        flatten_sol = [idx for row in self.repr for idx in row]

        # Choose an artist idx to switch with the next artist in the flatten matrix
        random_artist_idx = random.randint(0, len(self.artists)-2)

        flatten_new_sol = deepcopy(flatten_sol)
        flatten_new_sol[random_artist_idx] = flatten_sol[random_artist_idx+1]
        flatten_new_sol[random_artist_idx+1] = flatten_sol[random_artist_idx]

        # back to matrix
        new_sol = [flatten_new_sol[i *  self.time_slots:(i + 1) * self.time_slots] for i in range(self.stages)]

        return LUSASolution(repr=new_sol)
    



class LUHCSolution(LUSolution):
    # to use the HC we need to define a neigborhood
    def get_neighbors(self):
        """Neighbors are obtained by reversing the artists in one stage"""
        neighbors = []
        for i in range(len(self.repr)):
            new_sol = deepcopy(self.repr)
            new_sol[i] = new_sol[i][-1:] + new_sol[i][:-1]
            neighbor = LUHCSolution(repr=new_sol)
            neighbors.append(neighbor)
        return neighbors



class LUKAPHCSolution(LUKAPSolution):
    # in this neighborhood we shift the matrix one or tow positions to the right, to the left, up or down
    def get_neighbors(self):

        def shift_left(matrix, positions=1):
            return [row[positions:] + row[:positions] for row in matrix]

        def shift_right(matrix, positions=1):
            return [row[-positions:] + row[:-positions] for row in matrix]

        def shift_up(matrix, positions=1):
            return matrix[positions:] + matrix[:positions]

        def shift_down(matrix, positions=1):
            return matrix[-positions:] + matrix[:-positions]


        neighbors = []
        shift_functions = [shift_left, shift_right, shift_up, shift_down]

        for shift_fn in shift_functions:
            for k in [1, 2]:  # shift by 1 and 2
                new_repr = shift_fn(self.repr, positions=k)
                neighbor = LUKAPHCSolution(repr=new_repr)
                neighbors.append(neighbor)

        return neighbors



class LUKAPSASolution(LUKAPSolution):
    # get a random neighbor based on the neighborhood defined previously
    def get_random_neighbor(self):

        def shift_left(matrix, positions=1):
            return [row[positions:] + row[:positions] for row in matrix]

        def shift_right(matrix, positions=1):
            return [row[-positions:] + row[:-positions] for row in matrix]

        def shift_up(matrix, positions=1):
            return matrix[positions:] + matrix[:positions]

        def shift_down(matrix, positions=1):
            return matrix[-positions:] + matrix[:-positions]


        positions=[1,2]
        shift_functions = [shift_left, shift_right, shift_up, shift_down]
        random_position=random.choice(positions)
        random_function=random.choice(shift_functions)

        neighbor=deepcopy(self.repr)
        neighbor=random_function(self.repr, positions=random_position)
        neighbor = LUKAPSASolution(repr=neighbor)
        return neighbor

