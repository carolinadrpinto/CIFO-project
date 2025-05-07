import pandas as pd
from copy import deepcopy
import random
from library.solution import Solution


artists = pd.read_csv('data/artists(in).csv', index_col=0)
conflicts = pd.read_csv('data/conflicts(in).csv')

artists
conflicts


def get_dic(df, row):
    dic={}
    for i, value in enumerate(df[row].unique().tolist()):
        temporary_dic={value: i}
        dic.update(temporary_dic)
    return dic

dic_artists = get_dic(artists, 'name')
dic_genre = get_dic(artists, 'genre')

dic_genre
artists


artists_encoded=deepcopy(artists)
artists_encoded['name'] = artists_encoded['name'].map(dic_artists)
artists_encoded['genre'] = artists_encoded['genre'].map(dic_genre)
artists_encoded
artists_encoded.values.tolist()


conflicts=conflicts.values.tolist()
conflicts



class LUSolution(Solution):
    def __init__(
        self,
        artists=artists,
        conflicts: list[list[float]] = conflicts,
        time_slots: int = 7,
        stages: int = 5,
        repr: str = None,
    ):
        self.artists=artists
        self.conflicts=conflicts
        self.time_slots=time_slots
        self.stages=stages

        if repr:
            repr = self._validate_repr(repr) #criar função

        super().__init__(repr=repr)


    def _validate_repr(self, repr):
        # If repr is given as string, convert to list
        if isinstance(repr, str):
            repr = [int(bit) for bit in repr]
        if not isinstance(repr, list):
            raise TypeError("Representation must be string or list")
        # All list elements should be integers
        if not all([isinstance(bit, int) for bit in repr]):
            repr = [int(bit) for bit in repr]
        # Validate representation length and content
        if (len(repr) != len(self.values)) or (not set(repr).issubset({0, 1})):
            raise ValueError("Representation must be a binary string/list with as many values as objects")
        return repr

    def random_initial_representation(self):
        repr = []
        for _ in range(len(self.values)):
            repr.append(random.choice([0, 1]))
        return repr

    def popularity_score(self):
        pass
    def diversity_score(self):
        pass
    def conflict_score(self):    
        pass
    
    def fitness(self):
        fitness=self.popularity_score()+self.diversity_score()+(1-self.conflict_score())       
        return fitness



class LUGASolution(LUSolution):
    def __init__(
        self,
        crossover_function,
        mutation_function,
        artists=artists,
        conflicts: list[list[float]] = conflicts,
        time_slots: int = 7,
        stages: int = 5,
        repr: str = None,
    ):

        super().__init__(
            artists=artists,
            conflicts=conflicts,
            time_slots=time_slots,
            stages=stages,
            repr=repr,
            )
        
        self.mutation_function=mutation_function
        self.crossover_function=crossover_function,


