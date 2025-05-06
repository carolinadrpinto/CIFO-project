import pandas as pd
from copy import deepcopy
from library.solution import Solution


artists = pd.read_csv('data/artists(in).csv', index_col=0)
conflicts = pd.read_csv('data/conflicts(in).csv')


def get_dic(df, row):
    dic={}
    for i, value in enumerate(df[row].unique().tolist()):
        temporary_dic={value: i}
        dic.update(temporary_dic)
    return dic

dic_artists = get_dic(artists, 'name')
dic_genre = get_dic(artists, 'genre')

artists_encoded=deepcopy(artists)
artists_encoded['name'] = artists_encoded['name'].map(dic_artists)
artists_encoded['genre'] = artists_encoded['genre'].map(dic_genre)
artists_encoded
artists_encoded.values.to_list()


conflicts=conflicts.values.tolist()
conflicts


class MFLUSolution(Solution):
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

