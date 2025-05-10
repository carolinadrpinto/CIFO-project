import pandas as pd
from copy import deepcopy

artists = pd.read_csv('data/artists(in).csv', index_col=0)
conflicts = pd.read_csv('data/conflicts(in).csv', index_col=0)

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

artists_list = [tuple(a) for a in artists_encoded.values.tolist()]
artists_list  # list of tuples

conflicts_matrix =conflicts.values.tolist() # conflicts works as a distance matrix

