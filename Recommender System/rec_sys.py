import pandas as pd
import sys

from sqlalchemy import create_engine

import surprise

# This script only serves as an indication of the code used for the recommender system

USER = ''
PASSWORD = ''
HOST = ''
DATABASE = ''
PORT = ''
DB_CONNECTION_STR = ''
DB_CONNECTION = create_engine(DB_CONNECTION_STR)
SQL_QUERY = 'SELECT * FROM wiski_number_attempts'
DATAFRAME = df = pd.read_sql(SQL_QUERY, con=DB_CONNECTION)


num_arguments = len(sys.argv)
question_nids = [int(sys.argv[2])]
if len(sys.argv) >= 4:
    question_nids.append(int(sys.argv[3]))
if len(sys.argv) >= 5:
    question_nids.append(int(sys.argv[4]))
if len(sys.argv) >= 6:
    question_nids.append(int(sys.argv[5]))
if len(sys.argv) >= 7:
    question_nids.append(int(sys.argv[6]))
if len(sys.argv) >= 8:
    question_nids.append(int(sys.argv[7]))


def execute_collaborative_filtering():
    data = load_data(DATAFRAME)
    train_set = data.build_full_trainset()
    algo = surprise.KNNBaseline()
    algo.fit(train_set)
    for i in range(len(question_nids)):
        prediction, k_neighbours = algo.predict(get_user_uid(), get_question_nid(i), 0, verbose=True)

        print('# 0 ' + str(i) + ' ' + str(prediction))
        print('# 1 ' + str(i) + ' ' + str(k_neighbours))

        estimated_num_tries = prediction[3]
        neighbours_uids = ""
        neighbours_attempts = ""
        for entry in k_neighbours:
            to_add = str(entry[0])
            to_add_2 = str(entry[2])
            to_add += " "
            to_add_2 += " "
            neighbours_uids += to_add
            neighbours_attempts += to_add_2

        print('# 2 ' + str(i) + ' ' + str(estimated_num_tries))
        print('# 3 ' + str(i) + ' ' + str(neighbours_uids))
        print('# 4 ' + str(i) + ' ' + str(neighbours_attempts))
        print('# 5 ' + str(i) + ' ' + str(prediction[4]['was_impossible']))
        print('# 6 ' + str(i) + ' ' + str(len(k_neighbours)))


def get_user_uid():
    return int(sys.argv[1])


def get_question_nid(index):
    return question_nids[index]

def load_data(dataframe):
    reader = surprise.Reader()
    data = surprise.Dataset.load_from_df(dataframe, reader)
    return data


execute_collaborative_filtering()