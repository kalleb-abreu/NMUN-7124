import pandas as pd
import numpy as np
from modules import io
from modules import TSP

ROOT_PATH = "./data/labs/lab4_grasp"

filenames = io.get_filenames(ROOT_PATH)

data_filenames = io.get_filenames("./data/tsp_instances")

df = pd.DataFrame()

for filename in filenames:
    df = pd.concat([df, pd.read_csv(filename)])

best_df = (
    df[df["best_OF"].isin(df.groupby("instance").min()["best_OF"])]
    .drop_duplicates(subset=['instance'])
    .sort_values(by=["instance"])
)

for i, row in best_df.iterrows():
    data = io.read_tsp_instance(f"./data/tsp_instances/{row['instance']}")

    tsp = TSP.TSP(data)

    solution = row['best_SOL'].replace('[', '').replace(']', '').split(', ')
    solution = [int(sol) for sol in solution]

    score = tsp.objective_function(solution, tsp.distance_matrix)
    print(f'{(float(row["best_OF"] - score)):.6f}')

best_df.to_csv(f'{ROOT_PATH}/bks.csv', index=False)

