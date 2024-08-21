from modules.io import read_tsp_instance, get_filenames
from modules.TSP import TSP

ROOT_PATH = "./data/tsp_instances"
filenames = get_filenames(ROOT_PATH)

for filename in filenames:
    data = read_tsp_instance(filename)
    random_tsp = TSP(data)
    random_tsp.fit(data.shape[0]*10)
    print(f'{random_tsp.best_cost:.3f}')
