import pandas as pd
import numpy as np
import datetime
import os


class TSP:
    def __init__(self, instance):
        self.distance_matrix = instance.to_numpy()
        self.n = self.distance_matrix.shape[0]
        self.best_solution = self.generate_solution()
        self.best_cost = self.objective_function(self.best_solution)

    def objective_function(self, solution, distance_matrix=None):
        """
        Evaluates the objective function for a given solution.

        Parameters
        ----------
        solution : numpy.ndarray
            The solution to evaluate.
        distance_matrix : numpy.ndarray
            The distance matrix of the TSP instance.

        Returns
        -------
        cost : float
            The cost of the solution.
        """
        
        if distance_matrix is None:
            distance_matrix = self.distance_matrix

        i = 0
        cost = 0
        for i in range(self.n - 1):
            v_actual = solution[i]
            v_next = solution[i + 1]
            cost += distance_matrix[v_actual][v_next]
        v_actual = solution[-1]
        v_next = solution[0]
        cost += distance_matrix[v_actual][v_next]

        return cost

    def generate_solution(self):
        """
        Generates a random solution for the TSP instance.

        Returns
        -------
        numpy.ndarray
            The generated solution.
        """

        return [*np.random.permutation(self.n)]

    def first_improvement(self):
        """
        Performs a first-improvement local search on the current best solution.

        Iterates through pairs of cities and swaps them in the tour. The search stops
        as soon as an improvement is found.

        Returns
        -------
        solution : list
            The new solution with an improved tour, or the original solution if no improvement is found.
        """

        for i in range(self.n):
            for j in range(i + 1, self.n):
                solution = self.best_solution[:]
                solution[i], solution[j] = solution[j], solution[i]
                cost = self.objective_function(solution)
                if cost < self.best_cost:
                    return solution        
        return self.best_solution

    def best_improvement(self):
        """
        Performs a best-improvement local search on the current best solution.

        Iterates through all possible pairs of cities, swapping them to find the best
        possible improvement in the tour.

        Returns
        -------
        best_solution : list
            The best solution found after evaluating all possible swaps.
        """

        best_solution = self.best_solution[:]
        best_cost = self.best_cost

        for i in range(self.n):
            for j in range(i + 1, self.n):
                solution = self.best_solution[:]
                solution[i], solution[j] = solution[j], solution[i]
                cost = self.objective_function(solution, self.distance_matrix)
                if cost < best_cost:
                    best_solution = solution
                    best_cost = cost
        return best_solution

    def fit(self, method="first"):
        """
        Fits the TSP instance using a local search algorithm.

        Parameters
        ----------
        method : str, optional
            The type of local search algorithm to use, either 'first' for first improvement
            or 'best' for best improvement. Default is 'first'.

        Returns
        -------
        None
        """

        while True:
            if method == "first":
                solution = self.first_improvement()
            elif method == "best":
                solution = self.best_improvement()
            cost = self.objective_function(solution)
            if cost == self.best_cost:
                break
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = solution


def read_tsp_instance(filepath):
    """
    Reads a TSP instance from a CSV file.

    Parameters
    ----------
    filepath : str
        The path to the CSV file containing the TSP instance.

    Returns
    -------
    data : pandas.DataFrame
        A DataFrame containing the TSP instance data. If the file does not
        exist, returns `None`.

    Notes
    -----
    This function uses the `pandas` library to read the CSV file and the `os`
    library to check if the file exists. It drops the "X" and "Y" columns
    from the DataFrame, which are assumed to be coordinates and are not used
    in the TSP instance.
    """

    if os.path.exists(filepath):
        data = pd.read_csv(filepath)
        data.drop(columns=["X", "Y"], inplace=True)
        return data
    else:
        print(f"File {filepath} not found.")
        return None


def get_filenames(ROOT_PATH):
    """
    Returns a list of all filenames inside the given root path.

    Parameters
    ----------
    ROOT_PATH : str
        The path to the root directory.

    Returns
    -------
    filenames : list
        A list of full paths to all files inside the root directory.

    Notes
    -----
    This function uses the `os` module to list the contents of the root directory
    and filter out directories. It returns a list of full paths to all files.
    """

    filenames = []
    for f in os.listdir(ROOT_PATH):
        full_path = os.path.join(ROOT_PATH, f)
        if os.path.isfile(full_path):
            filenames.append(full_path)

    return filenames


# <---------------> MAIN LOGIC <--------------->

ROOT_PATH = "./data/tsp_instances"
filenames = get_filenames(ROOT_PATH)

instances = []
costs = []
solutions = []
method = 'first'
for filename in filenames:
    data = read_tsp_instance(filename)

    local_tsp = TSP(data)
    local_tsp.fit(method=method)

    instance = filename.split(ROOT_PATH)[-1][1:]
    print(f'{instance}: ', end='')
    instances.append(instance)

    cost = local_tsp.best_cost
    costs.append(cost)

    solution = local_tsp.best_solution
    solutions.append(solution)
    print('OK')

df = pd.DataFrame({"instance": instances, "best_OF": costs, "best_SOL": solutions})

now_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filename = f"./data/labs/lab3_local_search/kalleb_abreu_TSP_local_{method}_{now_str}.csv"

df.to_csv(filename, index=False)
