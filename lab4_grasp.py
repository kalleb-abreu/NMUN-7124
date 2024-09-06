import pandas as pd
import numpy as np
import datetime
import time
import os


class TSP:

    def __init__(self, instance, alpha=0.5, iter=None, local_search="first"):
        self.alpha = alpha
        self.local_search = (
            self.first_improvement if local_search == "first" else self.best_improvement
        )

        self.distance_matrix = instance.to_numpy()
        self.n = self.distance_matrix.shape[0]
        for i in range(self.n):
            self.distance_matrix[i][i] = np.inf
        self.RCL = self.generate_RCL()
        self.iter = 10 * self.n if iter == None else iter

        self.best_solution = []
        self.best_cost = np.inf

    def generate_RCL(self):
        """
        Generate the Restricted Candidate List (RCL).

        Parameters
        ----------
        None

        Returns
        -------
        rcl : list of bool
            A list of boolean values indicating whether the maximum distance
            of each candidate is greater than or equal to the threshold.

        Notes
        -----
        The threshold is calculated using the minimum and maximum distances
        in the distance matrix, with a scaling factor of `self.alpha`.
        """

        mask = self.distance_matrix != np.inf
        min_dist = np.min(self.distance_matrix[mask])
        max_dist = np.max(self.distance_matrix[mask])
        threshold = min_dist + self.alpha * (max_dist - min_dist)

        return [
            np.max(self.distance_matrix[mask][i]) >= threshold for i in range(self.n)
        ]

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

    def remove_node(self, nodes, distance_matrix, solution):
        """
        Remove a node from the solution and update the distance matrix and nodes list.

        Parameters
        ----------
        nodes : array_like
            Array of node indices.
        distance_matrix : array_like
            Distance matrix.
        solution : array_like
            Solution path.

        Returns
        -------
        nodes : array_like
            Updated array of node indices.
        distance_matrix : array_like
            Updated distance matrix.

        Notes
        -----
        The last two nodes in the solution are identified (n1 and n2).
        The node n2 is removed by setting its column and row in the distance matrix to infinity.
        The node n1 is then removed from the list of nodes.
        """
        n1 = solution[-1]
        nodes = np.delete(nodes, np.where(nodes == n1))

        n2 = solution[-2]
        distance_matrix[n2, :] = np.inf
        distance_matrix[:, n2] = np.inf

        return nodes, distance_matrix

    def generate_solution(self):
        """
        Generates a solution for the TSP instance.

        Returns
        -------
        numpy.ndarray
            The generated solution.
        """

        distance_matrix = self.distance_matrix.copy()
        nodes = np.array(np.arange(self.n))

        node = np.random.randint(self.n)
        nodes = np.delete(nodes, np.where(nodes == node))
        solution = [node]

        for _ in range(self.n - 1):
            if self.RCL[node]:
                node = nodes[np.random.randint(len(nodes))]
            else:
                node = np.argmin(distance_matrix[node])
            solution.append(node)
            nodes, distance_matrix = self.remove_node(nodes, distance_matrix, solution)

        return solution

    def first_improvement(self, base_solution):
        """
        Finds the first improving solution by applying 2-opt local search to the base_solution.

        Parameters
        ----------
        base_solution : array_like
            The base solution to start the local search from.

        Returns
        -------
        best_solution : array_like
            The first improving solution found by the local search, which is either the original
            base_solution or an improved solution.

        Notes
        -----
        The local search is applied by iterating over all possible pairs of cities
        and applying a 2-opt swap to the base_solution. If an improving solution is found,
        the function returns the improving solution; otherwise, the function returns
        the original base_solution.
        """

        for i in range(self.n):
            for j in range(i + 1, self.n):
                solution = base_solution[:]
                solution[i], solution[j] = solution[j], solution[i]
                cost = self.objective_function(solution)
                if cost < self.best_cost:
                    return solution

        return base_solution

    def best_improvement(self, base_solution):
        """
        Finds the best solution by applying 2-opt local search to the base_solution.

        Parameters
        ----------
        base_solution : array_like
            The base solution to start the local search from.

        Returns
        -------
        best_solution : array_like
            The best solution found by the local search.

        Notes
        -----
        The local search is applied by iterating over all possible pairs of cities
        and applying a 2-opt swap to the base_solution if the new solution is better.
        """

        best_solution = base_solution[:]
        best_cost = self.objective_function(base_solution, self.distance_matrix)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                solution = base_solution
                solution[i], solution[j] = solution[j], solution[i]
                cost = self.objective_function(solution, self.distance_matrix)
                if cost < best_cost:
                    best_solution = solution
                    best_cost = cost

        return best_solution

    def fit(self):
        """
        Fits the model using GRASP.

        Parameters
        ------------
        None

        Returns
        -------
        None
        """

        start_time = time.time()
        unimproved_iter = 0
        self.total_iter = 0
        while True:
            solution = self.generate_solution()
            solution = self.local_search(solution)
            cost = self.objective_function(solution, self.distance_matrix)

            if cost < self.best_cost:
                unimproved_iter = 0
                self.best_cost = cost
                self.best_solution = solution

            unimproved_iter += 1
            self.total_iter += 1
            if unimproved_iter == self.iter:
                break

        end_time = time.time()
        self.total_time = end_time - start_time


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
for filename in filenames:
    data = read_tsp_instance(filename)

    instance = filename.split(ROOT_PATH)[-1][1:]
    instances.append(instance)
    print(f"{instance}")

    tsp = TSP(data, alpha=0.75, iter=1)
    tsp.fit()

    cost = tsp.best_cost
    costs.append(cost)

    it = tsp.total_iter
    exec_time = tsp.total_time
    print(f"   cost: {cost:7.4f}, iter: {it:4}, execution_time: {exec_time:12.2f}s\n")

    solution = tsp.best_solution
    solutions.append(solution)

df = pd.DataFrame({"instance": instances, "best_OF": costs, "best_SOL": solutions})

now_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
filename = f"./data/labs/lab4_grasp/kalleb_abreu_TSP_grasp_{now_str}.csv"

df.to_csv(filename, index=False)