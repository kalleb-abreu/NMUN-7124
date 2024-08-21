import numpy as np


class TSP:
    def __init__(self, instance):
        self.distance_matrix = instance.to_numpy()
        self.n = self.distance_matrix.shape[0]

    def objective_function(self, solution, distance_matrix):
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

    def fit(self, steps=None):
        """
        Fits the TSP instance using a simple random search algorithm.

        Parameters
        ----------
        steps : int, optional
            The number of steps to perform in the random search algorithm. Default is 2*n.

        Returns
        -------
        None
        """

        if steps is None:
            steps = 2 * self.n

        self.best_solution = []
        self.best_cost = np.infty
        for i in range(steps):
            solution = self.generate_solution()
            cost = self.objective_function(solution, self.distance_matrix)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = solution
