import numpy as np

def simplex(a, b, c):
    # Create an array `base` representing the initial base indices
    base = np.arange(len(a)) + len(c)

    # Check if the number of columns in `a` and `c` is equal to the number of variables
    if len(a) + len(c) != len(a[0]):
        # If not, create an identity matrix `B` and append it horizontally to `a`
        B = np.identity(len(a))
        a = np.hstack((a, B))

    # Create the initial tableau by stacking arrays vertically
    tableau = np.vstack((
        np.array([None] + [0] + list(c) + [0] * len(a)),  # First row of the tableau with objective coefficients
        np.hstack((np.transpose([base]), np.transpose([b]), a))  # Remaining rows with constraints and variables
    ))

    iterations = 0
    # Iterate until all coefficients in the objective row are non-negative
    while not np.all(tableau[0, 2:] >= 0):
        # Get the coefficients of the objective function
        objective_coefficients = tableau[0, 2:]
        # Find the index of the entering variable (the one with the most negative coefficient)
        entering_variable_index = np.argmin(objective_coefficients) + 2

        # Create a mask to identify rows where the entering variable is positive
        mask = tableau[:, entering_variable_index] > 0
        # Select rows where the entering variable is positive
        selected_rows = tableau[mask]
        # Find the departing variable by selecting the row with the minimum ratio of b/entering_variable
        minimum_index = np.argmin(selected_rows[:, 1] / selected_rows[:, entering_variable_index])
        departing_row = np.nonzero(mask)[0][minimum_index]

        # Perform pivot operation to make the departing variable equal to 1
        pivot = tableau[departing_row, entering_variable_index]
        tableau[departing_row, 1:] /= pivot

        non_departing_rows = np.arange(len(tableau)) != departing_row

        # Update the tableau using pivot operations to make other coefficients zero
        pivot_factor = tableau[non_departing_rows, entering_variable_index] / tableau[departing_row, entering_variable_index]
        pivot_factor = pivot_factor[:, np.newaxis]

        tableau[non_departing_rows, 1:] -= pivot_factor * tableau[departing_row, 1:]

        # Update the base index for the departing variable
        tableau[departing_row, 0] = entering_variable_index - 2

        iterations += 1

    # Extract the optimal solution and objective value from the tableau
    optimal_solution = np.zeros(len(tableau[0, 2:]))
    indices = np.arange(1, len(tableau))

    valid_indices = np.logical_and(tableau[indices, 0] < len(tableau[0, 2:]), tableau[indices, 0] != None)
    decision_var_indices = tableau[indices[valid_indices], 0].astype(int)
    optimal_solution[decision_var_indices] = tableau[indices[valid_indices], 1]

    optimal_value = -1 * tableau[0, 1]

    # Get the indices of variables in the base
    base_indices = np.where(optimal_solution != 0)[0] + 1

    # Print the results
    print(f'Optimal value: {optimal_value}')
    print(f'Optimal solution: {optimal_solution}')
    print('Base:', " ".join(map(str, base_indices)))
    print(f'Iterations: {iterations}\n')
