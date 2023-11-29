from evolution import run_algorithm, plotter

m = run_algorithm(population_size=6, max_runs=20, max_epochs=10, seed=233, verbose=True)
plotter(m)