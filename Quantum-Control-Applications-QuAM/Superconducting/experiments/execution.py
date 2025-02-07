from qualang_tools.results import fetching_tool, progress_counter
from qm import QmJob


def print_progress_bar(job: QmJob, iteration_variable: str, total_number_of_iterations: int) -> None:
    results = fetching_tool(job, [iteration_variable], mode="live")
    while results.is_processing():
        # Fetch results
        n = results.fetch_all()[0]
        # Progress bar
        progress_counter(n, total_number_of_iterations, start_time=results.start_time)
