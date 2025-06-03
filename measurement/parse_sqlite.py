import argparse
import sqlite3


def get_cuda_events_for_kernel(db_path, kernel_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
    SELECT StringIds.value, (kernel_event.end - kernel_event.start) / 1000.0 / 1000.0 AS duration_ms
    FROM StringIds, CUPTI_ACTIVITY_KIND_KERNEL AS kernel_event
    WHERE StringIds.id = kernel_event.shortName
    AND value = '{kernel_name}';
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    return [{"name": row[0], "duration_ms": row[1]} for row in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_file", type=str, required=True, help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--target_kernels",
        type=str,
        nargs="+",
        required=True,
        help="List of kernel names to analyze.",
    )
    parser.add_argument(
        "--end_to_end_runtime_ms",
        type=float,
        required=True,
        help="Total runtime in milliseconds.",
    )
    parser.add_argument(
        "--target_runtime_ms",
        type=float,
        required=True,
        help="Total runtime of measurement target in milliseconds.",
    )
    parser.add_argument(
        "--measurement_iteration_count",
        type=int,
        required=True,
        help="Number of measurement iterations.",
    )
    args = parser.parse_args()

    kernel_duration_averages = {}
    kernel_duration_percentages = {}
    cumulative_kernel_duration_percentages = {}
    for target_kernel in args.target_kernels:
        events_for_kernel = get_cuda_events_for_kernel(args.db_file, target_kernel)
        event_count = len(events_for_kernel)
        if event_count > 0:
            kernel_duration_averages[target_kernel] = 0

        quotient = event_count // args.measurement_iteration_count
        actual_iteration_count = args.measurement_iteration_count * quotient

        for event in events_for_kernel[-actual_iteration_count:]:
            kernel_duration_averages[target_kernel] += event["duration_ms"]

        if target_kernel in kernel_duration_averages:
            kernel_duration_percentages[target_kernel] = (
                kernel_duration_averages[target_kernel]
                / args.measurement_iteration_count
                / args.target_runtime_ms
                * 100
            )

    cumulative_kernel_duration_percentages = (
        f"{sum([x for x in kernel_duration_percentages.values()])} %"
    )
    kernel_duration_percentages = {
        k: f"{v} %" for k, v in kernel_duration_percentages.items()
    }

    print(args.db_file)
    print(f"{args.end_to_end_runtime_ms} ms")
    print(f"{args.target_runtime_ms} ms")
    print(f"{args.target_runtime_ms / args.end_to_end_runtime_ms * 100} %")
    print(kernel_duration_percentages)
    print(cumulative_kernel_duration_percentages)
