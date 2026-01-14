import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for saving plots


def plot_route(x, y, tsp_path, dist, problem_type = 'tsp', before=None, dist_old=None, link=None, show_node_name=False, save_path='results/route.png'):
    if problem_type != 'tsp':
        print("only support tsp now")
        return
    # Extract the coordinates in the order of the TSP path
    path_coords = [(x[i], y[i]) for i in tsp_path]

    # Create the plot
    plt.figure(figsize=(10, 8))
    if dist_old and dist_old != dist:
        plt.title(f"TSP Solution Visualization, (Total Distance: {dist_old} -> {dist})")
    else:
        plt.title(f"TSP Solution Visualization, (Total Distance: {dist})")

    # Plot the points
    for point, (xi, yi) in enumerate(zip(x, y)):
        plt.plot(xi, yi, 'bo')  # 'bo' means blue color, circle marker
        if show_node_name:
            plt.text(xi, yi, f" {point}", fontsize=9)  # Adding labels

    # Plot the entire path
    x_coords, y_coords = zip(*path_coords)

    # Plot grey segment for 'before'
    if before is not None:
        for s, e in zip(before, before[1:]):
            plt.plot(
                [x[s], x[e]],
                [y[s], y[e]],
                'grey', linewidth=2
            )

    # Plot green segment for 'now'
    for s, e in zip(tsp_path, tsp_path[1:]):
        plt.plot(
            [x[s], x[e]],
            [y[s], y[e]],
            'g-', linewidth=2  # Green line for "after" segment
        )

    # Plot the old connection of the path in red
    if link is not None:
        for part in link:
            if len(part) == 1:
                continue
            for s, e in zip(part, part[1:]):
                plt.plot(
                    [x[s], x[e]],
                    [y[s], y[e]],
                    'r-', linewidth=2  # Red line for the rest of the path
                )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    
    # Save the plot
    if save_path:
        plt.savefig(save_path)
    plt.close()