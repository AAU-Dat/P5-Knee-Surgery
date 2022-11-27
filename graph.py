""" Creates a scatter plot graph with the expected data and predicted data. """
    fig, ax = plt.subplots()
    ax.scatter(actual_values, predicted_values, s=5, c='b', alpha=0.25, label='Data Points')

    min_lim, max_lim = min(actual_values), max(actual_values)
    five_percent = (max_lim - min_lim) * .05
    min_lim, max_lim = min_lim - five_percent, max_lim + five_percent

    plt.ylim(min_lim, max_lim)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, linewidth=2, c='r', label='Best Fit Line')

    plt.title(f"Predicted and Actual {target} Values")
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    plt.savefig(path, dpi=300)
    plt.close()
