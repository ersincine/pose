import numpy as np
import matplotlib.pyplot as plt



for dataset in ["Barn", "Truck", "Meetingroom"]:
    methods = ["snn", "nn"]

    # color_dict = {method: np.random.rand(3,) for method in methods}
    color_dict = {"snn": "r", "nn": "b"}

    q_errs_dict = {}
    t_errs_dict = {}
    for method in methods:
        q_errs = np.loadtxt(f"{dataset}_{method}_q_errs.txt")
        t_errs = np.loadtxt(f"{dataset}_{method}_t_errs.txt")
        q_errs_dict[method] = q_errs
        t_errs_dict[method] = t_errs

    q_errs_cumulative = {method: np.zeros(181) for method in methods}
    t_errs_cumulative = {method: np.zeros(181) for method in methods}

    for method in methods:
        for q_err in q_errs_dict[method]:
            q_err = int(q_err)
            q_errs_cumulative[method][q_err] += 1

        for t_err in t_errs_dict[method]:
            t_err = int(t_err)
            t_errs_cumulative[method][t_err] += 1

    for method in methods:
        q_errs_cumulative[method] = np.cumsum(q_errs_cumulative[method]) / len(q_errs_dict[method])
        t_errs_cumulative[method] = np.cumsum(t_errs_cumulative[method]) / len(t_errs_dict[method])
        
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for method in methods:
        axs[0].plot(q_errs_cumulative[method], label=method, color=color_dict[method])
        axs[1].plot(t_errs_cumulative[method], label=method, color=color_dict[method])

    axs[0].set_title(f"{dataset} q_errs")
    axs[0].set_xlabel("Degrees")
    axs[0].set_ylabel("Percentage")
    axs[0].set_xlim(0, 180)
    axs[0].set_xticks(np.arange(0, 181, 15))
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title(f"{dataset} t_errs")
    axs[1].set_xlabel("Degrees")
    axs[1].set_ylabel("Percentage")
    axs[1].set_xlim(0, 90)
    axs[1].set_xticks(np.arange(0, 91, 15))
    axs[1].legend()
    axs[1].grid()

    fig.suptitle("Cumulative histogram of errors for relative camera orientation (left) and relative translation (right)")

    plt.savefig(f"{dataset}_error.png")
    plt.show()
    plt.close()
