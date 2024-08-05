import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os
import datetime as dt
import warnings

warnings.filterwarnings("ignore")
sns.set()


def add_days(date, days):
    weekday = date.weekday() + math.ceil(days)
    complete_weeks = weekday // 7
    added_days = weekday + complete_weeks * 2
    return date + dt.timedelta(days=added_days)


def read_arrs(pattern_files):
    files = glob.glob(pattern_files)
    term = [int(x.split("_")[2]) for x in files]
    sort = np.argsort(term)
    files = [files[s] for s in sort]

    arrs = []
    names = []
    terms = []
    date_start, date_end = None, None

    for file in files:
        arr = np.loadtxt(file, delimiter=",")
        arrs.append(arr)

        _, name = os.path.split(file)
        name = name[:-4]  # remove .csv
        vals = name.split("_")
        ticker = vals[1]
        term = int(vals[2])
        date_start = dt.datetime.strptime(vals[3], "%Y%m%d")
        date_end = add_days(date_start, term)
        name = ticker + " " + date_end.strftime("%Y-%m-%d")
        names.append(name)
        terms.append(term)

    return arrs, names, terms, date_start, date_end


def generate_plots(curr_val, pattern, save_place):
    arrs, names, _, _, _ = read_arrs(pattern)

    # Define subplots
    rows = int(math.ceil(len(arrs) / 3))
    cols = min(len(arrs), 3)
    fig, ax = plt.subplots(rows, cols, squeeze=False)
    fig.set_figheight(12)
    fig.set_figwidth(20)

    k = 0
    for i in range(rows):
        if k >= len(arrs):
            break

        for j in range(cols):
            if k >= len(arrs):
                break

            arr = arrs[k]
            name = names[k]

            sns.distplot(arr, ax=ax[i][j], color="red")
            ymin, ymax = ax[i][j].get_ylim()

            ax[i][j].vlines(x=curr_val, ymin=ymin, ymax=ymax, color="b")
            # plt.text(curr_val, 0, str(curr_val), rotation=90)

            med = np.median(arr)
            ax[i][j].vlines(x=med, ymin=ymin, ymax=ymax, color="r")
            # plt.text(med, 0, str(med), rotation=90)

            print(
                f"{name} {np.quantile(arr, 0.05):2.2f} {np.median(arr):2.2f} {np.quantile(arr, 0.95):2.2f}"
            )
            ax[i][j].set_title(name)
            ax[i][j].get_yaxis().set_visible(False)

            k += 1

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_place)


def main():
    # pattern = 'outputs/preds_SPXUSDMXN_*_202309*.csv'
    # save_name = 'outputs/reports/pred_SPXUSDMXN.png'
    # curr_val1 = 4565.72 * 16.73
    # curr_val1 = 4460.77 * 17.60 # 2023-09-04

    pattern = "outputs/preds_USDMXN_*_20240108.csv"
    save_name = "outputs/reports/pred_USDMXN_20240108.png"
    curr_val1 = 16.85

    # pattern = 'outputs/preds_SPX_*_20230909.csv'
    # save_name = 'outputs/reports/pred_SPX_20230909.png'
    # curr_val1 = 4450

    generate_plots(curr_val1, pattern, save_name)


if __name__ == "__main__":
    main()
