import numpy as np
from plot_output import read_arrs
from datetime import datetime
from datetime import timedelta


def convolve_series(pattern1, pattern2):
    arrs1, names1, terms1, date_start, date_end = read_arrs(pattern1)
    arrs2, names2, terms2, date_start, date_end = read_arrs(pattern2)

    for i in range(min(len(arrs1), len(arrs2))):
        arr1, name1, term1 = arrs1[i], names1[i], terms1[i]
        arr2, name2, term2 = arrs2[i], names2[i], terms2[i]

        if term1 != term2:
            print(f"Warning different terms: {term1}, {term2}, exiting")

        col1, est_date = name1.split(" ")
        col2, est_date = name2.split(" ")

        s1 = np.random.choice(arr1, size=int(len(arr1) ** 0.7))
        s2 = np.random.choice(arr2, size=int(len(arr2) ** 0.7))

        conv = np.outer(s1, s2).reshape(-1)
        res = np.random.choice(conv, size=min(max(len(arr1), len(arr2)), len(conv)))

        np.savetxt("outputs/preds_" + col1 + col2 + "_" + str(term1) + "_" + datetime.strftime(date_start, "%Y%m%d") + ".csv", res,
                   delimiter=",")

    print("A")


def main():
    pattern1 = 'outputs/preds_SPX_*_202309*.csv'
    pattern2 = 'outputs/preds_USDMXN_*_202309*.csv'

    convolve_series(pattern1, pattern2)


if __name__ == '__main__':
    main()