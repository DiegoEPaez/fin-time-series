from plot_output import read_arrs
import numpy as np
import datetime as datetime


def returns(current, pattern):
    arrs1, names1, terms1, _, _ = read_arrs(pattern)
    print("Name, return, annualized_return, cetes_annualized_return")
    for i in range(len(arrs1)):
        arr1, name1, term1 = arrs1[i], names1[i], terms1[i]

        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            q1 = np.quantile(arr1, q)
            r = q1 / current - 1
            print(name1, f"{q * 100}", f"{r * 100:4.2f}%", f"{((r + 1) ** (252 / term1) - 1) * 100:4.2f}%", f"{term1 * 11 / 252:4.2f} %")

    print("A")


def main():
    pattern = 'outputs/preds_SPXUSDMXN_*_202309*.csv'
    curr = 4465.77 * 17.60

    #pattern = 'outputs/preds_USDMXN_*_202309*.csv'
    #curr = 17.60
    returns(curr, pattern)


if __name__ == '__main__':
    main()