import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def attempts_analysis():
    X_df = pd.read_csv("backend/data/aggregated_stats.csv")
    y_df = pd.read_csv("backend/data/weekly_stats.csv")
    X_df = X_df[X_df["Total_Ply"] != 0]

    X_df = X_df.set_index(["Home", "Away", "Week", "Year"])
    y_df = y_df.set_index(["Home", "Away", "Week", "Year"])

    df = X_df.join(y_df[["H_Att", "A_Att"]])
    df = df.reset_index()
    atts = []
    for index, row in df.iterrows():
        if row["Team"] == row["Home"]:
            atts.append(row["H_Att"])
        else:
            atts.append(row["A_Att"])
    df["y_Att"] = atts
    for col in df.columns:
        plt.scatter(x=df["Att"], y=df["y_Att"], alpha=.05)
        plt.xlabel(col)
        plt.ylabel('Att')
        plt.title(col + " vs Atts")
        plt.show()


def main():
    attempts_analysis()


if __name__ == '__main__':
    main()
