import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from backend.scraping.Game_Stats import convert_poss
from backend.scraping.odds import scrape_vegas


def convert_odds(odds):
    if odds < 0:
        odds = odds * -1
        return odds / (100 + odds)
    else:
        return 100 / (100 + odds)


def calc_profit(stake, odds):
    if odds < 0:
        odds = odds * -1
        return stake / (odds / 100)
    else:
        return stake * (odds / 100)


def load_data_classifier():
    """
    Loads data and splits into X and Y training and testing sets
    :return:
    """
    x_df = pd.read_csv("backend/data/current_last_five_games.csv")
    x_df.fillna(0, inplace=True)
    y_df = pd.read_csv("backend/data/current_season_stats.csv")
    odds = pd.read_csv("backend/data/odds/odds.csv")

    df = pd.merge(y_df, x_df,
                  how="left",
                  left_on=["Home", "Week", "Year"],
                  right_on=["team", "week", "year"])

    df = pd.merge(df, x_df,
                  how="left",
                  left_on=["Away", "Week", "Year"],
                  right_on=["team", "week", "year"])

    df = df[(df["Week"] > 5) | (df["Year"] != 2010)]
    df.set_index(["Home", "Away", "Week", "Year"], inplace=True)

    x_cols = []
    for col in df.columns:
        if "poss" in col:
            df[col] = df[col].apply(lambda x: convert_poss(x) if ":" in x else x)
            x_cols.append(col)
        elif col[-4:] in ["_1_x", "_1_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 3
        elif col[-4:] in ["_2_x", "_2_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 3
        elif col[-4:] in ["_3_x", "_3_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2

        elif col[-4:] in ["_4_x", "_4_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2
        elif col[-4:] in ["_5_x", "_5_y"] \
                and "named" not in col and "opponent" not in col and "season_length" not in col:
            x_cols.append(col)
            df[col] = df[col] * 2

    # Standardized X values
    # x_cols = ["3rd_att", "3rd_cmp", "3rd_att_def", "3rd_cmp_def", "4th_att", "4th_cmp", "4th_att_def", "4th_cmp_def",
    #           "cmp", "cmp_def", "poss", "total_y", "total_y_def"]
    x_cols = ["cmp_pct", "cmp_pct_def", "3rd_pct", "3rd_pct_def", "4th_pct", "4th_pct_def", "fg_pct", "yds_per_rush",
              "yds_per_rush_def", "yds_per_att", "yds_per_att_def", "yds_per_ply", "yds_per_ply_def", "poss",
              "pass_yds", "pass_yds_def", "pen_yds", "pen_yds_def", "punts", "punts_def", "rush_yds", "rush_yds_def",
              "sacks", "sacks_def", "score", "score_def", "cmp", "cmp_def", "total_y", "total_y_def", "pts_per_ply",
              "pts_per_ply_def", "punts_per_ply", "punts_per_ply_def", "pts_per_yd", "pts_per_yd_def"]

    final_cols = []
    for col in x_cols:
        for i in range(1, 6):
            final_cols.append(col + "_" + str(i) + "_x")
            final_cols.append(col + "_" + str(i) + "_y")
    X = df[final_cols]
    X.astype(float)
    scaler = StandardScaler()
    X_standardized = pd.DataFrame(scaler.fit_transform(X))
    X_standardized.index = X.index

    # Y values
    df = pd.merge(df, odds.iloc[:, 1:],
                  how="left",
                  left_on=["Home", "Away", "Week", "Year"],
                  right_on=["Home", "Away", "Week", "Year"])
    df.set_index(["Home", "Away", "Week", "Year"], inplace=True)
    df["win_lose"] = df["H_Score"] - df["A_Score"]
    df["win_lose"] = df["win_lose"] > 0
    df["win_lose"] = df["win_lose"].astype(int)
    y = df[["win_lose", "ML_h", "ML_a"]]
    y.index = df.index

    return X_standardized, y


def current_week(cw, favorites, underdogs):
    # Load data and model
    odds = pd.read_csv("backend/data/odds/2021/Week_11.csv")
    X, y = load_data_classifier()
    svm = pickle.load(open("backend/modeling/models/svm.pkl", "rb"))
    X = X.reset_index()
    y = y.reset_index()
    X = X[(X["Year"] == 2021) & (X["Week"] == cw)]

    # Predict
    svm_prob = svm.predict_proba(X.drop(["Home", "Away", "Week", "Year"], axis=1))
    svm_odds = pd.DataFrame(svm_prob, columns=["away_win_prob", "home_win_prob"])
    odds = pd.merge(odds, svm_odds, how="left", left_index=True, right_index=True)
    odds["Home_odds_actual"] = odds["ML_h"].apply(lambda x: convert_odds(x))
    odds["Away_odds_actual"] = odds["ML_a"].apply(lambda x: convert_odds(x))
    odds["home_divergence"] = odds["home_win_prob"] - odds["Home_odds_actual"]
    odds["away_divergence"] = odds["away_win_prob"] - odds["Away_odds_actual"]

    odds["favorite"] = np.where(odds["ML_h"] < 0, 1, 0)
    odds["underdog"] = np.where(odds["ML_h"] > 0, 1, 0)
    odds["fav_win_prob"] = np.where(odds["ML_h"] < 0,
                                    round(odds["home_win_prob"], 1),
                                    round(odds["away_win_prob"], 1))
    odds["under_win_prob"] = np.where(odds["ML_h"] > 0,
                                      round(odds["home_win_prob"], 1),
                                      round(odds["away_win_prob"], 1))
    odds["fav_div"] = np.where(odds["ML_h"] < 0,
                               round(odds["home_divergence"], 1),
                               round(odds["away_divergence"], 1))
    odds["under_div"] = np.where(odds["ML_h"] > 0,
                                 round(odds["home_divergence"], 1),
                                 round(odds["away_divergence"], 1))
    predicted_outcome = []
    for index, row in odds.iterrows():
        if (row["under_win_prob"], row["under_div"]) in underdogs:
            predicted_outcome.append(row["underdog"])
        elif (row["fav_win_prob"], row["fav_div"]) in favorites:
            predicted_outcome.append(row["favorite"])
        else:
            predicted_outcome.append(None)
    odds["predicted_outcome"] = predicted_outcome
    odds = odds.dropna(axis=0)
    odds["potential_payout"] = np.where(odds["predicted_outcome"],
                                        odds["ML_h"].apply(lambda x: calc_profit(100, x)),
                                        odds["ML_a"].apply(lambda x: calc_profit(100, x)))

    # Format for excel file
    odds["bet"] = np.where(odds["predicted_outcome"], odds["Home"], odds["Away"])
    odds["home_win_prob"] = odds["home_win_prob"].apply(lambda x: str(round(x * 100)) + "%")
    odds["away_win_prob"] = odds["away_win_prob"].apply(lambda x: str(round(x * 100)) + "%")
    odds["potential_payout"] = odds["potential_payout"].apply(lambda x: round(x))
    odds["potential_units"] = odds["potential_payout"].apply(lambda x: x / 100)
    odds["Home_odds_actual"] = odds["Home_odds_actual"].apply(lambda x: str(round(x * 100)) + "%")
    odds["Away_odds_actual"] = odds["Away_odds_actual"].apply(lambda x: str(round(x * 100)) + "%")
    odds = odds[["Home", "home_win_prob", "Home_odds_actual", "ML_h",
                 "Away", "away_win_prob", "Away_odds_actual", "ML_a",
                 "bet", "potential_units"]]
    odds = odds.rename(columns={"home_win_prob": "home_predicted_prob", "away_win_prob": "away_predicted_prob",
                                "Home_odds_actual": "home_vegas_prob", "Away_odds_actual": "away_vegas_prob"})
    odds.to_csv("backend/data/predictions/Week_" + str(cw) + "_predictions.csv")


def current_season_odds(cw):
    odds = pd.DataFrame()
    for week in range(1, cw):
        df = pd.read_excel("backend/data/odds/2021/Week_" + str(week) + ".xlsx")
        odds = odds.append(df.drop(["Unnamed: 0"], axis=1))

    teams_dict = {"Patriots": "new-england-patriots", "Colts": "indianapolis-colts", "Falcons": "atlanta-falcons",
                  "Bills": "buffalo-bills", "Ravens": "baltimore-ravens", "Bears": "chicago-bears",
                  "Lions": "detroit-lions", "Browns": "cleveland-browns", "Texans": "houston-texans",
                  "Titans": "tennessee-titans", "Packers": "green-bay-packers", "Vikings": "minnesota-vikings",
                  "Dolphins": "miami-dolphins", "Jets": "new-york-jets", "Saints": "new-orleans-saints",
                  "Eagles": "philadelphia-eagles", "Washington": "washington-football-team",
                  "Panthers": "carolina-panthers", "49ers": "san-francisco-49ers", "Jaguars": "jacksonville-jaguars",
                  "Bengals": "cincinnati-bengals", "Raiders": "las-vegas-raiders", "Cowboys": "dallas-cowboys",
                  "Chiefs": "kansas-city-chiefs", "Cardinals": "arizona-cardinals", "Seahawks": "seattle-seahawks",
                  "Steelers": "pittsburgh-steelers", "Chargers": "los-angeles-chargers", "Giants": "new-york-giants",
                  "Buccaneers": "tampa-bay-buccaneers", "Rams": "los-angeles-rams", "Broncos": "denver-broncos"}

    odds["Home"] = odds["Home"].apply(lambda x: teams_dict[x])
    odds["Away"] = odds["Away"].apply(lambda x: teams_dict[x])

    odds.to_csv("backend/data/odds/current_season_odds.csv")


def current_season(cw, favorites, underdogs):
    odds = pd.read_csv("backend/data/odds/current_season_odds.csv")

    X, y = load_data_classifier()
    svm = pickle.load(open("backend/modeling/models/svm.pkl", "rb"))
    y = y.reset_index()
    X = X.reset_index()
    X = X[(X["Year"] == 2021) & (X["Week"] < cw)]
    y = y[(y["Year"] == 2021) & (y["Week"] < cw)]
    y = pd.merge(y.drop(["ML_a", "ML_h"], axis=1), odds.drop(["Unnamed: 0"], axis=1),
                 left_on=["Home", "Away", "Week", "Year"],
                 right_on=["Home", "Away", "Week", "Year"])

    # Predict
    svm_prob = svm.predict_proba(X.drop(["Home", "Away", "Week", "Year"], axis=1))
    svm_odds = pd.DataFrame(svm_prob, columns=["away_win_prob", "home_win_prob"])
    odds = pd.merge(y, svm_odds, how="left", left_index=True, right_index=True)
    odds["Home_odds_actual"] = odds["ML_h"].apply(lambda x: convert_odds(x))
    odds["Away_odds_actual"] = odds["ML_a"].apply(lambda x: convert_odds(x))
    odds["home_divergence"] = odds["home_win_prob"] - odds["Home_odds_actual"]
    odds["away_divergence"] = odds["away_win_prob"] - odds["Away_odds_actual"]
    odds["favorite"] = np.where(odds["ML_h"] < 0, 1, 0)
    odds["underdog"] = np.where(odds["ML_h"] > 0, 1, 0)
    odds["fav_win_prob"] = np.where(odds["ML_h"] < 0,
                                    round(odds["home_win_prob"], 1),
                                    round(odds["away_win_prob"], 1))
    odds["under_win_prob"] = np.where(odds["ML_h"] > 0,
                                      round(odds["home_win_prob"], 1),
                                      round(odds["away_win_prob"], 1))
    odds["fav_div"] = np.where(odds["ML_h"] < 0,
                               round(odds["home_divergence"], 1),
                               round(odds["away_divergence"], 1))
    odds["under_div"] = np.where(odds["ML_h"] > 0,
                                 round(odds["home_divergence"], 1),
                                 round(odds["away_divergence"], 1))
    predicted_outcome = []
    for index, row in odds.iterrows():
        if (row["under_win_prob"], row["under_div"]) in underdogs:
            predicted_outcome.append(row["underdog"])
        elif (row["fav_win_prob"], row["fav_div"]) in favorites:
            predicted_outcome.append(row["favorite"])
        else:
            predicted_outcome.append(None)
    odds["predicted_outcome"] = predicted_outcome
    odds = odds.dropna(axis=0)
    odds["potential_payout"] = np.where(odds["predicted_outcome"],
                                        odds["ML_h"].apply(lambda x: calc_profit(100, x)),
                                        odds["ML_a"].apply(lambda x: calc_profit(100, x)))
    odds["payout"] = np.where(odds["predicted_outcome"] == odds["win_lose"], odds["potential_payout"], -100)

    num_hit = odds[odds["predicted_outcome"] == odds["win_lose"]]["payout"].count()
    num_placed = odds["payout"].count()
    profit = odds["payout"].sum()
    print("Current Season performance:")
    print("\tNumber of bets hit: {}".format(num_hit))
    print("\tNumber of bets placed: {}".format(num_placed))
    print("\tAccuracy: {}%".format(round(num_hit / num_placed * 100)))
    print("\tProfit: ${}".format(round(profit)))
    print(odds.groupby(["Week"])["payout"].sum())
    odds = odds[["Home", "Away", "Week", "ML_h", "ML_a", "predicted_outcome", "win_lose"]]
    odds.to_csv("backend/data/predictions/season_picks.csv")


if __name__ == '__main__':
    week = 11
    favorite_index = [(0.4, -0.3), (0.5, -0.3), (0.6, -0.2), (0.6, -0.1), (0.7, -0.2), (0.7, 0.1), (0.7, 0.2)]
    underdog_index = [(0.3, -0.0), (0.4, -0.1), (0.4, 0.2), (0.5, -0.0), (0.5, 0.2), (0.6, 0.1)]
    scrape_vegas(week)
    current_season_odds(week)
    current_season(week, favorite_index, underdog_index)
    current_week(week, favorite_index, underdog_index)
