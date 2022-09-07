import pandas as pd
import numpy as np


def round_totals(number, base):
    return base * round(number / base)


def round_spreads(number, base):
    return base * round(number / base)


def all_seasons(df):
    # Create pivot table for all games
    table = df.groupby(['spread', 'total']).agg({'3_straight': ['sum', 'count']})
    table['perc'] = table[('3_straight', 'sum')] / table[('3_straight', 'count')]
    table['perc'] = table['perc'].apply(lambda x: 100 * round(x, 2))
    table = table.reset_index()
    table.columns = ["spread", "total", "sum", "count", "perc"]

    return table


def last_5_seasons(df):
    # Create pivot table for last 5 years of games
    five_years = df[df['season'] > 2022 - 5]
    last_five = five_years.groupby(['spread', 'total']).agg({'3_straight': ['sum', 'count']})
    last_five['perc'] = last_five[('3_straight', 'sum')] / last_five[('3_straight', 'count')]
    last_five['perc'] = last_five['perc'].apply(lambda x: 100 * round(x, 2))
    last_five = last_five.reset_index()
    last_five.columns = ["spread", "total", "sum", "count", "perc"]

    return last_five


def last_season(df):
    # Create pivot table for last season
    last_season = df[df['season'] == 2021]
    last_season = last_season.groupby(['spread', 'total']).agg({'3_straight': ['sum', 'count']})
    last_season["perc"] = last_season[('3_straight', 'sum')] / last_season[('3_straight', 'count')]
    last_season["perc"] = last_season['perc'].apply(lambda x: 100 * round(x, 2))
    last_season = last_season.reset_index()
    last_season.columns = ["spread", "total", "sum", "count", "perc"]

    return last_season


def write_files(table, five_years, past_season):
    with pd.ExcelWriter("backend/data/analytics/3_straight.xlsx") as writer:
        perc_table = pd.pivot_table(table, values="perc", index="total", columns="spread")
        perc_table = perc_table.fillna(0)
        perc_table.to_excel(writer, sheet_name="10 Year Percentages")

        count_table = pd.pivot_table(table, values="count", index="total", columns="spread")
        count_table = count_table.fillna(0)
        count_table.to_excel(writer, sheet_name="10 Year Counts")

        perc_table = pd.pivot_table(five_years, values="perc", index="total", columns="spread")
        perc_table = perc_table.fillna(0)
        perc_table.to_excel(writer, sheet_name="5 Year Percentages")

        count_table = pd.pivot_table(five_years, values="count", index="total", columns="spread")
        count_table = count_table.fillna(0)
        count_table.to_excel(writer, sheet_name="5 Year Counts")

        perc_table = pd.pivot_table(past_season, values="perc", index="total", columns="spread")
        perc_table = perc_table.fillna(0)
        perc_table.to_excel(writer, sheet_name="Last Season Percentages")

        count_table = pd.pivot_table(past_season, values="count", index="total", columns="spread")
        count_table = count_table.fillna(0)
        count_table.to_excel(writer, sheet_name="Last Season Counts")


def ui(table, five_years, past_season):
    done = False
    while done != 'Q':
        spread = round_spreads(float(input("What is the game's spread?  ")), 3.5)
        total = round_totals(float(input("What is the game's total?   ")), 2)

        table_info = table.set_index(['spread', 'total']).loc[(spread, total), ['perc', 'count']]
        five_years_info = five_years.set_index(['spread', 'total']).loc[(spread, total), ['perc', 'count']]
        past_season_info = past_season.set_index(['spread', 'total']).loc[(spread, total), ['perc', 'count']]

        print(
            f"""
            All seaons: {round(table_info['perc'])}% ({table_info['count']})
            Last 5 seasons: {round(five_years_info['perc'])}% ({five_years_info['count']})
            Past season: {round(past_season_info['perc'])}% ({past_season_info['count']})
            """
        )
        
        done = input('If done type: Q   ')


def pivot_table():
    # Load data
    scores = pd.read_csv('backend/data/games/scores.csv')
    scores['home'] = np.where(scores['home_field'], scores['team'], scores['opponent'])
    scores['away'] = np.where(scores['home_field'], scores['opponent'], scores['team'])
    scores = scores[['home', 'away', 'week', 'season', '3_straight']].drop_duplicates(['home', 'away', 'week', 'season'])
    scores.set_index(['home', 'away', 'week', 'season'], inplace=True)
    odds = pd.read_csv('backend/data/odds/odds.csv')
    odds.set_index(['home', 'away', 'week', 'season'], inplace=True)

    # Join data
    df = pd.merge(scores, odds, left_index=True, right_index=True)

    # Round spreads and totals
    df['spread'] = df['spread'].apply(lambda x: round_spreads(x, 3.5))
    df['total'] = df['total'].apply(lambda x: round_totals(x, 2))
    df.reset_index(inplace=True, drop=False)

    # Create pivot table for all games
    table = all_seasons(df)

    # Create pivot table for last 5 years of games
    five_years = last_5_seasons(df)

    # Create pivot table for last season
    past_season = last_season(df)

    write_files(table, five_years, past_season)

    ui(table, five_years, past_season)


def main():
    pivot_table()


if __name__ == '__main__':
    main()
