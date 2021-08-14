import mysql.connector
import pandas as pd
import Rosters
from Schedule import scrape_all_schedules


def main():
    db = mysql.connector.connect(
        host="localhost",
        user="tyler",
        passwd="root",
        db="SeniorProject"
    )

    mycursor = db.cursor()

    mycursor.execute("DROP TABLE IF EXISTS Rosters")
    mycursor.execute("DROP TABLE IF EXISTS Schedules")
    mycursor.execute("DROP TABLE IF EXISTS PlayerDefense")
    mycursor.execute("DROP TABLE IF EXISTS PlayerPassing")
    mycursor.execute("DROP TABLE IF EXISTS PlayerReceiving")
    mycursor.execute("DROP TABLE IF EXISTS PlayerRushing")
    mycursor.execute("DROP TABLE IF EXISTS PlayerKicking")
    mycursor.execute("DROP TABLE IF EXISTS PlayerPunting")
    mycursor.execute("DROP TABLE IF EXISTS Players")

    mycursor.execute("SHOW TABLES")
    for table in mycursor:
        print(table)

    mycursor.execute("""
    CREATE TABLE Players(
        PId VARCHAR(10) PRIMARY KEY,
        Name VARCHAR(100),
        Age INTEGER,
        College VARCHAR(100)
    )""")

    mycursor.execute("""
    CREATE TABLE Rosters(
        SId VARCHAR(3),
        TId VARCHAR(3),
        PId VARCHAR(10),
        Pos VARCHAR(3),
        G INTEGER, 
        GS INTEGER,
        Num DOUBLE,
        PRIMARY KEY (SId, TId, PId),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (TId) REFERENCES Teams(TId),
        FOREIGN KEY (PId) REFERENCES Players(PId)
    )""")

    mycursor.execute("""
    CREATE TABLE PlayerPassing(
        SId VARCHAR(3),
        TId VARCHAR(3),
        PId VARCHAR(10),
        Att INTEGER,
        Cmp INTEGER, 
        Yds INTEGER,
        YPA DOUBLE,
        YPG DOUBLE,
        TD INTEGER,
        Ints INTEGER, 
        Sack INTEGER,
        QBR DOUBLE,
        PRIMARY KEY (SId, TId, PId),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (TId) REFERENCES Teams(TId),
        FOREIGN KEY (PId) REFERENCES Players(PId)
    )""")

    mycursor.execute("""
    CREATE TABLE PlayerRushing(
        SId VARCHAR(3),
        TId VARCHAR(3),
        PId VARCHAR(10),
        Att INTEGER,
        Yds INTEGER, 
        Avg DOUBLE,
        YPG DOUBLE,
        Lg INTEGER,
        TD INTEGER,
        `10+` INTEGER,
        PRIMARY KEY (SId, TId, PId),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (TId) REFERENCES Teams(TId),
        FOREIGN KEY (PId) REFERENCES Players(PId)
    )""")

    mycursor.execute("""
    CREATE TABLE PlayerReceiving(
        SId VARCHAR(3),
        TId VARCHAR(3),
        PId VARCHAR(10),
        Rec INTEGER,
        Yds INTEGER, 
        Avg DOUBLE,
        YPG DOUBLE,
        Lg INTEGER,
        TD INTEGER,
        `20+` INTEGER,
        Tar INTEGER,
        YAC INTEGER,
        PRIMARY KEY (SId, TId, PId),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (TId) REFERENCES Teams(TId),
        FOREIGN KEY (PId) REFERENCES Players(PId)
    )""")

    mycursor.execute("""
    CREATE TABLE PlayerDefense(
        SId VARCHAR(3),
        TId VARCHAR(3),
        PId VARCHAR(10),
        Ints INTEGER,
        IntYds INTEGER, 
        IntAvg DOUBLE,
        IntLong INTEGER,
        IntTD INTEGER,
        Solo INTEGER,
        Ast INTEGER,
        Tot INTEGER,
        Sack INTEGER,
        SackYds INTEGER,
        PRIMARY KEY (SId, TId, PId),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (TId) REFERENCES Teams(TId),
        FOREIGN KEY (PId) REFERENCES Players(PId)
    )""")

    mycursor.execute("""
    CREATE TABLE PlayerKicking(
        SId VARCHAR(3),
        TId VARCHAR(3),
        PId VARCHAR(10),
        FGM INTEGER,
        FGA INTEGER, 
        FPct DOUBLE,
        `0-19` VARCHAR(10),
        `20-29` VARCHAR(10),
        `30-39` VARCHAR(10),
        `40-49` VARCHAR(10),
        `50+` VARCHAR(10),
        Lg INTEGER,
        XPM INTEGER,
        XPA INTEGER,
        XPct DOUBLE,
        Pts INTEGER,
        PRIMARY KEY (SId, TId, PId),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (TId) REFERENCES Teams(TId),
        FOREIGN KEY (PId) REFERENCES Players(PId)
    )""")

    mycursor.execute("""
    CREATE TABLE PlayerPunting(
        SId VARCHAR(3),
        TId VARCHAR(3),
        PId VARCHAR(10),
        Num INTEGER,
        Yds INTEGER, 
        Avg DOUBLE,
        Lg INTEGER,
        TB INTEGER,
        `In20` INTEGER,
        `50+` DOUBLE,
        Blk INTEGER,
        PRIMARY KEY (SId, TId, PId),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (TId) REFERENCES Teams(TId),
        FOREIGN KEY (PId) REFERENCES Players(PId)
    )""")

    mycursor.execute("""
    CREATE TABLE Schedules(
        SId VARCHAR(3),
        TId VARCHAR(3),
        Date DATE,
        Home VARCHAR(3),
        Away VARCHAR(3),
        H_Q1 Integer,
        H_Q2 Integer,
        H_Q3 Integer,
        H_Q4 Integer,
        A_Q1 Integer,
        A_Q2 Integer,
        A_Q3 Integer,
        A_Q4 Integer,
        H_Final Integer,
        A_Final Integer,
        PRIMARY KEY (SId, TId, Date),
        FOREIGN KEY (SId) REFERENCES Seasons(SId),
        FOREIGN KEY (Home) REFERENCES Teams(TId),
        FOREIGN KEY (Away) REFERENCES Teams(TId)
    )""")

    rosters = Rosters.scrape_all_rosters()
    rosters = rosters.replace(["--", "-"], 0)
    print("Rosters done")

    # players
    players = rosters.drop_duplicates(subset=["PId"])
    sql = """INSERT INTO Players (PId, Name, Age, College)
                VALUES (%s, %s, %s, %s)"""
    values = []
    for index, row in players.iterrows():
        values.append((row["PId"], row["Name"], row["Age"], row["College"]))

    mycursor.executemany(sql, values)

    # rosters
    sql = """INSERT INTO Rosters (SId, TId, PId, Pos, G, GS, Num)
        VALUES (%s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in rosters.iterrows():
        values.append((row["SId"], row["TId"], row["PId"], row["Pos"], row["G"], row["GS"], row["Number"]))

    mycursor.executemany(sql, values)

    passing = pd.read_csv("passing.csv")
    rushing = pd.read_csv("rushing.csv")
    receiving = pd.read_csv("receiving.csv")
    defense = pd.read_csv("defense.csv")
    kicking = pd.read_csv("kicking.csv")
    punting = pd.read_csv("punting.csv")

    # insert into passing
    passing = passing.replace(["--", "-"], 0)
    sql = """INSERT INTO PlayerPassing (SId, TId, PId, Att, Cmp, Yds, YPA, YPG, TD, Ints, Sack, QBR)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in passing.iterrows():
        values.append((row["SId"], row["TId"], row["PId"], row["Att"], row["Cmp"], row["Yds"].replace(',', ''),
                       row["YPA"], row["YPG"], row["TD"], row["Int"], row["Sack"], row["QBR"]))

    mycursor.executemany(sql, values)

    # insert into rushing
    rushing = rushing.replace(["--", "-"], 0)
    sql = """INSERT INTO PlayerRushing (SId, TId, PId, Att, Yds, Avg, YPG, Lg, TD, `10+`)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in rushing.iterrows():
        values.append((row["SId"], row["TId"], row["PId"], row["Att"], row["Yds"].replace(',', ''),
                       row["Avg"], row["YPG"], row["Long"].replace('t', ''), row["TD"], row["10+"]))

    mycursor.executemany(sql, values)

    # insert into receiving
    receiving = receiving.replace(["--", "-"], 0)
    sql = """INSERT INTO PlayerReceiving (SId, TId, PId, Rec, Yds, Avg, YPG, Lg, TD, `20+`, Tar, YAC)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in receiving.iterrows():
        values.append((row["SId"], row["TId"], row["PId"], row["Rec"], row["Yds"].replace(',', ''), row["Avg"],
                       row["YPG"], str(row["Long"]).replace('t', ''), row["TD"], row["20+"], row["Tar"], row["YAC"]))

    mycursor.executemany(sql, values)

    # insert into defense
    defense = defense.replace(["--", "-"], 0)
    sql = """INSERT INTO
    PlayerDefense (SId, TId, PId, Ints, IntYds, IntAvg, IntLong, IntTD, Solo, Ast, Tot, Sack, SackYds)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in defense.iterrows():
        values.append((row["SId"], row["TId"], row["PId"], row["Int"], row["IntYds"], row["IntAvg"],
                       str(row["IntLong"]).replace('t', ''), row["IntTd"], row["Solo"], row["Ast"], row["Tot"],
                       row["Sack"], row["SackYds"]))

    mycursor.executemany(sql, values)

    # insert into kicking
    kicking = kicking.replace(["--", "-"], 0)
    sql = """INSERT INTO
    PlayerKicking (SId, TId, PId, FGM, FGA, FPct, `0-19`, `20-29`, `30-39`, `40-49`, `50+`, Lg, XPM, XPA, XPct, Pts)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in kicking.iterrows():
        values.append((row["SId"], row["TId"], row["PId"], row["FGM"], row["FGA"], row["FPct"], row["0-19"],
                       row["20-29"], row["30-39"], row["40-49"], row["50+"], str(row["Long"]).replace('t', ''),
                       row["XPM"], row["XPA"], row["XPct"], row["Pts"]))

    mycursor.executemany(sql, values)

    # insert into punting
    punting = punting.replace(["--", "-"], 0)

    sql = """INSERT INTO
    PlayerPunting (SId, TId, PId, Num, Yds, Avg, Lg, TB, `In20`, `50+`, Blk)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in punting.iterrows():
        values.append((row["SId"], row["TId"], row["PId"], row["Num"], str(row["Yds"]).replace(',', ''), row["Avg"],
                       str(row["Long"]).replace('t', ''), row["TB"], row["In20"], row["50+"], row["Blk"]))

    mycursor.executemany(sql, values)

    # insert into schedules
    print("Schedules beginning")
    schedules = scrape_all_schedules()
    print("Schedules done")
    sql = """INSERT INTO
        Schedules (SId, TId, Date, Home, Away, H_Q1, H_Q2, H_Q3, H_Q4, A_Q1, A_Q2, A_Q3, A_Q4, H_Final, A_Final)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = []
    for index, row in schedules.iterrows():
        date = row["Date"].split()[1].split('/')
        month = date[0]
        day = date[1]
        year = row["Year"]
        if len(month) == 1 and len(day) == 1:
            date = year + "-0" + month + "-0" + day
        elif len(month) == 1:
            date = year + "-0" + month + "-" + day
        elif len(day) == 1:
            date = year + "-" + month + "-0" + day
        else:
            date = year + "-" + month + "-" + day
        values.append((row["SId"], row["TId"], date, row["Home"], row["Away"], row["H_Q1"], row["H_Q2"], row["H_Q3"],
                       row["H_Q4"], row["A_Q1"], row["A_Q2"], row["A_Q3"], row["A_Q4"], row["H_final"], row["A_final"]))

    mycursor.executemany(sql, values)

    db.commit()

    mycursor.execute("SELECT * FROM Schedules")
    for player in mycursor:
        print(player)


if __name__ == '__main__':
    main()
