import random
from play_types import (
    Pass, Rush,
    Kickoff, Punt,
    FieldGoal
)


class Game:
    def __init__(self, home_team, away_team):
        # Game states
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = 0
        self.away_score = 0
        self.quarter = 1
        self.secs_left = 900
        self.is_game_over = False
        self.box_score = {
            # Offensive stats
            f"{self.home_team}_pass_yards":     0, f"{self.away_team}_pass_yards":     0,
            f"{self.home_team}_pass_atts":      0, f"{self.away_team}_pass_atts":      0,
            f"{self.home_team}_pass_tds":       0, f"{self.away_team}_pass_tds":       0,
            f"{self.home_team}_pass_ints":      0, f"{self.away_team}_pass_ints":      0,
            f"{self.home_team}_rush_yards":     0, f"{self.away_team}_rush_yards":     0,
            f"{self.home_team}_rush_atts":      0, f"{self.away_team}_rush_atts":      0,
            f"{self.home_team}_rush_tds":       0, f"{self.away_team}_rush_tds":       0,
            f"{self.home_team}_rush_fums":      0, f"{self.away_team}_rush_fums":      0,

            # Defensive stats
            f"{self.home_team}_def_ints":       0, f"{self.away_team}_def_ints":       0,
            f"{self.home_team}_def_fums":       0, f"{self.away_team}_def_fums":       0,

            # Special teams stats
            f"{self.home_team}_kickoff_yards":  0, f"{self.away_team}_kickoff_yards":  0,
            f"{self.home_team}_kickoff_atts":   0, f"{self.away_team}_kickoff_atts":   0,
            f"{self.home_team}_punt_yards":     0, f"{self.away_team}_punt_yards":     0,
            f"{self.home_team}_punt_atts":      0, f"{self.away_team}_punt_atts":      0,
            f"{self.home_team}_fg_atts":        0, f"{self.away_team}_fg_atts":        0,
            f"{self.home_team}_fg_made":        0, f"{self.away_team}_fg_made":        0,
            f"{self.home_team}_fg_yards":       0, f"{self.away_team}_fg_yards":       0,
        }

        # Counter for state printing
        self.print_counter = 0

    def change_possession(self):
        offense = self.offense_team
        self.offense_team = self.defense_team
        self.defense_team = offense

    def flip_field(self):
        self.yds_to_goal = 100 - self.yds_to_goal

    def adjust_score(self, score):
        if self.offense_team == self.home_team:
            self.home_score += score
        else:
            self.away_score += score

    def adjust_clock(self, duration):
        self.secs_left -= duration

    def is_first_down(self, play):
        if play.yds_gained >= self.distance:
            self.yds_to_goal -= play.yds_gained
            self.distance = 10 if self.yds_to_goal >= 10 else self.yds_to_goal
            self.down = 1
            return True
        else:
            return False

    def is_touchdown(self, play):
        if play.yds_gained >= self.yds_to_goal:
            self.box_score[f"{self.offense_team}_{play.play_type}_tds"] += 1
            self.adjust_score(7)
            self.yds_to_goal = 65
            return True
        else:
            return False

    def is_turnover(self, play):
        if play.change_of_possession:
            turnover_type = 'ints' if play.play_type == 'pass' else 'fums'
            self.box_score[f"{self.offense_team}_{play.play_type}_{turnover_type}"] += 1
            self.box_score[f"{self.defense_team}_def_{turnover_type}"] += 1
            self.change_possession()
            self.flip_field()
            return True
        else:
            return False
        
    def is_turnover_on_downs(self, play):
        if self.down == 4 and play.yds_gained < self.distance:
            self.change_possession()
            self.flip_field()
            return True
        else:
            return False

    def is_end_of_quarter(self):
        if self.secs_left <= 0:
            self.print_score()
            self.secs_left = 900
            self.quarter += 1
            return True
        else:
            return False

    def print_score(self):
        print(f"{self.away_team} {self.away_score} - {self.home_score} {self.home_team}")

    def print_state(self, play):
        # Set the scoreboard string
        scoreboard = (
            f"{self.away_team} {self.away_score} @ {self.home_score} *{self.home_team}" 
            if self.offense_team == self.home_team 
            else f"*{self.away_team} {self.away_score} @ {self.home_score} {self.home_team}"
        )

        # Set the clock string
        quarter = f"Q{self.quarter}"
        time_left = (
            f"{self.secs_left // 60}:{self.secs_left % 60}" 
            if self.secs_left % 60 >= 10 
            else f"{self.secs_left // 60}:0{self.secs_left % 60}"
        )
        clock = f"{quarter} {time_left}"
        
        # Set the down and distance string
        down_distance = f"{self.down} & {self.distance}" if play.play_type != 'kickoff' else 'Kickoff'
        
        # Set the yards to goal string
        yards_to_goal = f"{self.yds_to_goal} yds to goal"

        # Set the play duration string
        play_duration = f"{play.duration} secs"

        # Set the play description string
        if play.play_type == 'kickoff':             # Play description for kickoff
            play_description = f"{self.offense_team} kicked the ball {play.yds_kicked} yards."
            if play.touchback:
                play_description += " TOUCHBACK."
            else:
                play_description += f" Returned for {play.yds_returned} yards."
        elif play.play_type == 'pass':              # Play description for pass
            play_description = f"{self.offense_team} passed the ball for {play.yds_gained} yards."
            if play.change_of_possession:
                play_description += " INTERCEPTION."
            elif play.touchdown:
                play_description += " TOUCHDOWN."
            elif play.first_down:
                play_description += " FIRST DOWN."
        elif play.play_type == 'rush':              # Play description for rush
            play_description = f"{self.offense_team} rushed the ball for {play.yds_gained} yards."
            if play.change_of_possession:
                play_description += " FUMBLE."
            elif play.touchdown:
                play_description += " TOUCHDOWN."
            elif play.first_down:
                play_description += " FIRST DOWN."
        elif play.play_type == 'punt':              # Play description for punt
            play_description = f"{self.offense_team} punted the ball {play.yds_kicked} yards."
            if play.touchback:
                play_description += " TOUCHBACK."
            else:
                play_description += f" Returned for {play.yds_returned} yards."
        elif play.play_type == 'field goal':        # Play description for field goal
            made = 'MADE' if play.made else 'MISSED'
            play_description = f"{self.offense_team} {made} a {self.yds_to_goal} yard field goal."
        
        # Headers and states
        headers = ["Scoreboard", "Clock", "Down & Distance", "Yards to Goal", "Play Duration", "Play Description"]
        states = [scoreboard, clock, down_distance, yards_to_goal, play_duration, play_description]

        # Set column widths
        preset_widths = [16, 8, 7, 14, 7, 55]
        column_widths = [max(len(header), width) for header, width in zip(headers, preset_widths)]

        # Header row
        header_row = " | ".join(header.ljust(width) for header, width in zip(headers, column_widths))
        state_row = " | ".join(state.ljust(width) for state, width in zip(states, column_widths))

        # Print the header ever 15 plays
        if self.print_counter % 15 == 0:
            print("-" * len(header_row))
            print(header_row)
            print("-" * len(header_row))
        print(state_row)
        self.print_counter += 1

    def print_boxscore(self):
        team_names = [self.home_team, self.away_team]
        tables = {
            "Offensive Stats": [
                "Team", 
                "Pass Yards", "Pass Atts", "Pass TDs", "Pass INTs", 
                "Rush Yards", "Rush Atts", "Rush TDs", "Rush Fums"
            ],
            "Defensive Stats": [
                "Team", "Def INTs", "Def Fums"
            ],
            "Special Teams Stats": [
                "Team",
                "Punt Yards", "Punt Atts", 
                "FG Atts", "FG Made", "FG Yards"
            ]
        }

        for table in tables:
            print(f"\n{table}")
            
            # Calculate maximum width for each column
            headers = tables[table]
            column_widths = [len(header) for header in headers]

            # Print the header row
            header_row = " | ".join(header.rjust(width) for header, width in zip(headers, column_widths))
            print("-" * len(header_row))
            print(header_row)
            print("-" * len(header_row))

            for team_name in team_names:
                team_stats = [f"{team_name}".rjust(column_widths[0])]

                for stat, width in zip(headers[1:], column_widths[1:]):
                    team_stats.append(
                        str(self.box_score[f"{team_name}_{stat.lower().replace(' ', '_')}"]).rjust(width)
                    )

                row = " | ".join(team_stats)
                print(row)
                print("-" * len(header_row))

    def update_state(self, play):
        self.adjust_clock(play.duration)
        if play.play_type in ['kickoff', 'punt']:
            self.yds_to_goal -= play.yds_gained
            self.box_score[f"{self.offense_team}_{play.play_type}_yards"] += play.yds_kicked
            self.box_score[f"{self.offense_team}_{play.play_type}_atts"] += 1
            self.change_possession()
            self.flip_field() 
            return play.play_type
        elif play.play_type == 'field goal':
            self.box_score[f"{self.offense_team}_fg_atts"] += 1
            if play.made:
                self.box_score[f"{self.offense_team}_fg_made"] += 1
                self.box_score[f"{self.offense_team}_fg_yards"] += play.yds_kicked
                self.yds_to_goal = 65
                self.adjust_score(3)
                return play.play_type
            else:
                self.change_possession()
                self.flip_field()
                return 'turnover'
        else:
            self.box_score[f"{self.offense_team}_{play.play_type}_yards"] += play.yds_gained
            self.box_score[f"{self.offense_team}_{play.play_type}_atts"] += 1
            if self.is_turnover(play):
                return 'turnover'
            elif self.is_touchdown(play):
                return 'touchdown'
            elif self.is_first_down(play):
                return 'first down'
            elif self.is_turnover_on_downs(play):
                return 'turnover'
            else:
                self.yds_to_goal -= play.yds_gained
                self.distance -= play.yds_gained
                self.down += 1

    def coin_toss(self):
        coin_toss_result = random.choice(["heads", "tails"])
        if coin_toss_result == "heads":
            self.offense_team = self.home_team
            self.defense_team = self.away_team
            self.home_started = True
        else:
            self.offense_team = self.away_team
            self.defense_team = self.home_team
            self.home_started = False
        self.yds_to_goal = 65
        play_description = f"{self.defense_team} will receive the ball"
        print(play_description)
    
    def simulate_kickoff(self):
        kickoff = Kickoff(self.yds_to_goal)
        kickoff.sim()
        self.print_state(play=kickoff)
        self.update_state(kickoff)

    def simulate_play(self):
        # Sim play
        if self.down < 4 or (self.yds_to_goal <= 3 and self.down == 4):
            play = random.choice([
                Pass(self.yds_to_goal, self.distance), 
                Rush(self.yds_to_goal, self.distance)
            ])
        elif self.yds_to_goal > 50:
            play = Punt(self.yds_to_goal)
        elif self.yds_to_goal > 3:
            play = FieldGoal(self.yds_to_goal)
        play.sim()
        self.print_state(play=play)
        return self.update_state(play)

    def simulate_possession(self):
        # Initialize down and distance 
        self.down = 1
        self.distance = 10
    
        # Loop until punt, field goal or turnover
        while self.down <= 4:
            # Check for end of quarter
            if self.is_end_of_quarter():
                if self.quarter == 3:
                    return "halftime"
                if self.quarter == 5:
                    self.is_game_over = True
                    return "game over"
            
            # Simlaute play
            result = self.simulate_play()

            # Check for touchdown, turnover, punt, field goal
            if result in ["touchdown", "turnover", "punt", "field goal"]:
                return result

    def simulate_game(self):
        print(f"Welcome to the NFL Football Game Simulation between {self.home_team} and {self.away_team}!")
        
        # Perform the coin toss
        self.coin_toss()

        # Kickoff to start the game
        self.simulate_kickoff()

        # Simulate possessions until end of quarter
        while not self.is_game_over:
            ended_in = self.simulate_possession()
            if ended_in == "touchdown" or ended_in == "field goal":
                self.simulate_kickoff()
            elif ended_in == "halftime":
                # Kickoff to start the second half
                self.offense_team = self.away_team if self.home_started else self.home_team
                self.defense_team = self.home_team if self.home_started else self.away_team
                self.simulate_kickoff()

        print("\nEnd of the game!")
        self.print_score()
        self.print_boxscore()


if __name__ == "__main__":
    home_team = "PIT"
    away_team = "CLE"

    game = Game(home_team, away_team)
    game.simulate_game()
