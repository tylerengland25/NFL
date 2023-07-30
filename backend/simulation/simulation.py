import random
from play_types import (
    Pass, Rush,
    Kick, Punt,
    FieldGoal
)

class Game:
    def __init__(self, home_team, away_team):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = 0
        self.away_score = 0
        self.quarter = 1
        self.secs_left = 900
        self.is_game_over = False
        self.box_score = {
            f"{self.home_team}_pass_yards": 0, f"{self.away_team}_pass_yards": 0,
            f"{self.home_team}_pass_plays": 0, f"{self.away_team}_pass_plays": 0,
            f"{self.home_team}_pass_tds":  0, f"{self.away_team}_pass_tds":  0,
            f"{self.home_team}_rush_yards": 0, f"{self.away_team}_rush_yards": 0,
            f"{self.home_team}_rush_plays": 0, f"{self.away_team}_rush_plays": 0,
            f"{self.home_team}_rush_tds":   0, f"{self.away_team}_rush_tds":   0,
            f"{self.home_team}_ints":       0, f"{self.away_team}_ints":       0,
            f"{self.home_team}_fums":       0, f"{self.away_team}_fums":       0,
        }

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
        self.yds_to_goal = 35
        play_description = f"{self.defense_team} will receive the ball"
        print(play_description)
    
    def kickoff(self):
        kickoff = Kick()
        kickoff.play()
        self.print_state(play=kickoff)
        self.update_state(kickoff)

    def turnover_on_downs(self):
        self.play_outcome = 'turnover on downs'
        self.yds_to_goal -= self.yds_on_play
        self.flip_field()
        self.change_possession()
        self.print_state()

    def change_possession(self):
        offense = self.offense_team
        self.offense_team = self.defense_team
        self.defense_team = offense

    def flip_field(self):
        self.yds_to_goal = 100 - self.yds_to_goal

        return 3

    def adjust_score(self, score):
        if self.offense_team == self.home_team:
            self.home_score += score
        else:
            self.away_score += score

    def adjust_clock(self):
        self.secs_left -= random.randint(30, 45)

    def is_touchdown(self):
        if self.yds_to_goal < 0:
            self.box_score[f"{self.offense_team}_{self.play_type}_tds"] += 1
            self.play_outcome = 'touchdown'
            self.adjust_score(self.score_touchdown())
            self.print_state()
            return True
        else:
            return False

    def is_turnover(self):
        if random.choice([True] + [False] * 19):
            if self.play_type == 'pass':
                self.box_score[f"{self.offense_team}_ints"] += 1
                self.play_outcome = 'interception'
            else:
                self.box_score[f"{self.offense_team}_fums"] += 1
                self.play_outcome = 'fumble'
            self.print_state()
            self.yds_to_goal -= self.yds_on_play
            self.flip_field()
            self.change_possession()
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
        print(f"{self.home_team} {self.home_score} - {self.away_score} {self.away_team}\n")

    def print_state(self, play):
        # Set the possession string
        possession = (
            f"{self.away_team} @ {self.home_team}*" 
            if self.offense_team == self.home_team 
            else f"*{self.away_team} @ {self.home_team}"
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
        yards_to_goal = f"{self.yds_to_goal} yards to goal"
        
        # Set the play description string
        if play.play_type == 'kickoff':
            play_description = f"{self.offense_team} kicked the ball {play.yds_kicked} yards."
            play_description = (
                play_description + " Touchback." 
                if play.touchback 
                else play_description + f" Returned for {play.yds_gained} yards."
            )

        # Set the fixed width for each part of the printed output
        possession = possession.ljust(10)
        clock = clock.ljust(10)
        down_distance = down_distance.ljust(10)
        yards_to_goal = yards_to_goal.ljust(15)
        play_description = play_description.ljust(40)
        
        print(f"| {possession} | {clock} | {down_distance} | {yards_to_goal} | {play_description}")


    def update_state(self, play):
        self.secs_left -= play.duration
        self.yds_to_goal -= play.yds_gained
        if play.change_of_possession:
            self.change_possession()
            self.flip_field()

    def simulate_play(self):
        play = random.choice([Pass(), Rush()])
        play.play()
        self.print_state(play=play)
        self.update_state(play)

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
            self.simulate_play()

            # Account for a turnover
            if self.is_turnover():
                return 'turnover'

            # Adjust boxscore
            self.box_score[f"{self.offense_team}_{self.play_type}_yards"] += self.yds_on_play
            
            # Check for touchdown
            if self.is_touchdown():
                return 'touchdown'

            # Adjust yards to goal
            self.yds_to_goal -= self.yds_on_play

            # Check for first down
            if self.yds_on_play >= self.distance:
                self.play_outcome = 'first down'
                self.print_state()
                self.distance = 10 if self.yds_to_goal >= 10 else self.yds_to_goal
                self.down = 1
            else:
                self.play_outcome = ''
                self.print_state()
                self.distance -= self.yds_on_play
                self.down += 1

            # Check for 4th down
            if self.down == 4:
                if self.yds_to_goal > 45:
                    self.punt()
                    return 'punt'
                elif self.yds_to_goal > 3:
                    self.field_goal()
                    return 'field goal'
            elif self.down == 5:
                self.turnover_on_downs()
                return 'turnover'

    def simulate_game(self):
        # Kickoff to start the game
        self.kickoff()

        # Simulate possessions until end of quarter
        while not self.is_game_over:
            ended_in = self.simulate_possession()
            if ended_in == "touchdown" or ended_in == "field goal":
                self.kickoff()
            elif ended_in == "halftime":
                # Kickoff to start the second half
                self.offense_team = self.away_team if self.home_started else self.home_team
                self.defense_team = self.home_team if self.home_started else self.away_team
                self.kickoff()

    def start(self):
        print(f"Welcome to the NFL Football Game Simulation between {self.home_team} and {self.away_team}!")
        
        # Perform the coin toss
        self.coin_toss()

        # Simulate the game
        self.simulate_game()

        print("\nEnd of the game!")
        self.print_score()


if __name__ == "__main__":
    home_team = "PIT"
    away_team = "CLE"

    game = Game(home_team, away_team)
    game.start()
