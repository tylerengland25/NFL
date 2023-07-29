import random

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
        return random.choice(["heads", "tails"])

    def kickoff(self):
        self.adjust_clock()
        return random.randint(40, 64)

    def play_offense(self):
        self.play_type = random.choice(["pass", "rush"])
        self.box_score[f"{self.offense_team}_{self.play_type}_plays"] += 1
        if self.play_type == "pass":
            self.secs_left -= random.randint(0, 20)
            return random.randint(0, 30)
        else:
            self.adjust_clock()
            return random.randint(0, 15)

    def play_defense(self):
        defense_play = random.choice(["pass", "rush"])
        if defense_play == "pass":
            return random.randint(-5, 5)
        else:
            return random.randint(-3, 3)

    def punt(self):
        self.adjust_clock()
        punt_yards = random.randint(30, 60)
        touchback = True if self.yds_to_goal - punt_yards < 0 else False
        self.yds_to_goal = 75 if touchback else self.yds_to_goal - punt_yards
        print(f"PUNT | {self.offense_team} punts the ball to {self.defense_team}")
        self.change_possession()

    def field_goal(self):
        self.adjust_clock()
        field_goal_yards = random.randint(30, 60)
        if field_goal_yards > self.yds_to_goal + 17:
            print(f"FIELD GOAL MISSED")
            self.change_possession()
            self.yds_to_goal = 100 - self.yds_to_goal
        else:
            self.adjust_score(self.score_field_goal())
            print(f"FIELD GOAL MADE")
            self.change_possession()

    def change_possession(self):
        offense = self.offense_team
        self.offense_team = self.defense_team
        self.defense_team = offense

    def score_touchdown(self):
        return 7

    def score_field_goal(self):
        return 3

    def print_score(self):
        print(f"{self.home_team} {self.home_score} - {self.away_score} {self.away_team}\n")

    def print_state(self):
        possession = (
            f"{self.away_team} @ {self.home_team}*" 
            if self.offense_team == self.home_team 
            else f"*{self.away_team} @ {self.home_team}"
        )
        quarter = f"Q{self.quarter}"
        time_left = (
            f"{self.secs_left // 60}:{self.secs_left % 60}" 
            if self.secs_left % 60 >= 10 
            else f"{self.secs_left // 60}:0{self.secs_left % 60}"
        )
        game_state = f"{possession} {quarter} {time_left}"
        down_distance = f"{self.down} & {self.distance}"
        yards_to_goal = f"{self.yds_to_goal} yards to goal"
        print(f"{game_state} | {down_distance} | {yards_to_goal} | ", end="")

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
            self.adjust_score(self.score_touchdown())
            print(f"TOUCHDOWN")
            return True
        else:
            return False

    def is_turnover(self):
        if random.choice([True] + [False] * 19):
            print(f"TURN OVER")
            self.adjust_clock()
            self.yds_to_goal = 100 - self.yds_to_goal
            if self.play_type == 'pass':
                self.box_score[f"{self.offense_team}_ints"] += 1
            else:
                self.box_score[f"{self.offense_team}_fums"] += 1
            self.change_possession()
            return True
        else:
            return False

    def is_end_of_quarter(self):
        if self.secs_left <= 0:
            print(f"END OF Q{self.quarter} | ", end="")
            self.print_score()
            self.secs_left = 900
            self.quarter += 1
            return True
        else:
            return False

    def simulate_possession(self):
        # Initialize down and distance 
        self.down = 1
        self.distance = 10
        print(f"{self.offense_team}'s ball. {self.down} & {self.distance} with {self.yds_to_goal} yards to goal")
        
        # Loop until punt, field goal or turnover
        while self.down <= 4:
            # Check for end of quarter
            if self.is_end_of_quarter():
                if self.quarter == 3:
                    return "halftime"
                if self.quarter == 5:
                    self.is_game_over = True
                    return "game over"

            # Print game state
            self.print_state()
            
            # Simlaute play
            offense_yards = self.play_offense()
            defense_bonus = self.play_defense()
            total_yards = offense_yards + defense_bonus

            # Account for a turnover
            if self.is_turnover():
                return 'turnover'

            # Adjust yards to goal
            self.yds_to_goal -= total_yards

            # Adjust boxscore
            self.box_score[f"{self.offense_team}_{self.play_type}_yards"] += total_yards
            
            # Check for touchdown
            if self.is_touchdown():
                return 'touchdown'

            # Check for first down
            if total_yards >= self.distance:
                print(f"FRIST DOWN | {self.offense_team} gains {total_yards} yards.")
                self.distance = 10 if self.yds_to_goal >= 10 else self.yds_to_goal
                self.down = 1
            else:
                print(f"{self.offense_team} gains {total_yards} yards.")
                self.distance -= total_yards
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
                self.change_possession()
                self.yds_to_goal = 100 - self.yds_to_goal    
                print("TUNROVER ON DOWNS")
                return 'turnover'

    def simulate_game(self):
        # Kickoff to start the game
        self.yds_to_goal = self.kickoff() + 35

        # Simulate possessions until end of quarter
        while not self.is_game_over:
            ended_in = self.simulate_possession()
            if ended_in == "touchdown" or ended_in == "field goal":
                self.yds_to_goal = self.kickoff() + 35
            elif ended_in == "halftime":
                # Kickoff to start the second half
                self.offense_team = self.away_team if self.home_started else self.home_team
                self.defense_team = self.home_team if self.home_started else self.away_team
                self.yds_to_goal = self.kickoff() + 35

    def start(self):
        print(f"Welcome to the NFL Football Game Simulation between {self.home_team} and {self.away_team}!")
        
        # Perform the coin toss
        coin_toss_result = self.coin_toss()
        print(f"Coin toss: {coin_toss_result}")
        if coin_toss_result == "heads":
            self.offense_team = self.home_team
            self.defense_team = self.away_team
            self.home_started = True
        else:
            self.offense_team = self.away_team
            self.defense_team = self.home_team
            self.home_started = False

        # Simulate the game
        self.simulate_game()

        print("\nEnd of the game!")
        self.print_score()


if __name__ == "__main__":
    home_team = "PIT"
    away_team = "CLE"

    game = Game(home_team, away_team)
    game.start()
