import random

class Game:
    def __init__(self, home_team_name, away_team_name):
        self.home_team_name = home_team_name
        self.away_team_name = away_team_name
        self.home_score = 0
        self.away_score = 0

    def coin_toss(self):
        return random.choice(["heads", "tails"])

    def kickoff(self):
        return random.randint(20, 80)

    def play_offense(self, team_name):
        print(f"{team_name} is on offense...")
        offense_play = random.choice(["pass", "run"])
        if offense_play == "pass":
            return random.randint(0, 30)
        else:
            return random.randint(0, 15)

    def play_defense(self):
        defense_play = random.choice(["pass", "run"])
        if defense_play == "pass":
            return random.randint(-5, 5)
        else:
            return random.randint(-3, 3)

    def turnover(self):
        return random.choice([True, False, False])

    def score_touchdown(self):
        return random.randint(6, 7)

    def score_field_goal(self):
        return random.randint(3, 4)

    def print_score(self):
        print(f"{self.home_team_name} {self.home_score} - {self.away_score} {self.away_team_name}\n")

    def simulate_possession(self, team_name, yard_line, score, down):
        print(f"{team_name} starts at their {yard_line}-yard line")
        while down <= 4:
            yards_to_go = 100 - yard_line
            print(f"{down} down and {yards_to_go} yards to go")
            
            offense_yards = self.play_offense(team_name)
            defense_bonus = self.play_defense()
            total_yards = offense_yards + defense_bonus
            
            if total_yards >= yards_to_go:
                score += self.score_touchdown()
                print(f"Touchdown! {team_name} scores!")
                return score

            if total_yards >= 10:
                yard_line += 10
                down = 1
            else:
                down += 1

            if self.turnover():
                print("Turnover! The defense intercepts the ball!")
                return score

            yard_line += max(0, total_yards)

        print("4th down! No more attempts, turnover on downs.")
        return score

    def simulate_quarter(self):
        kickoff_distance = self.kickoff()
        print(f"Kickoff distance: {kickoff_distance} yards")

        # Simulate possessions for the home team
        self.home_score = self.simulate_possession(self.home_team_name, kickoff_distance, self.home_score, 1)

        # Simulate possessions for the away team
        self.away_score = self.simulate_possession(self.away_team_name, kickoff_distance, self.away_score, 1)

    def start(self):
        print(f"Welcome to the NFL Football Game Simulation between {self.home_team_name} and {self.away_team_name}!")
        
        # Perform the coin toss
        coin_toss_result = self.coin_toss()
        print(f"Coin toss: {coin_toss_result}")
        if coin_toss_result == "heads":
            offense_team = self.home_team_name
            defense_team = self.away_team_name
        else:
            offense_team = self.away_team_name
            defense_team = self.home_team_name

        # Simulate the quarter
        self.simulate_quarter()

        print("\nEnd of the quarter!")
        self.print_score()


if __name__ == "__main__":
    home_team_name = "Pittsburgh Steelers"
    away_team_name = "Cleveland Browns"

    game = Game(home_team_name, away_team_name)
    game.start()
