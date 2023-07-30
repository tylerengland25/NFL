import random
from abc import ABC, abstractmethod


class Play(ABC):
    def __init__(self, yds_to_goal, distance) -> None:
        self.yds_to_goal = yds_to_goal
        self.distance = distance
        self.play_type = None
        self.yds_gained = None
        self.duration = random.randint(5, 15)
        self.change_of_possession = False
        self.touchdown = False
        self.first_down = False

    @abstractmethod
    def sim(self):
        defensive_play = random.choice(["run", "pass"])
        if defensive_play == self.play_type:
            self.yds_gained = random.randint(-5, 5)
        else:
            self.yds_gained = random.randint(0, 15)
        self.change_of_possession = random.choice(
            [True] * 5 + [False] * 95
        )
        if self.yds_gained > self.yds_to_goal and not self.change_of_possession:
            self.yds_gained = self.yds_to_goal
            self.touchdown = True
        elif self.yds_gained >= self.distance and not self.change_of_possession:
            self.first_down = True


class Pass(Play):
    def __init__(self, yds_to_goal, distance) -> None:
        super().__init__(yds_to_goal, distance)
        self.play_type = "pass"

    def sim(self):
        super().sim()


class Rush(Play):
    def __init__(self, yds_to_goal, distance) -> None:
        super().__init__(yds_to_goal, distance)
        self.play_type = "rush" 

    def sim(self):
        super().sim()


class SpecialTeams(Play):
    def __init__(self, yds_to_goal) -> None:
        super().__init__(yds_to_goal, distance=None)
        self.yds_kicked = random.randint(30, 60)
        self.touchback = True if self.yds_kicked > self.yds_to_goal else False

    def sim(self):
        self.yds_returned = random.randint(0, 30)
        self.yds_gained = self.yds_to_goal - 25 if self.touchback else self.yds_kicked - self.yds_returned
        self.change_of_possession = True


class Kickoff(SpecialTeams):
    def __init__(self, yds_to_goal) -> None:
        super().__init__(yds_to_goal)
        self.play_type = "kickoff"
        self.yds_kicked = random.randint(40, 80)
        self.touchback = True if self.yds_kicked > self.yds_to_goal else False


class Punt(SpecialTeams):
    def __init__(self, yds_to_goal) -> None:
        super().__init__(yds_to_goal)
        self.play_type = "punt"


class FieldGoal(SpecialTeams):
    def __init__(self, yds_to_goal) -> None:
        super().__init__(yds_to_goal)
        self.play_type = "field goal"
        self.made = True if self.yds_kicked > yds_to_goal + 17 else False

    def sim(self):
        self.yds_gained = 0