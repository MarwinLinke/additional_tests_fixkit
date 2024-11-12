from typing import List

class RepairEvaluationMatrix():

    def __init__(
        self,
        original_passing: List[str],
        original_failing: List[str],
        patched_passing: List[str],
        patched_failing: List[str]
    ):
        
        self.len_original_passing = len(original_passing)
        self.len_original_failing = len(original_failing)
        self.len_total = self.len_original_failing + self.len_original_passing

        self.still_passing: List[str] = []
        self.now_failing: List[str] = []
        self.now_passing: List[str] = []
        self.still_failing: List[str] = []

        self.undefined: List[str] = []

        for passing in original_passing:
            if passing in patched_passing:
                self.still_passing.append(passing)
            elif passing in patched_failing:
                self.now_failing.append(passing)
            else:
                self.undefined.append(passing)

        for failing in original_failing:
            if failing in patched_passing:
                self.now_passing.append(failing)
            elif failing in patched_failing:
                self.still_failing.append(failing)
            else:
                self.undefined.append(failing)

        self.len_still_passing = len(self.still_passing)
        self.len_now_failing = len(self.now_failing)
        self.len_now_passing = len(self.now_passing)
        self.len_still_failing = len(self.still_failing)

        if (self.len_now_passing + self.len_now_failing) > 0:
            self.precision = self.len_now_passing / (self.len_now_passing + self.len_now_failing)
        else:
            self.precision = 0.0

        if (self.len_now_passing + self.len_still_failing) > 0:
            self.recall = self.len_now_passing / (self.len_now_passing + self.len_still_failing)
        else:
            self.recall = 0.0

        if self.len_total > 0:
            self.accuracy = (self.len_still_passing + self.len_now_passing) / self.len_total
        else:
            self.accuracy = 0.0
    
    def __str__(self):
        string = "\n---------- Evaluation Matrix ----------\n"
        string += f"STILL PASSING: [{self.len_still_passing}/{self.len_original_passing}]\n"
        string += f"NOW FAILING: [{self.len_now_failing}/{self.len_original_passing}]\n"
        string += f"NOW PASSING: [{self.len_now_passing}/{self.len_original_failing}]\n"
        string += f"STILL FAILING: [{self.len_still_failing}/{self.len_original_failing}]\n"
        string += f"PRECISION: {self.precision}\n"
        string += f"RECALL: {self.recall}\n"
        string += f"ACCURACY: {self.accuracy}\n"
        string += f"---------------------------------------"
        return string

