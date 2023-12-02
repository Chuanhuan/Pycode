import numpy as np
import matplotlib.pyplot as plt

class person:
    def __init__(self, x=.5) -> None:
        self.s = x
        self.m = 10

    def luck(self):
        if np.random.rand() < self.s:
            self.m = self.m * 2

    def bad(self):
        self.m = self.m / 2

    def wealthy(self):
        return self.m 


p1 = person(.9)

p1.bad()
print(p1.m)


p_list = []
n = 10**5
for i in range(0, 10^5):
    skill = np.random.randn()
    p_list.append(person(skill))

# simulation start

for i in range(0, 10^2):
    samples = np.random.choice(range(n), size=250, replace=False)
    
