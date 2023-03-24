

from functools import reduce


class student:
    def __init__(self, name='Jack', gender='F', dep="IT", ID='9453', pro='Python'):
        self.name = name
        self.gender = gender
        self.dep = dep
        self.ID = ID
        self.pro = pro

    @property
    def gender(self):
        return self._gender

    @gender.setter
    def gender(self, gender):
        if gender == 'F':
            self._gender = 'Female'
        elif gender == 'M':
            self._gender = 'Male'
        else:
            raise TypeError('gender need F/M.')
        return


class final_grade:
    def __init__(self, we=[1, 2, 3], score=[1, 2, 3]):
        self.we = we
        self.score = score

    def calculate(self):
        weights = self.we
        scores = self.score
        ra = reduce(lambda a, b: a+b, map(lambda x, y: x*y, scores, weights))
        return ra


class Person(student, final_grade):
    def __init__(self, name, gender, dep, ID, we, score, bonus=0):
        student.__init__(self, name, gender, dep, ID)
        final_grade.__init__(self, we, score)
        self.bonus = bonus

    def total(self):
        tot = self.bonus+self.calculate()
        return tot

    def rank(self):
        total1 = self.total()
        if total1 >= 90:
            return 'A+'
        elif total1 >= 80 and total1 < 90:
            return 'A'
        elif total1 >= 70 and total1 < 80:
            return 'B'
        elif total1 >= 60 and total1 < 70:
            return 'C'
        else:
            return 'F'


# Main Function :
if __name__ == "__main__":
    
    w = [0.25, 0.35, 0.4]
    sc = [80, 60, 88]
    A = Person('Eason', 'M', 'Civil Engineering', '9487943', w, sc, 2)
    print('Name:{}  Gender:{}   Department:{}     ID:{}'.format(
        A.name, A.gender, A.dep, A.ID))
    print('Rank:{} (Initial Score:{})'.format(A.rank(), A.total()))

# output:
# Name:Eason  Gender:Male   Department:Civil Engineering     ID:9487943
# Rank:B (Initial Score:78.2)
