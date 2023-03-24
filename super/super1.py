

class student:
    def __init__(self, name=None, gender=None, dep=None, ID=None):
        self.name = name
        self.gender = gender
        self.dep = dep
        self.ID = ID


class Person(student):
    def __init__(self, name, gender, dep, ID, pro_qua):
        super().__init__(name, gender, dep, ID)
        self.pro_qua = pro_qua

    def Profess(self):
        General_Programming = {'C', 'C++', 'C#', 'JAVA'}
        Statics_Programming = {'Python', 'R'}
        Engineering_Programming = {'Matlab', 'Fortran'}

        if self.pro_qua in General_Programming:
            return'General_Programming'
        elif self.pro_qua in Statics_Programming:
            return'Statics_Programming'
        elif self.pro_qua in Engineering_Programming:
            return 'Engineering_Programming'
        else:
            return 'Offices'

A = Person('Eason', 'Male', 'Civil Engineering', '9487943', 'Python')

if __name__ == '__main__':
    A = Person('Eason', 'Male', 'Civil Engineering', '9487943', 'Python')
    print('Name:{}  Department:{}  ID:{}'.format(A.name, A.dep, A.ID))
    print('Skill:{} ({})'.format(A.Profess(), A.pro_qua))

# output:
# Name:Eason     Department:Civil Engineering     ID:9487943
#Skill:Statics_Programming (Python)
