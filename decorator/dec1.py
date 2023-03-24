# [ref](https://medium.com/citycoddee/python%E9%80%B2%E9%9A%8E%E6%8A%80%E5%B7%A7-3-%E7%A5%9E%E5%A5%87%E5%8F%88%E7%BE%8E%E5%A5%BD%E7%9A%84-decorator-%E5%97%B7%E5%97%9A-6559edc87bc0)

class Dog:
    def __init__(self, func):
        self.talent = func

    def bark(self):
        print("Bark !!!")


@Dog
def dog_can_pee():
    print("I can pee very hard......")


@Dog
def dog_can_jump():
    print("I can jump uselessly QQQ")


@Dog
def dog_can_poo():
    print("I can poo like a super pooping machine!")


dog_11 = dog_can_jump

if __name__ == "__main__":
    dog_1 = dog_can_pee
    dog_1.talent()
    # > I can pee very hard......

    dog_2 = dog_can_jump
    dog_2.talent()
    # > I can jump uselessly QQQ

    dog_3 = dog_can_poo
    dog_3.talent()
    # > I can poo like a super pooping machine!
