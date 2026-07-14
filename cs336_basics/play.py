# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# import regex as re

# text = """
# u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
# <|endoftext|>
# Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
# Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
# Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
# They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
# <|endoftext|>
# """

# print(re.findall(PAT, text))


class MyClass:
    a = "a"

    def instance_foo(self, arg):
        print(arg, self.a)

    @classmethod
    def class_foo(cls, arg):
        print(arg, cls.a)
    

MyClass.class_foo("类方法")

mycls = MyClass()
mycls.a = "b"
mycls.instance_foo("实例方法")

mycls.class_foo("类方法on instance")