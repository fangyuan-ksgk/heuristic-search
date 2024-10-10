from .test import TestClass

def another_function():
    return "hey you"

class CompositeClass:
    def __init__(self):
        self.test_class = TestClass()

    def composite_method(self):
        out = self.test_class.test_method()
        out2 = another_function()
        return out + out2