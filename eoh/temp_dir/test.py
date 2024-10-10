import os 

def test_subfunction(input: str):
    return "hahaha"

class TestClass:
    def __init__(self):
        self.name = "Test"

    def test_method(self):
        return test_subfunction("some_input")