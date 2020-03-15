# coding: utf-8 -*-

"""
"""


class MyStack:
    """ 实现栈的几班功能，所有操作的时间复杂度为O(1), push, pop, getMin
    """

    def __init__(self):
        self.min = []
        self.stack = []

    def push(self, data):
        if not self.min:
            self.min.append(data)
        elif self.min[0] > data:
            self.min[0] = data
        self.stack.append(data)

    def pop(self):
        if not self.stack:
            raise ValueError("Empty stack")
        data = self.stack.pop()
        if self.min[0] == data:
            self.min.pop()
        return data

    def get_min(self):
        return self.min[0] if self.min else "EMPTY"


class Stack2Queue:
    """ 使用两个栈实现队列，实现基本操作add, poll, peek
    """

    def __init__(self):
        self.left = []
        self.right = []

    def add(self, data):
        self.left.append(data)

    def poll(self):
        if not self.left and not self.right:
            raise ValueError(f"Queue is Empty!")
        elif not self.right:
            while self.left:
                self.right.append(self.left.pop())
        return self.right.pop()

    def peek(self):
        if not self.left and not self.right:
            raise ValueError(f"Queue is Empty")
        elif not self.right:
            while self.left:
                self.right.append(self.left.pop())
        return self.right.pop()

