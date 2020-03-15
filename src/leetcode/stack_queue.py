# -*- coding: utf-8 -*-


import heapq


def is_valid(s: str) -> bool:
    stack = []
    string_dict = {"(": ")", "{": "}", "[": "]"}
    for char in s:
        if char in string_dict.keys():
            stack.append(char)
        else:
            # } ] )开始或者 不成对
            if not stack or string_dict[stack[-1]] != char:
                return False
            stack.pop()

    return False if stack else True


class MinStack:

    def __init__(self):
        self.stack_data = []
        self.stack_min = []

    def push(self, x):
        self.stack_data.append(x)
        if self.stack_min:
            self.stack_min.append(min(x, self.stack_min[-1]))
        else:
            self.stack_min.append(x)

    def pop(self):
        self.stack_min.pop()
        self.stack_data.pop()

    def top(self):
        if self.stack_data:
            return self.stack_data[-1]
        else:
            return None

    def get_min(self):
        if self.stack_min:
            return self.stack_min[-1]


def find_k_largest(nums, k):
    topk = heapq.nlargest(k, nums)
    return topk[-1]


def find_k_largest_iter(nums, k):
    if len(nums) < k:
        return
    ans = nums[0: k]
    for elem in nums[k:]:
        min_elem = min(ans)
        if elem > min_elem:
            ans[ans.index(min_elem)] = elem
    return sorted(ans)[0]


class MyQueue:

    def __init__(self):
        self.data = []
        self.help = []

    def push(self, x):
        self.data.append(x)

    def pop(self):
        if not self.help:
            while self.data:
                self.help.append(self.data.pop())
        return self.help.pop()

    def peek(self):
        if not self.help:
            while self.data:
                self.help.append(self.data.pop())

        return self.help[-1]

    def empty(self):
        if not self.data and not self.help:
            return False

        return True


class MyStack:

    def __init__(self):
        self.data = []
        self.help = []

    def push(self, x):
        self.data.append(x)

    def pop(self):
        if not self.help:
            while self.data:
                self.help.append(self.data.pop(0))
        return self.help.pop()

    def top(self):
        if not self.help:
            while self.data:
                self.help.append(self.data.pop(0))
        return self.help[-1]

    def empty(self):
        if not self.data and not self.help:
            return False
        return True


def valid_push_pop(push, pop):
    if not push or not pop:
        return False

    stack = []
    for node in push:
        stack.append(node)

        while stack and stack[-1] == pop[0]:
            stack.pop()
            pop.pop(0)

    return not stack


if __name__ == '__main__':
    seq = [19, 1, 23, 25, 11, 9, 10, 19, 18]
    ans = find_k_largest_iter(seq, 4)
    print(ans)





