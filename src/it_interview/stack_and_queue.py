# -*- coding: utf-8 -*-


class MyStack:
    """ 实现特殊额栈，实现栈的基本功能的基础上，实现返回栈中最小元素的操作
        pop, push, getMine O(1)
    """

    def __init__(self):
        self.stack_data = []  # 保存栈中的元素
        self.stack_min = []   # 保存每个操作的最小值

    def pop(self):
        if not self.stack_data:
            raise ValueError("StackData is Empty")
        value = self.stack_data.pop(-1)
        self.stack_data.pop(-1)
        return value

    def push(self, value):
        if not self.stack_min:
            self.stack_min.append(value)
        elif value < self.get_mine():
            self.stack_min.append(value)
        else:
            min_value = self.stack_min[-1]
            self.stack_min.append(min_value)
        self.stack_data.append(value)

    def get_mine(self):
        if not self.stack_min:
            raise ValueError("StackMin isEmpty")
        return self.stack_min[-1]


class QueueByStack:
    """ 用两个栈实现队列，支持add, poll, peek
    """

    def __init__(self):
        self.stack_push = []
        self.stack_pop = []

    def add(self, value):
        self.stack_push.append(value)

    def poll(self):
        if not self.stack_push and not self.stack_pop:
            raise ValueError("Empty queue")
        elif not self.stack_pop:
            while self.stack_push:
                self.stack_pop.append(self.stack_push.pop(-1))
        return self.stack_pop.pop(-1)

    def peek(self):
        if not self.stack_pop and not self.stack_push:
            raise ValueError("Empty queue")
        elif not self.stack_pop:
            while self.stack_push:
                self.stack_pop.append(self.stack_push.pop(-1))
        return self.stack_pop[-1]


def stack_sort(stack: list):
    """ 使用辅助栈对栈进行排序
    """
    _help = []
    while stack:
        cur = stack.pop(-1)
        while _help and _help[-1] > cur:
            stack.append(_help.pop(-1))
        _help.append(cur)
    while _help:
        stack.append(_help.pop(-1))


def max_slide_window_iter(nums: list, k: int) -> list:
    if not nums or len(nums) < k or k <= 0:
        return []
    size, ans = len(nums) - k + 1, []
    for i in range(size):
        ans.append(max(nums[i:i+k]))

    return ans


def max_slide_window(nums: list, k: int) -> list:
    if not nums or k <= 0 or k > len(nums):
        return []
    window, ans = [], []
    for i, x in enumerate(nums):
        # 窗口滑动弹出过期元素
        if i >= k and window[0] <= i-k:
            window.pop(0)
        while window and nums[window[-1]] <= x:
            window.pop()
        window.append(i)
        if i >= k-1:
            ans.append(nums[window[0]])
    return ans


def max_rec_size(nums: list):
    """ 求给定数组中1围成的矩形的最大面积
    """
    if not nums:
        return 0
    max_area, height = 0, [0] * len(nums[0])
    for i in range(len(nums)):
        for j in range(len(nums[i])):
            height[j] = 0 if nums[i][j] == 0 else height[j] + 1
        max_area = max(max_area, max_rec_from_bottom(height))
    return max_area


def max_rec_from_bottom(height: list) -> int:
    if not height:
        return 0
    max_area, stack = 0, []
    for i in range(len(height)):
        while stack and height[i] < height[stack[-1]]:
            j = stack.pop()
            k = -1 if not stack else stack[-1]
            cur_area = (i - k - 1) * height[j]
            max_area = max(max_area, cur_area)
        stack.append(i)
    while stack:
        j = stack.pop()
        k = -1 if not stack else stack[-1]
        cur_area = (len(height) - k - 1) * height[j]
        max_area = max(max_area, cur_area)
    return max_area


def get_max_tree(arr: list):
    """  通过给定的数组生成MaxTree,
    """
    pass


def get_num(arr: list, num: int) -> int:
    """  求给定数组中,子数组满足如下情况的数量
        max(arr[i~j]) - min(arr[i~j]) <= num
    """
    pass


if __name__ == '__main__':
    stack = [1, 5, 2, 4, 8]
    stack_sort(stack)
    print(stack)
    l1 = max_slide_window(stack, 2)
    l2 = max_slide_window_iter(stack, 2)
    print(l1, l2)

    s = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 0], [1, 0, 1, 1]]
    a = max_rec_size(s)
    print(a)

    a = [3, 2, 3, 0]
    ans = max_rec_from_bottom(a)
    print(ans)
