# -*- coding: utf-8 -*-

"""
"""


def two_number(seq: list, target: int) -> list:
    if not seq:
        return [-1, -1]

    exists = dict()
    for idx, n in enumerate(seq):
        if target - n in exists:
            return [idx, exists[target-n]]
        else:
            exists[n] = idx

    return [-1, -1]


def max_area(nums: list) -> int:
    """ 岛屿的最大面积
    """
    if not nums:
        return 0

    left, right, area = 0, len(nums) - 1, 0

    while left < right:
        area = max(area, min(nums[left], nums[right]) * (right - left))

        if nums[left] < nums[right]:
            left += 1
        else:
            right -= 1

    return area


def remove_duplicates(nums: list):
    """ 26 去除重复元素, 返回长度
    """
    if not nums:
        return 0
    left = 1

    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[left] = nums[i]
            left += 1

    return left


def remove_element(nums: list, val: int):
    """ 27　移除重复元素
    """
    if not nums:
        return

    left = 0

    for i in range(len(nums)):
        if nums[i] != val:
            nums[left] = nums[i]
            left += 1

    return left


def majority_element(nums: list):
    """ 169 过半元素
    """
    if not nums:
        return -1

    exists = dict()

    for num in nums:
        if num in exists:
            exists[num] += 1
        else:
            exists[num] = 1

        if exists[num] > len(nums) // 2:
            return num


def move_zeros(nums: list):
    """ 283 移动非０元素到数组前半部分，０到后半部分，不改变相对位置
    """
    left = 0

    for i in range(len(nums)):
        if nums[i] != 0:
            nums[left] = nums[i]
            left += 1

    while left < len(nums):
        nums[left] = 0
        left += 1


def find_disappeared_numbers(nums: list):
    """  448 查找数组中缺少的元素
    """
    if not nums:
        return []

    for val in nums:
        idx = abs(val) - 1
        nums[idx] = -nums[idx] if nums[idx] > 0 else nums[idx]
        # nums[idx] = -abs(nums[idx])

    return [idx+1 for idx, num in enumerate(nums) if num > 0]


if __name__ == '__main__':
    a = [1, 2, 0, 0, 0, 5, 0, 1]
    print(a)
    move_zeros(a)
    print(a)

    b = [4, 3, 2, 7, 8, 2, 3, 1]
    c = find_disappeared_numbers(b)
    print(c)



