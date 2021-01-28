# -*- coding: utf-8 -*-

"""
"""
import os
import psutil


def reverse(s, left, right):
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


class RotateStr:
    def __init__(self, s, f, t):
        self.s = list(s)
        self.f = f
        self.t = t

    def rotate(self):
        self._sup(0, self.f)
        self._sup(self.f+1, self.t)
        self._sup(0, self.t)

    def _sup(self, start, end):
        while start < end:
            self.s[start], self.s[end] = self.s[end], self.s[start]
            start += 1
            end -= 1


def two_sum(nums: list, target: int) -> list:
    if not nums or target is None:
        return [-1, -1]

    exists = dict()
    for idx, num in enumerate(nums):
        if target - num in exists:
            return [idx, exists[target - num]]
        exists[num] = idx

    return [-1, -1]


def max_area(nums: list) -> int:
    if not nums:
        return 0

    left, right, area = 0, len(nums) - 1, 0

    while left < right:
        area = max(area, min(nums[left], nums[right]) * (right - left))

        if nums[right] > nums[left]:
            left += 1
        else:
            right -= 1

    return area


def marea(nums: list):
    if not nums:
        return 0
    left, right, area = 0, len(nums)-1, 0
    while left < right:
        area = max(area, min(nums[left], nums[right]) * (right - left))
        if nums[left] < nums[right]:
            left += 1
        else:
            right -= 1
    return area


def remove_duplicates(nums: list):
    if not nums:
        return

    left = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[left] = nums[i]
            left += 1

    return left


def move_zero(nums: list):
    if not nums:
        return

    left = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[left] = nums[i]
            left += 1

    while left < len(nums):
        nums[left] = 0
        left += 1


def remove_target(nums: list, target: int):
    if not nums:
        return

    left = 0
    for i in range(len(nums)):
        if nums[i] != target:
            nums[left] = nums[i]
            left += 1

    return left


def majority_elem(nums: list):
    if not nums:
        return

    from collections import defaultdict
    exists = defaultdict(int)
    for num in nums:
        exists[num] += 1

        if exists[num] > len(nums) // 2:
            return num


def disappear_elem(nums: list):
    if not nums:
        return []

    for i in range(len(nums)):
        idx = abs(nums[i]) - 1
        nums[idx] = -abs(nums[idx])

    return [idx + 1 for idx, val in enumerate(nums) if val > 0]


def word_distance(word1: str, word2: str) -> int:
    """ 单词的编辑距离
        dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1 if word1[i] != word2[j]
        dp[i][j] = dp[i-1][j-1] if word1[i] == word2[j]
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = i

    for j in range(1, n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[-1][-1]


class LongestPalindrome:
    """ 最长回文串,中间开始，左右两边字符相等
    """
    def __init__(self):
        self.max_len = 0
        self.start_idx = 0

    def longest_palindrome(self, s):
        if not s:
            return ""

        if len(s) < 2:
            return s

        for idx in range(len(s)):
            self._process(idx, idx, s)
            self._process(idx, idx+1, s)

        return s[self.start_idx: self.start_idx+self.max_len]

    def _process(self, left, right, s):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        if self.max_len < right - left - 1:
            self.max_len = right - left - 1
            self.start_idx = left + 1


def longest_parentheses_stack(s: str) -> int:
    """ 求成对括号的长度
    """
    if not s:
        return 0
    stack = []
    ans = 0
    for i in s:
        if i == "(":
            stack.append(i)
        if i == ")":
            if stack and stack[-1] == "(":
                stack.pop(-1)
                ans += 2
    return ans


def max_sub_array_sum(nums: list) -> int:
    if not nums:
        return 0

    tmp, ans = nums[0], nums[0]
    for num in nums[1:]:
        tmp = max(num, num + tmp)
        ans = max(ans, tmp)

    return ans


def min_path_sum(grid: list) -> int:
    if not grid:
        return -1

    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[-1][-1]


def climb_stirs(n: int) -> int:
    if n <= 2:
        return n

    s1, s2 = 1, 2
    for i in range(3, n+1):
        s1, s2 = s2, s1 + s2

    return s2


def min_triangle_path_sum(tri: list) -> int:
    if not tri:
        return 0

    m, n = len(tri), len(tri[-1])
    dp = [[0] * n for _ in range(m)]

    for j in range(n):
        dp[-1][j] = tri[-1][j]

    for i in range(m-2, -1, -1):
        for j in range(len(tri[i])):
            dp[i][j] = min(dp[i+1][j], dp[i+1][j+1]) + tri[i][j]

    return dp[0][0]


def word_break(s, word_list):
    if not word_list:
        return not s
    dp = [False for _ in range(len(s) + 1)]
    dp[0] = True

    for i in range(1, len(s) + 1):
        for j in range(i)[::-1]:
            if dp[j] and s[j:i] in word_list:
                dp[i] = True
                break
    return dp[-1]


def calculate_min_hp(dungeon):
    """ dp[i][j] 走到ｉ,j处至少还有多少血dp[i][j]滴血
        dp[i][j] =
    :return:
    """
    row, col = len(dungeon), len(dungeon[0])
    dp = [[0] * col for _ in range(row)]

    dp[-1][-1] = 1 if dungeon[-1][-1] > 0 else abs(dungeon[-1][-1]) + 1

    for i in range(col-2, -1, -1):
        dp[-1][i] = max(1, dp[-1][i+1] - dungeon[-1][i])

    for j in range(row-2, -1, -1):
        dp[j][-1] = max(1, dp[j+1][-1] - dungeon[j][-1])

    for i in range(row-2, -1, -1):
        for j in range(col-2, -1, -1):
            dp[i][j] = max(1, min(dp[i+1][j], dp[i][j-1]) - dungeon[i][j])

    return dp[0][0]


def rob_house(nums):
    """ dp[i] = max(dp[i-1], dp[i-2] + nums[i]) if i > 2
    """
    if not nums:
        return -1

    if len(nums) <= 2:
        return max(nums)

    dp = [0 for _ in range(len(nums))]
    dp[0] = nums[0]
    dp[1] = max(nums[:2])
    for j in range(2, len(nums)):
        dp[j] = max(dp[j-1], dp[j-2] + nums[j])

    return dp[-1]


def rob_house_cycle(nums):
    """ max(nums, nums[1:] + nums[:1])
    """
    return max(rob_house(nums), rob_house(nums[1:] + nums[:1]))


def num_squares(n: int):
    """  dp[i] = min(dp[i], dp[i-j*j] + 1)
    """
    if n <= 0:
        return 0
    dp = [float("inf") for _ in range(n+1)]
    for i in range(n+1):
        j = 1
        while i-j*j >= 0:
            dp[i] = min(dp[i], dp[i-j*j] + 1)
            j += 1
    return dp[-1]


def length_of_lis(nums: list) -> int:
    """　递增子序列,　dp存储以i结尾时递增子串的长度　
    """
    if not nums:
        return 0

    # nums从大到小排序时返回最后一个
    dp = [1 for _ in range(len(nums))]
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def max_profit_k(prices: list) -> int:
    """ 无限制次数操作股票,当天买入不能当天卖出
    dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + price[i])
    dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - price[i])
    """
    if not prices:
        return 0

    dp_i_0, dp_i_1 = 0, float("-inf")
    for i in range(len(prices)):
        tmp = dp_i_0
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
        dp_i_1 = max(dp_i_1, tmp - prices[i])

    return dp_i_0


def max_profit_1(prices: list) -> int:
    """ 操作一次股票
    dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1] + prices[i])
    dp[i][0][1] = max(dp[i-1][0][1], dp[i-1][0][0] - price[i])
    """
    if not prices:
        return -1

    dp_0, dp_1 = 0, float("-inf")
    for i in range(len(prices)):
        dp_0 = max(dp_0, dp_1 + prices[i])
        dp_1 = max(dp_1, -prices[i])
    return dp_0


def show_memory(msg):
    pid = os.getpid()
    p = psutil.Process(pid)

    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{msg} memory used: {memory} MB")


if __name__ == '__main__':
    # s = "(())()))((("
    # print(longest_parentheses_stack(s))

    # a, b = "abc", "adecc"
    # ans = word_distance(a, b)
    # print(ans)

    ans = LongestPalindrome().longest_palindrome("ABCCBD")
    print(ans)

    a = RotateStr("abcdef", 1, 5)
    a.rotate()
    print(a.s)


    # import pymongo
    # conn = pymongo.MongoClient("localhost", 27017)
    # col = conn["ent_link"]["t1"]
    #
    # col.find_one_and_update()




