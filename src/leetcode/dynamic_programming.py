# -*- coding: utf-8 -*-

"""
"""


def min_distances(word1, word2):
    """
    """
    m, n = len(word1), len(word2)

    if m * n == 0:
        return m + n

    dp = [[0] * (m+1) for _ in range(n+1)]  # (n+1) * (m+1)

    # word2 列数据变为空字符时所需的步数
    for i in range(1, n+1):
        dp[i][0] = i

    # word1　行数据变为空字符串时所需的步数
    for j in range(1, m+1):
        dp[0][j] = j

    for i in range(1, n+1):
        for j in range(1, m+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1], dp[i-1][j]) + 1

    return dp[m][n]


class LongestPalindrome:

    def __init__(self):
        self.max_len = 0
        self.start_idx = 0

    def longest_palindrome(self, s):
        if not s:
            return ""

        if len(s) < 2:
            return s

        for idx in range(len(s)):
            self._support(idx, idx, s)
            self._support(idx, idx+1, s)

        return s[self.start_idx: self.start_idx + self.max_len]

    def _support(self, left, right, s):
        while left >= 0 and right <= len(s) - 1 and s[left] == s[right]:
            left -= 1
            right += 1
        if self.max_len < right - left - 1:
            self.max_len = right - left - 1
            self.start_idx = left + 1


def longest_valid_parentheses(s: str):
    """ case: (())
        dp[i] = dp[i-2] + 2 ==> s[i] == ")" && s[i-1] == "("
        dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-1]  ==> s[i] == ")" && s[i-1] == ")"
    """
    if not s:
        return 0

    dp = [0 for _ in range(len(s))]
    max_len = 0

    for i in range(1, len(s)):
        if s[i] == ")":
            if s[i-1] == "(":
                dp[i] = dp[i-2] + 2
            elif i - dp[i-1] > 0 and s[i-dp[i-1]-1] == ")":
                dp[i] = dp[i-1] + (dp[i-dp[i-1]-2] if (i-dp[i-1]) >= 2 else 0) + 2
            max_len = max_len(max_len, dp[i])

    return max_len


def max_sub_array(nums: list):
    if not nums:
        return 0

    tmp, max_val = nums[0], nums[0]

    # 负数相加整体变小
    for i in range(1, len(nums)):
        tmp = max(tmp + nums[i], nums[i])
        max_val = max(max_val, tmp)

    return max_val


def min_path_sum(grid: list) -> int:
    """ dp(i, j) 表示在点 从(0, 0) 到(i, j)处的最小路径和
    """
    if not grid:
        return -1

    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]  # m *n

    dp[0][0] = grid[0][0]

    # row = m
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    # col = n
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[-1][-1]


def climb_stairs(n: int) -> int:
    """ dp[i] = dp[i-1] + dp[i-2]
    """
    if n < 3:
        return n

    start_1, start_2 = 1, 2

    for i in range(3, n+1):
        start_1, start_2 = start_2, start_1 + start_2

    return start_2


def min_triangle_path_sum(triangle: list) -> int:
    """ 从底到顶
            dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])
    """
    if not triangle:
        return -1

    n = len(triangle)
    dp = [[0] * len(triangle[-1]) for _ in range(n)]
    for i in range(len(triangle[-1])):
        dp[-1][i] = triangle[-1][i]

    for i in range(n-2, -1, -1):
        for j in range(len(triangle[i])):
            dp[i][j] = triangle[i][j] + min(dp[i+1][j], dp[i+1][j+1])

    return dp[0][0]


def work_break(s: str, word_list: list) -> bool:
    """  "applepenapple", wordDict = ["apple", "pen"] => True
    设dp(i)表示以第i个字符结尾(不包含第i个字符)的子字符串是否能拆分成字典中的单词
    """
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


def word_break_list(s, words) -> list:
    # TODO
    if not s:
        return []

    pass


def calculate_min_start_hp(dungeon: list) -> int:
    """ 计算从左上角走到右下角中骑士所需的最少生命值, 从左下角开始计算
    """
    row, col = len(dungeon), len(dungeon[0])

    dp = [[0] * col for _ in range(row)]
    # 走到最后一格最少为一滴血
    dp[-1][-1] = 1 if dungeon[-1][-1] > 0 else abs(dungeon[-1][-1]) + 1

    for i in range(col-2, -1, -1):
        dp[-1][i] = max(1, dp[-1][i+1] - dungeon[-1][i])

    for j in range(row-2, -1, -1):
        dp[j][-1] = max(1, dp[j+1][-1] - dungeon[j][-1])

    for i in range(row-2, -1, -1):
        for j in range(col-2, -1, -1):
            dp[i][j] = max(1, min(dp[i+1][j], dp[i][j-1]) - dungeon[i][j])

    return dp[0][0]


def rob_house_max_profit(nums: list) -> int:
    """ 抢劫房屋是利益最大化，注意：不能抢劫相邻的两座房屋 198
    """
    if not nums:
        return -1

    if len(nums) <= 2:
        return max(nums)

    pre, max_profit = nums[0], max(nums[0], nums[1])

    for i in range(2, len(nums)):
        tmp = max_profit
        max_profit, pre = max(pre + nums[i], max_profit), max_profit
        pre = tmp

    return max_profit


def rob_house_max_profit_2(nums: list) -> int:
    """ max(0~len-1, 1~len)
    """
    return max(rob_house_max_profit(nums), rob_house_max_profit(nums[1:] + nums[:1]))


def num_squares(n: int):
    """ 给出整数找到组成该数所需的最小的平方数个数 279
        dp[i] = min(dp[i], dp[i-j*j] + 1) j < sqrt(i)
    """
    if n <= 0:
        return 0

    dp = [float("inf") for _ in range(n+1)]

    for i in range(n+1):
        j = 1

        # 问题转换为去掉一个数的平方之后的数是由几个平方数构成
        while i - j*j >= 0:
            dp[i] = min(dp[i], dp[i-j*j] + 1)
            j += 1

    return dp[n]


def length_of_lis(nums: list) -> int:
    """  300 最长递增子序列　[10,9,2,5,3,7,101,18]　＝》[2, 3, 7, 101] => 4
         dp[i] 表示以 nums[i] 为结尾的最长递增子串的长度
    """
    if not nums:
        return 0

    dp = [1 for _ in range(len(nums))]

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def max_profit_k(prices: list) -> int:
    """ 309 操作买卖股票，赚取最大的利益, 不限制操作次数,买进当天不能卖出

        dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
        dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])

        if k ==> +inf k == k+1
    """
    if not prices:
        return 0

    # 第ｉ天手里没有股票和有股票最大利益
    dp_i_0, dp_i_1 = 0, float("-inf")

    for i in range(len(prices)):
        tmp = dp_i_0
        dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
        dp_i_1 = max(dp_i_1, tmp - prices[i])

    return dp_i_0


def max_profit_1(prices: list) -> int:
    """ 买卖股票，只有一次操作机会  i, k, l:  第i天，操作次数，手里是否持股, 操作次数ｋ只和买入有关
        dp[i][1][0] = max(dp[i-1][1][0], dp[i-1][1][1] + prices[i])
        dp[i][0][1] = max(dp[i-1][0][1], dp[i-1][0][0] - prices[i])
        dp[i-1][0][0] == None
    """
    if not prices:
        return -1

    dp_0, dp_1 = 0, float("-inf")

    for i in range(len(prices)):
        dp_0 = max(dp_0, dp_1 + prices[i])
        dp_1 = max(dp_1, -prices[i])

    return dp_0


def coin_change(coins: list, amount: int) -> int:
    """ 322 用给予的coins来进行找零对amount进行找零，数量最少，不能满足要求返回-1
    """
    dp = [amount + 1 for i in range(amount + 1)]

    dp[0] = 0

    for i in range(1, amount+1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i-coin] + 1)

    return -1 if dp[amount] > amount else dp[amount]


def rob_tree(root):
    """ 注意，不能抢劫相邻的两层树
    """

    if not root:
        return 0

    # 不抢劫根节点
    not_rot = 0
    if root.left:
        not_rot += rob_tree(root.left)

    if root.right:
        not_rot += rob_tree(root.right)

    # 抢劫根节点
    rob_root = root.val

    if root.left:
        if root.left.left:
            rob_root += rob_tree(root.left.left)
        if root.left.right:
            rob_root += rob_tree(root.left.right)

    if root.right:
        if root.right.left:
            rob_root += rob_tree(root.right.left)
        if root.right.right:
            rob_root += rob_tree(root.right.right)

    return max(not_rot, rob_root)


def can_partition(nums: list) -> bool:
    """
    """
    pass


def print_strings(strs: str):
    """
    """
    left, stack, curr = 0, [], ""
    while left < len(strs):
        if strs[left] == "(" or strs[left].isalpha():
            stack.append(strs[left])
            left += 1
        elif strs[left] == ")":
            tmp = []
            while stack and left <= len(strs):
                c = stack.pop()
                if c == "(":
                    break
                else:
                    tmp.append(c)
            left += 1

            times = []
            for elem in strs[left:]:
                if elem.isdigit():
                    times.append(elem)
                else:
                    break
            time = int("".join(times))
            left += len(times)
            stack.extend(tmp[::-1] * time)
        else:
            time = []
            a = stack.pop()
            for val in strs[left:]:
                if val.isdigit():
                    time.append(val)
                else:
                    break
            left += len(time)
            stack.append(a * int("".join(time)))

    return "".join(stack)


if __name__ == '__main__':
    # p = [2, 0, 4, 6, 2, 4]
    # ans = max_profit_1(p)
    # print(ans)
    #
    # ans = max_profit_k(p)
    # print(ans)

    a = print_strings("A11B")
    print(a)

    b = print_strings("(AA)2A")
    print(b)
    print(print_strings("((A2B)2)2G"))
    print(print_strings("(YUANFUDAO)2JIAYOU"))
    print(print_strings("A2BC4D2"))


