# -*- coding: utf-8 -*-

"""
"""


def matrix_min_path_sum(m: list):
    """ TIME: O(M*N)   SPACE: O(M*N)
    """
    if not m:
        return 0
    row, col = len(m), len(m[0])
    dp = [[0] * col for _ in range(row)]
    dp[0][0] = m[0][0]
    for i in range(1, row):
        dp[i][0] = dp[i-1][0] + m[i][0]
    for j in range(1, col):
        dp[0][j] = dp[0][j-1] + m[0][j]
    for i in range(1, row):
        for j in range(1, col):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + m[i][j]
    return dp[-1][-1]


def coin_change(coins: list, amount: int) -> int:
    """ 换零钱问题
    """
    if not coins:
        return 0
    dp = [amount+1 for _ in range(amount+1)]
    dp[0] = 0
    for i in range(1, amount+1):
        for coin in coins:
            if i - coin < 0:
                continue
            dp[i] = min(dp[i], dp[i-coin] + 1)
    return -1 if dp[-1] == amount+1 else dp[-1]


def max_length_sub_seq(arr: list):
    """ 最长递增子序列 Time O(N^2), Space O(N)
    """
    if not arr:
        return 0
    dp = [0] * len(arr)
    for i in range(len(arr)):
        dp[i] = 1
        for j in range(i):
            if arr[i] > arr[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def max_sub_sum(arr: list):
    if not arr:
        return
    ans = arr[0]
    for i in range(1, len(arr)):
        tmp = max(0, arr[i] + arr[i-1])
        ans = max(ans, ans + tmp)
    return ans


def max_common_sub_seq(s1: str, s2: str) -> int:
    """ 公共子序列
    """
    if not s1 or not s2:
        return 0
    l1, l2 = len(s1), len(s2)
    dp = [[0] * l2 for _ in range(l1)]
    for i in range(1, l1):
        dp[i][0] = max(dp[i-1][0], 1 if s1[i] == s2[0] else 0)
    for j in range(1, l2):
        dp[0][j] = max(dp[0][j-1], 1 if s1[0] == s2[j] else 0)
    for i in range(1, l1):
        for j in range(1, l2):
            dp[i][j] = max(dp[i][j-1], dp[i-1][j])
            if s1[i] == s2[j]:
                dp[i][j] = max(dp[i][j], dp[i-1][j-1] + 1)
    return dp[-1][-1]


def max_common_sub_str(s1: str, s2: str) -> int:
    """ 公共子串长度
    """
    pass


def words_distance(s1: str, s2: str) -> int:
    """ 编辑距离，dp[m][n] = min(dp[m-1][n], dp[m][n-1], dp[m-1][n-1]) + 1
    """
    pass


def win_game(arr: list)->int:
    """
    """
    if not arr:
        return 0
    return max(f(arr, 0, len(arr)-1), s(arr, 0, len(arr)-1))


def f(arr, i, j):
    """ 先拿能够获取的分数
    """
    if i == j:
        return arr[i]
    return max(arr[i] + s(arr, i+1, j), arr[j] + s(arr, i, j-1))


def s(arr, i, j):
    """后拿能够获取的分数
    """
    if i == j:
        return 0
    return min(f(arr, i+1, j), f(arr, i, j-1))


def jump_game(arr: list) -> int:
    """  jump: 目前跳了多少步
         cur:  如果只跳jump步，最远能跳到的位置
         _next: 在多跳一步，能够到达的位置
    """
    if not arr:
        return 0
    jump, cur, _next = 0, 0, 0
    for i in range(len(arr)):
        if cur < i:
            jump += 1
            cur = _next
        _next = max(_next, arr[i] + 1)
    return jump


if __name__ == '__main__':
    a = [2, 1, 5, 3, 6, 4, 8, 9, 7]
    print(max_length_sub_seq(a))
    a = [1, 2, 100, 4]
    print(win_game(a))

