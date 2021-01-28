# -*- coding: utf-8 -*-

""" string relations algorithm
"""

from typing import List


def is_deformation(s1: str, s2: str) -> bool:
    """ 变形词： 出现的字符一样且每个字符的数量一样
    """
    if not s1 or not s2 or len(s1) != len(s2):
        return False
    bit_arr = [0] * 256
    for char in s1:
        bit_arr[ord(char)] += 1
    for char in s2:
        if bit_arr[ord(char)] == 0:
            return False
        else:
            bit_arr[ord(char)] -= 1
    return True


def sub_num_sum(s: str) -> int:
    """ 字符串中所有数字字符的和
    """
    if not s:
        return 0
    res, num, postag = 0, 0, True
    for idx, char in enumerate(s):
        if not char.isdigit():
            res += num
            num = 0
            if char == "-":
                if idx - 1 > -1 and s[idx-1] == "-":
                    postag = not postag
                else:
                    postag = False
            else:
                postag = True
        else:
            num = num * 10 + (int(char) if postag else -int(char))
    res += num
    return res


def is_rotation(a: str, b: str) -> bool:
    """ 判断ｂ是否是a的旋转词
    """
    if not a or not b or len(a) != len(b):
        return False
    return b in a + a


def cstrs(s: str) -> str:
    # todo
    if not s:
        return ""
    ans, cc = [], 1
    for idx, char in enumerate(s):
        if idx - 1 > -1 and s[idx-1] == char:
            cc += 1
        if ans and str(ans[-1]).isdigit():
            ans[-1] = cc


def replace_space(strs: List[str]) -> str:
    if not strs:
        return ""

    space_num, char_num = 0, 0
    for char in strs:
        if char_num != 0:
            char_num += 1
        if char == " ":
            space_num += 1

    j = char_num + 2 * space_num - 1
    for i in range(char_num-1, -1, -1):
        if strs[i] != " ":
            strs[j] = strs[i]
            j -= 1
        else:
            strs[j] = "0"
            strs[j-1] = "2"
            strs[j-2] = "%"
            j -= 3


def merge_sorted_arrays(arr1: List[int], arr2: List[int]) -> None:
    if not arr1 or not arr2:
        return

    num, total = 0, len(arr1)
    for elem in arr1:
        if elem != 0:
            num += 1

    aidx, bidx = num, len(arr2)
    for i in range(total-1, -1, -1):
        if arr1[aidx-1] >= arr2[bidx-1]:
            arr1[i] = arr1[aidx-1]
            arr1[aidx-1] = 0
            aidx -= 1
        else:
            arr1[i] = arr2[bidx-1]
            arr2[bidx-1] = 0
            bidx -= 1



if __name__ == '__main__':
    s1, s2 = "abcdd", "acdec"
    assert is_deformation(s1, s2) is False

    s = "AE--902B8"
    # print(sub_num_sum(s))

    a = [2, 4, 5, 7, 0, 0, 0]
    b = [1, 3, 6]
    merge_sorted_arrays(a, b)
    print(a)
