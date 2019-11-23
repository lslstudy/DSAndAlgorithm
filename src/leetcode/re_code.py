# -*- coding: utf-8 -*-

""" useful code
"""

import sys


NUM_LETTER = {
    "2": ["a", "b", "c"],
    "3": ["d", "e", "f"],
    "4": ["g", "h", "i"],
    "5": ["j", "k", "l"],
    "6": ["m", "n", "o"],
    "7": ["p", "q", "r", "s"],
    "8": ["t", "u", "v"],
    "9": ["w", "x", "y", "z"]}


class LinkNode:

    def __init__(self, val=None):
        self.val = val
        self.next = None


def two_sum(seq: list, target: int) -> list:
    if not seq:
        return [-1, -1]
    exists = dict()
    for idx, elem in enumerate(seq):
        if target - elem in exists:
            return [exists[target - elem], idx]
        exists[elem] = idx
    return [-1, -1]


def two_link_sum(l1: LinkNode, l2: LinkNode):
    if not l1 or not l2:
        return l1 or l2
    tmp = 0
    phead = pnode = LinkNode(0)
    while tmp or l1 or l2:
        if l1:
            tmp += l1.val
        if l2:
            tmp += l2.val
        pnode.next = LinkNode(tmp % 10)
        pnode = pnode.next
        tmp //= 10
    return phead.next


def longest_sub_str(strs: str) -> int:
    if not strs:
        return 0
    left, answer = 0, 0
    exists = dict()
    for idx, char in enumerate(strs):
        if char not in exists or exists[char] < left:
            answer = max(answer, idx - left + 1)
        else:
            left = exists[char] + 1
        exists[char] = idx
    return answer


def longest_palindrome(strs: str) -> str:
    if not strs:
        return ""
    start, end = 0, 0
    for i in range(0, len(strs)):
        len1 = expand_around_center(strs, i, i)
        len2 = expand_around_center(strs, i, i + 1)
        length = max(len1, len2)
        if length > end - start:
            start = i - (length - 1) // 2
            end = i + length // 2
    return strs[start: end+1]


def expand_around_center(strs: str, left, right):
    """ 中心扩散法：对比中心左右两边的字符相等
    """
    while left >= 0 and right < len(strs) and strs[left] == strs[right]:
        left -= 1
        right += 1
    # 跳出循环时下标在回文串的左右两边
    return right - left - 1


def max_area(seq: list) -> int:
    """ 盛水最多的容器:左右两根指针表示边长，那根指针段短，那根指针移动 O(N)
    """
    if not seq:
        return 0
    answer, left, right = 0, 0, len(seq)
    while left < right:
        answer = max(answer, min(seq[left], seq[right]) * (right - left))
        if seq[left] < seq[right]:
            left += 1
        else:
            right -= 1
    return answer


def longest_common_prefix(strs: str) -> str:
    """ 求序列里面字符串的最长前缀 O(N^2)
    """
    if not strs:
        return ""
    answer = ""
    for j in range(0, len(strs[0])):
        char = strs[0][j]
        for i in range(0, len(strs)):
            # 字符不相同或者没有该字符直接返回
            if char != strs[i][j] or j > len(strs[i]):
                return answer
        answer += char
    return answer


def three_sum(seq: list) -> list:
    """ 列出所有三个数相加为0的数组，且不能有重复  O(N^2)
    """
    if not seq:
        return []
    # 排序操作很重要，确保从小到达的顺序  n*log(n)
    seq.sort()

    answer = list()
    # iter
    for i, elem in enumerate(seq):   # n(n)
        # 减枝优化：元素大于0时直接返回现有的结果或者空
        if elem >= 0:
            return answer if answer else []
        new_seq = seq[i+1:]
        left, right = 0, len(new_seq) - 1
        while left < right:
            # 满足条件时记得改变游标
            if new_seq[left] + new_seq[right] == -elem:
                tmp = [elem, new_seq[left], new_seq[right]]
                if tmp not in answer:
                    answer.append(tmp)
                left += 1
                right -= 1
            elif new_seq[left] + new_seq[right] < -elem:
                left += 1
            else:
                right -= 1
    return answer


def closest_three_sum(nums: list, target: int) -> list:
    """ 最接近的三数和
    """
    if not nums:
        return []

    nums.sort()
    diff = sys.maxsize       # 用来记录迭代时最接近的差
    closest = sys.maxsize    # 记录最接近的结果和
    for i, elem in enumerate(nums):
        new_nums = nums[i+1:]
        left, right = 0, len(new_nums) - 1
        while left < right:
            sums = new_nums[left] + new_nums[right] + elem
            new_diff = abs(sums - target)
            if diff > new_diff:
                diff = new_diff
                closest = sums
            if sums < target:
                left += 1
            else:
                right -= 1
    return closest


def letter_combinations(number: int) -> list:
    """
    """
    if not number:
        return []
    answer = list()

    def _helper(num: str, tmp: str):
        if not num:
            answer.append(tmp)
            return
        char, new_num = num[:1], num[1:]
        for c in NUM_LETTER[char]:
            _helper(new_num, tmp + c)
    _helper(str(number), "")
    return answer


def drop_from_last_nth(head: LinkNode, n: int) -> LinkNode:
    """ 删除倒数第N个节点： O(N)
            快慢指针：快指针领先慢指针n个位置，快指针为空是，操作慢指针
    """
    if not head:
        return head
    slow = fast = head
    while n:
        fast = fast.next
        if not fast:
            return head
        n -= 1
    while fast.next:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head


def valid_parentheses(seq: list):
    """ 验证括号的合法性
    """
    parentheses_dict = {
        "}": "{",
        "]": "[",
        ")": "("}
    stack = []
    for char in seq:
        # 追加左半边括号
        if char not in parentheses_dict:
            stack.append(char)
        else:
            # 右半边括号与栈顶元素匹配时弹出栈顶元素
            if not stack:
                return False
            if stack[-1] == parentheses_dict[char]:
                stack.pop()
    return False if stack else True


def merge_sorted_link(l1: LinkNode, l2: LinkNode) -> LinkNode:
    """ 合并两个有序链表
    """
    if not l1 or not l2:
        return l1 or l2
    pnode = phead = LinkNode(0)
    while l1 or l2:
        if l1.val < l2.val:
            pnode.next = l1
            l1 = l1.next
        else:
            pnode.next = l2
            l2 = l2.next
        pnode = pnode.next
    pnode.next = l1 or l2
    return phead.next


def generate_parentheses(num: int) -> list:
    """ 括号生成
    """
    if not num:
        return []
    answer = list()

    def _helper(left, right, tmp):
        if left == right == num:
            answer.append(tmp)
            return
        if left < num:
            _helper(left+1, right, tmp+"(")
        # 右边括号是生成有限制，必须有左边括号并且数量小于总共生成的对数
        if left > right and right < num:
            _helper(left, right+1, tmp+")")
    _helper(0, 0, "")
    return answer


def merge_k_sorted_link(links: list) -> LinkNode:
    """ 1.需要合并次数最少：分治法
        2.暴力法：相邻两个有序链表两两合并
        3.最小堆：所有链表的头元素放到最小堆，取出最小堆的堆顶元素，再将堆顶元素的下一个元素放入堆中，重复次过程，直到没有元素为止。
    """

    pass


if __name__ == '__main__':
    # a = "aacbceabcd"
    # ans = longest_sub_str(a)
    # print(ans)
    # a = "abcba"
    # ans = longest_palindrome(a)
    # print(ans)

    # nums = [-1, 0, 1, 2, -1, -4]
    # print(three_sum(nums))

    # print(letter_combinations(number=23))

    print(generate_parentheses(num=3))
