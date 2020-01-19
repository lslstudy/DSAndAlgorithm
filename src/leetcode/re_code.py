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
            l1 = l1.next
        if l2:
            tmp += l2.val
            l2 = l2.next
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
    answer, left, right = 0, 0, len(seq) + 1
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


def merge_k_sorted_link(links: list):
    """ 1.需要合并次数最少：分治法
        2.暴力法：相邻两个有序链表两两合并
        3.最小堆：所有链表的头元素放到最小堆，取出最小堆的堆顶元素，再将堆顶元素的下一个元素放入堆中，重复次过程，直到没有元素为止。
    """
    if not links:
        return links
    n = len(links)
    while n > 1:
        step = (n+1) >> 1  # step
        # 迭代处理：i位置和i+mid位置两两合并
        for i in range(n >> 1):
            links[i] = merge_sorted_link(links[i], links[i+step])
        # 缩减迭代条件，n变为原来的一半。
        n = step
    return links[0]


def swap_pairs_rec(head: LinkNode):
    if not head or not head.next:
        return head
    # 暂存第二个节点的指向地址空间
    second = head.next
    # 改变后的第二个节点指向下递归的返回值
    head.next = swap_pairs_rec(head.next.next)
    # 第二个几点指向第一个节点
    second.next = head
    # 交换后的头结点
    return second


def reverse_k_group(head: LinkNode, k: int):
    """ 思路：两个函数实现：一个用于按照k个元素切分，一个用于把k个元素翻转
    """
    if not head or k == 1:
        return head
    dummy = LinkNode(-1)
    pre, curr = dummy, head
    dummy.next = head
    i = 0
    while curr:
        i += 1
        # k个元素进行一次翻转
        if i % k == 0:
            pre = reverse_one_group(pre, curr.next)
            curr = pre.next
        else:
            curr = curr.next
    return dummy.next


def reverse_one_group(start: LinkNode, end: LinkNode):
    last = start.next
    curr = last.next
    while curr != end:
        last.next = curr.next
        curr.next = start.next
        start.next = curr
        curr = last.next
    return last


def remove_duplicates(seq: list) -> tuple:
    """ 删除排序数组中额重复元素，快慢指针，都指向第一个元素，当后一个元素和前一个元素不同时，两指针都向前移动，两元素相同时，快指针移动
    """
    if not seq:
        return 0
    fast, slow = 0, 0
    while fast < len(seq):
        if seq[fast] == seq[slow]:
            fast += 1
        else:
            # 注意：要先跳过不相同元素
            slow += 1
            seq[slow] = seq[fast]
            fast += 1
    return slow + 1, seq[:slow+1]


def remove_element(nums: list, target: int):
    """ 删除数组追踪等于目标值的元素，不相等移动元素，相等就跳过
    """
    if not nums:
        return 0
    res = 0
    for i in range(len(nums)):
        if nums[i] != target:
            nums[res] = nums[i]
            res += 1
    return res


def str_str(parent: str, child: str) -> int:
    """ 自字符串在父字符串中的起始下标
    """
    if not child:
        return 0
    m, n = len(parent), len(child)
    if m < n:
        return -1
    for i in range(m - n):
        j = 0
        for j in range(n):
            if parent[i+j] != child[j]:
                break
        # 下标和长度差值为1
        j += 1
        if j == n:
            return i
    return -1


def longest_valid_parentheses(strs: str) -> int:
    """这里我们还是借助栈来求解，需要定义个start变量来记录合法括号串的起始位置，我们遍历字符串，如果遇到左括号，则将当前下标压入栈，如果遇到右括号，且当前栈为空，则将下一个坐标位置记录到start，如果栈不为空，则将栈顶元素取出，此时若栈为空，则更新结果和i - start + 1中的较大值，否则更新结果和i - 栈顶元素中的较大值
    """
    if not strs:
        return 0
    stack = list()
    start, res = 0, 0
    for i in range(len(strs)):
        if strs[i] == "(":
            stack.append(strs[i])
        else:
            if not stack:
                start = i + 1
            else:
                stack.pop()
                index = rindex(stack, stack[-1])
                res = max(res, i-index) if stack else max(res, i - start + 1)
    return res


def rindex(seq: list, target) -> int:
    for i in range(len(seq)-1, -1, -1):
        if seq[i] == target:
            return i
    return -1


def search_route(nums: list, target: int):
    """  如果中间的数小于最右边的数，则右半段是有序的，若中间数大于最右边数，则左半段是有序的，我们只要在有序的半段里用首尾两个数组来判断目标值是否在这一区域内，这样就可以确定保留哪半边了
    """
    if not nums:
        return -1
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left + right) >> 1
        if nums[mid] == target:
            return mid
        # 右边有序
        elif nums[mid] < nums[right]:
            if nums[mid] < target and nums[right] >= target:
                left = mid + 1
            else:
                right = mid - 1
        # 左边有序
        else:
            if nums[left] <= target and nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
    return -1


def search_insert(nums: list, target: int) -> int:
    for i in range(len(nums)):
        if nums[i] >= target:
            return i
    return len(nums)


def first_missing_positive(nums: list) -> int:
    """ 时间复杂度应为O(n)，并且只能使用常数级别的空间
    给定一个未排序的整数数组，找出其中没有出现的最小的正整数。
    思路是把1放在数组第一个位置nums[0]，2放在第二个位置nums[1]，即需要把nums[i]放在nums[nums[i] - 1]上，那么我们遍历整个数组，如果nums[i] != i + 1, 而nums[i]为整数且不大于n，另外nums[i]不等于nums[nums[i] - 1]的话，我们将两者位置调换，如果不满足上述条件直接跳过，最后我们再遍历一遍数组，如果对应位置上的数不正确则返回正确的数
    """
    size = len(nums)
    for i in range(size):
        while 0 < nums[i] <= size and nums[nums[i] - 1] != nums[i]:
            nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
    for i in range(size):
        if nums[i] != i + 1:
            return i + 1
    return size + 1


def max_sub_sum(nums: list) -> int:
    """ 有正数和负数，正数加负数永远小于正数与正数相加
    """
    res, tmp_sum = -sys.maxsize, 0
    for num in nums:
        tmp_sum = max(tmp_sum + num, num)
        res = max(tmp_sum, res)
    return res


def min_sum(nums: list) -> int:
    if not nums:
        return 0
    row, col = len(nums), len(nums[-1])
    dp = [[0] * col for _ in range(row)]
    dp[-1] = nums[-1]

    for i in range(row-2, -1, -1):
        for j in range(i + 1):
            dp[i][j] = min(dp[i+1][j], dp[i+1][j+1]) + nums[i][j]
    return dp[0][0]


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

    # print(generate_parentheses(num=3))
    # print(remove_duplicates(seq=[0, 0, 1, 1, 1, 2, 2]))
    # print(remove_element([0, 1, 2, 2, 3, 0, 4, 2], 2))
    # print(str_str(parent="hello", child="ll"))

    print(rindex(seq=[1, 2, 3, 4, 5], target=4))
    # print(combination_sum(nums=[2, 3, 5], target=8))
