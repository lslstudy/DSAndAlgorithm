# -*- coding: utf-8 -*-

""" high frequency interview code
"""


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def binary_search(arr: list, elem: int) -> int:
    """  Given an sorted array and find elem using binary search method
    :return:
    """
    if not arr:
        return -1
    if elem > arr[-1] or elem < arr[0]:
        return -1
    left, right = 0, len(arr) - 1
    mid = (left + right) >> 1
    while left < right:
        if arr[mid] == elem:
            return mid
        elif arr[mid] > elem:
            right = mid - 1
        else:
            left = mid + 1
        mid = (left + right) >> 1
    return -1


def get_largest_kth(arr: list, k: int) -> list:
    """ Given array and find largest K elements
    :return:
    """
    if not arr:
        return []
    if len(arr) <= k:
        return arr
    answer = arr[0: k]
    answer.sort(reverse=True)
    tmp = arr[k:]
    for item in tmp:
        if item > answer[-1]:
            answer[-1] = item
            answer.sort(reverse=True)
    return answer


def print_tuple(arr: list) -> list:
    """ Given an shuffle array, and print tuple like:
            [3, 6, 2, 7, 1, 9] ======> [(3, 6), (2, 7), (6, 7),
                                        (1, 9), (7, 9), (9, -1)]
    :return:
    """
    if not arr:
        return []
    stack, answer = [arr[0]], []
    for item in arr[1:]:
        while stack and stack[-1] < item:
            answer.append((stack.pop(), item))
        stack.append(item)
    while stack:
        answer.append((stack.pop(), -1))
    return answer


def quick_sort_with_rec(arr: list) -> list:
    """ QuickSort Algorithm
    :return:
    """
    if not arr:
        return []
    if len(arr) <= 1:
        return arr
    less, middle, greater = [], arr.pop(), []
    for item in arr:
        if item < middle:
            less.append(item)
        else:
            greater.append(item)
    return quick_sort_with_rec(less) + [middle] + \
           quick_sort_with_rec(greater)


def pre__order_tree(root: TreeNode) -> list:
    """ pre_order binary tree with recursive
    :return:
    """
    if not root:
        return []
    answer = []

    def helper(node: TreeNode):
        if not node:
            return
        answer.append(node.val)
        helper(node.left)
        helper(node.right)
    helper(node=root)
    return answer


def in_order_tree(root: TreeNode) -> list:
    """ in_order binary tree with recursive
    :return:
    """
    if not root:
        return []
    answer = []

    def helper(node: TreeNode):
        if not node:
            return
        helper(node.left)
        answer.append(node.val)
        helper(node.right)
    helper(node=root)
    return answer


def post_order_tree(root: TreeNode) -> list:
    """ post order binary tree with recursive
    :return:
    """
    if not root:
        return []
    answer = []

    def helper(node: TreeNode):
        if not node:
            return
        helper(node.left)
        helper(node.right)
        answer.append(node.val)
    helper(node=root)
    return answer


def traversal_tree_by_level(root: TreeNode) -> list:
    """ traversal tree by level
        example:
                3           [ [3],
               / \     =>     [4, 5]]
              4   5
    :return:
    """
    if not root:
        return []
    answer = []
    curr, next_ = [root], []
    while curr:
        tmp = []
        for node in curr:
            tmp.append(node.val)
            if node.left:
                next_.append(node.left)
            if node.right:
                next_.append(node.right)
        answer.append(tmp)
        curr.clear()
        curr.extend(next_)
        next_.clear()
    return answer


def max_tree_depth(root: TreeNode) -> int:
    """ Given an tree and find max depth
    :return:
    """
    if not root:
        return 0
    if not root.left and root.right:
        return 1
    return 1 + max(max_tree_depth(root.left),
                   max_tree_depth(root.right))


def min_tree_depth(root: TreeNode) -> int:
    """ Given an tree and find min depth
    :return:
    """
    if not root:
        return 0
    if not root.left:
        return 1 + min_tree_depth(root.right)
    if not root.right:
        return 1 + min_tree_depth(root.left)
    return 1 + min(min_tree_depth(root.left),
                   min_tree_depth(root.right))


if __name__ == '__main__':
    # a = [2, 3, 5, 6, 7, 8, 10]
    # idx = binary_search(arr=a, elem=8)
    # print(a[idx])
    a = [3, 6, 2, 7, 1, 9]
    ans = get_largest_kth(arr=a, k=3)
    print(ans)
    ans = print_tuple(arr=a)
    print(ans)
    s1 = quick_sort_with_rec(arr=a)
    print(s1)

