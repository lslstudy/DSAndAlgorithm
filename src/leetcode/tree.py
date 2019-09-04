# -*- coding: utf-8 -*-

""" tree data structure and algorithm
"""


class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


def has_path_sum(root: TreeNode, target: int) -> bool:
    """ given a tree and find if node path sum equal target
    """
    if not root:
        return False
    if root.val == target and not root.left and not root.right:
        return True
    return has_path_sum(root.left, target-root.val) or \
           has_path_sum(root.right, target - root.val)


def has_path_sum2(root: TreeNode, target: int) -> list:
    """ given a tree and find root-to-left path and return as list
    """
    if not root:
        return []
    if not root.left and not root.right:
        return [] if root.val != target else [[root.val]]
    answer = []
    for i in has_path_sum2(root.left, target-root.val):
        i.insert(0, root.val)
        answer.append(i)
    for i in has_path_sum2(root.right, target-root.val):
        i.insert(0, root.val)
        answer.append(i)
    return answer


def traversal_tree(root: TreeNode) -> list:
    """ traversal tree by width
    """
    if not root:
        return []
    answer, queue = [], [root]
    while queue:
        tmp = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            tmp.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        answer.append(tmp)
    return answer


def max_tree_width(root: TreeNode) -> int:
    """ given an tree and find tree width
    """
    if not root:
        return 0
    if root and not root.left and not root.right:
        return 1
    width, queue = 0, [root]
    while queue:
        cur_width = len(queue)
        width = max(width, cur_width)
        for _ in range(cur_width):
            node = queue.pop(0)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return width


def mirror_tree(root: TreeNode):
    """ given a tree and  exchange left node and right node
    """
    if not root:
        return
    if not root.left and not root.right:
        return root
    root.left = mirror_tree(root.right)
    root.right = mirror_tree(root.left)
    return root


