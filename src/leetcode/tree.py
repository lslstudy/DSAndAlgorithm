# -*- coding: utf-8 -*-

""" tree data structure and algorithm
"""


class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


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


def in_order_rec(root: TreeNode):
    if not root:
        return []

    ans = []

    def _helper(node):
        if not node:
            return
        _helper(node.left)
        ans.append(node.val)
        _helper(node.right)
    _helper(root)
    return ans


def in_order_iter(root: TreeNode):
    if not root:
        return
    stack, ans, p = [], [], root
    while p or stack:
        while p:
            stack.append(p)
            p = p.left
        tmp = stack.pop()
        ans.append(tmp.val)
        # 第一次到达
        p = tmp.right if tmp.right else None
    return ans


def generate_tree_nums(number: int) -> int:
    """ given an int number and return generate tree numbers
        dp[i] = dp[j] * dp[i-j-1]
    """
    dp = [0 for _ in range(number + 1)]
    dp[0] = 1  # 空树
    dp[1] = 1  # 只有根节点

    for i in range(2, number+1):
        for j in range(i):
            dp[i] += dp[j] * dp[i-j-1]
    return dp[-1]


def valid_is_bst(root: TreeNode, lower=float("-inf"), upper=float("inf")):
    """
    """
    if not root:
        return True
    node = root.val

    # 当层节点检查: 左节点小于根节点　右节点大于根节点
    if node <= lower or node >= upper:
        return False

    if not valid_is_bst(node.left, lower, node):
        return False

    if not valid_is_bst(node.right, node, upper):
        return False

    return True


def is_same_tree(root1: TreeNode, root2: TreeNode) -> bool:
    if not root1 and not root2:
        return True

    if not root1 or not root2:
        return False

    # 两棵树的根节点相同并且左右同时相同
    if root1.val == root2.val:
        left = is_same_tree(root1.left, root2.left)
        right = is_same_tree(root1.right, root2.right)
        return left and right
    else:
        return False


def is_symmetric(root: TreeNode):
    """ 是否是对称树
    """
    def _helper(left, right):
        if not left and not right:
            return True

        if not left or not right:
            return False

        return left.val == right.val and\
               _helper(left.left, right.left) and\
               _helper(left.right, right.right)
    return _helper(root, root)


def level_order(root: TreeNode) -> list:
    """ 树的层次遍历
    """
    if not root:
        return []
    res, queue = [], [root]

    while queue:
        tmp, next_level = [], []

        for node in queue:
            tmp.append(node.val)

            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        queue = next_level
        res.append(tmp)

    return res


def max_depth(root: TreeNode) -> int:
    """ 树的最大深度
    """
    if not root:
        return 0

    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    return max(left_depth, right_depth) + 1


def build_tree(preorder, inorder):
    """ 使用树的谦虚遍历和中序遍历重构树
    """
    if not preorder and not inorder:
        return

    root = TreeNode(preorder[0])
    mid = inorder.index[root.val]

    root.left = build_tree(preorder[1: mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:], inorder[mid+1:])

    return root


def sort_array2tree(array: list):
    """ 有序数组构件树
    """
    if not array:
        return

    left, right = 0, len(array) - 1
    mid = (left + right) // 2

    root = TreeNode(array[mid])
    root.left = sort_array2tree(array[:mid])
    root.right = sort_array2tree(array[mid+1:])

    return root


def is_balance_tree(root: TreeNode) -> bool:
    """ 是否是平衡树
    """
    if not root:
        return True

    left = max_depth(root.left)
    right = max_depth(root.right)

    if abs(left - right) > 1:
        return False
    else:
        return is_balance_tree(root.left) and is_balance_tree(root.right)


def min_depth(root: TreeNode):
    """ 树的最小深度
    """
    if not root:
        return 0

    if not root.left:
        return min_depth(root.right) + 1

    if not root.right:
        return min_depth(root.left) + 1

    return min(min_depth(root.left), min_depth(root.right)) + 1


def has_path_sum(root: TreeNode, target: int) -> bool:
    """ 从根节点到叶子节点是否存在和为 `target` 的路径
    """
    if not root:
        return False

    if not root.left and not root.right:
        return root.val == target

    return has_path_sum(root.left, target-root.val) or \
           has_path_sum(root.right, target-root.val)


def print_path_sum(root: TreeNode, target: int) -> list:
    """ 打印根节点到叶子节点和为　target　的路径
    """

    def dfs(node, sum_val, item):
        if not node:
            return

        if not node.left and not node.right and node.val == sum_val:
            item += [node.val]
            result.append(item)
            return

        dfs(node.left, sum_val - node.val, item + [node.val])
        dfs(node.right, sum_val - node.val, item + [node.val])

    result = []
    dfs(root, target, [])
    return result


def pre_order_rec(root: TreeNode):

    def helper(node):
        if not node:
            return
        res.append(node.val)
        helper(node.left)
        helper(node.right)

    res = []
    helper(root)
    return res


def pre_order_iter(root: TreeNode):
    res = []

    if not root:
        return res

    stack = [root]
    while stack:
        node = stack.pop()
        res.append(node.val)

        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res


def post_order_rec(root: TreeNode):

    def helper(node):
        if not node:
            return

        helper(node.left)
        helper(node.right)
        res.append(node.val)

    res = []
    helper(root)

    return res


def post_order_iter(root: TreeNode):
    res = []
    if not root:
        return res

    stack = [root]
    while stack:
        node = stack.pop()

        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return res[::-1]


def count_nodes(root: TreeNode):
    """  计算树中所有节点数量
    """
    if not root:
        return 0

    left_height, left_node = 0, root
    right_height, right_node = 0, root

    while left_node:
        left_node = left_node.left
        left_height += 1

    while right_node:
        right_node = right_node.right
        right_height += 1

    # 满二叉树
    if left_height == right_height:
        return pow(2, left_height - 1)

    # count(root) + count(left_node) + count(right_node)
    return 1 + count_nodes(root.left) + count_nodes(root.right)


def invert_tree(root: TreeNode):
    """ 倒置树，将树的左右进行调换
    """
    if not root:
        return

    invert_tree(root.left)
    invert_tree(root.right)

    root.left, root.right = root.right, root.left

    return root


def lowest_common_ancestor(root: TreeNode, node1: TreeNode, node2:TreeNode):
    """ 查找树中两个节点的最低公共祖先
    """

    def recurse_tree(current_node):
        if not current_node:
            return False

        left = recurse_tree(current_node.left)
        right = recurse_tree(current_node.right)

        mid = current_node == node1 or current_node == node2

        if mid + left + right >= 2:
            return current_node

        return mid or left or right

    recurse_tree(root)

