# -*- coding: utf-8 -*-


class TreeNode:

    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


def pre_order_rec(root: TreeNode):
    if not root:
        return []

    ans = []

    def _helper(node: TreeNode):
        if not node:
            return
        ans.append(node.val)
        _helper(node.left)
        _helper(node.right)
    _helper(root)
    return ans


def in_order_rec(root: TreeNode):
    if not root:
        return []

    ans = []

    def _helper(node: TreeNode):
        if not node:
            return
        _helper(node.left)
        ans.append(node.val)
        _helper(node.right)
    _helper(root)
    return ans


def post_order_rec(root: TreeNode):
    if not root:
        return []

    ans = []

    def _helper(node: TreeNode):
        if not node:
            return
        _helper(node.left)
        _helper(node.right)
        ans.append(node.val)
    _helper(root)
    return ans


def pre_order_iter(root: TreeNode):
    if not root:
        return []
    stack, ans = [root], []
    while stack:
        node = stack.pop(-1)
        ans.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return ans


def in_order_iter(root: TreeNode):
    if not root:
        return []
    ans = []
    if root:
        stack = []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop(-1)
                ans.append(node.val)
                root = node.right
    return ans


def post_order_iter(root: TreeNode):
    ans = []
    if root:
        s1, s2 = [], []
        s1.append(root)
        while s1:
            head = s1.pop()
            s2.append(s1)
            if head.left:
                s1.append(head.left)
            if head.right:
                s1.append(head.right)
        while s2:
            ans.append(s2.pop().val)
    return ans


def tree_serialize(head: TreeNode) -> str:
    """ 二叉树序列化(前序遍历)
    """
    if not head:
        return "#!"
    res = head.val + "!"
    res += tree_serialize(head.left)
    res += tree_serialize(head.right)
    return res


def tree_deserialize(s: str):
    """ 反序列化二叉树
    """
    a = [x for x in s.split("!") if x]

    def _recover(queue):
        elem = queue.pop(0)
        if elem == "#":
            return None
        node = TreeNode(int(elem))
        node.left = _recover(queue)
        node.right = _recover(queue)
        return node

    return _recover(a)


def print_by_level(head: TreeNode):
    if not head:
        return
    queue, ans = [], []
    queue.append(head)
    while queue:
        curr_len = len(queue)
        _tmp = []
        for _ in range(curr_len):
            node = queue.pop(0)
            _tmp.append(str(node.val))
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        ans.append(_tmp)
    for i in range(len(ans)):
        print(f"Level: {i+1} : {' '.join(ans[i])}")


if __name__ == '__main__':
    a = TreeNode(1)
    b = TreeNode(2)
    c = TreeNode(3)
    d = TreeNode(4)
    e = TreeNode(5)
    f = TreeNode(6)
    g = TreeNode(7)
    h = TreeNode(8)
    a.left, a.right = b, c
    b.left = d
    c.left, c.right = e, f
    e.left, e.right = g, h
    ans = print_by_level(a)
    print(ans)
