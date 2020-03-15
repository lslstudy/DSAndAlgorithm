# -*- coding: utf-8 -*-

""" target offer
"""

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Singleton(object):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            obj = super(Singleton, cls)
            cls._instance = obj.__new__(cls, *args, **kwargs)
        return cls._instance


def singleton_decorator(cls, *args, **kwargs):
    instances = dict()

    def get_instance():
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


def first_common_node(phead1, phead2):
    """ given two list and find first common node
    """
    l1 = get_list_length(phead1)
    l2 = get_list_length(phead2)
    diff = abs(l1 - l2)

    if l1 > l2:
        long, short = phead1, phead2
    else:
        long, short = phead2, phead1

    for i in range(diff):
        long = long.next

    while long and short and short != long:
        long = long.next
        short = short.next
    return long


def get_list_length(phead):
    if not phead:
        return 0
    length = 0
    while phead:
        phead = phead.next
        length += 1
    return length


def binary_search_tree2list(root: TreeNode):
    if not root:
        return None
    if not root.left and not root.right:
        return root

    binary_search_tree2list(root.left)
    left = root.left

    if left:
        while left.right:
            left = left.right
        root.left, left.right = left, root

    binary_search_tree2list(root.right)
    right = root.right

    if right:
        while right.left:
            right = right.left
        root.right, right.left = right, root

    # iter to head node and return
    while root.left:
        root = root.left
    return root


def valid_post_order(sequence):
    """ 输入序列判断是否为某二叉搜索树的后续遍历结果
    """
    if not sequence:
        return False
    root = sequence[-1]
    length = len(sequence)
    if min(sequence) > root or max(sequence) < root:
        return True
    index = 0
    for i in range(length - 1):
        index = i
        if sequence[i] > root:
            break
    for j in range(index+1, length-1):
        if sequence[j] < root:
            return False

    left = True
    if index > 0:
        left = valid_post_order(sequence[:index])

    right = True
    if index < length - 1:
        right = valid_post_order(sequence[index: length-1])

    return left and right


def common_parent(pnode1, pnode2, root):
    if not pnode1 or not pnode2:
        return None
    if pnode1 == pnode2:
        return None
    val1, val2 = pnode1.val, pnode2.val
    while root:
        if (val1 - root.val) * (val2 - root.val) <= 0:
            return root.val
        elif val1 > root.val and val2 > root.val:
            root = root.right
        else:
            root = root.left
    return None


def path_sum(root, target):
    if not root:
        return []

    if not root.left and not root.right:
        return [[root.val]] if target == root.val else []

    # stack = []
    # left_stack = path_sum(root.left, target - root.val)
    # for i in left_stack:
    #     i.insert(0, root.val)
    #     stack.append(i)
    # right_stack = path_sum(root.right, target - root.val)
    # for i in right_stack:
    #     i.insert(0, root.val)
    #     stack.append(i)
    # return stack

    a = path_sum(root.left, target-root.val) + path_sum(root.right, target-root.val)
    return [[root.val] + seq for seq in a]


def mirror_tree(root):
    if not root:
        return
    if not root.left and not root.right:
        return root
    root.left, root.right = root.right, root.left
    mirror_tree(root.left)
    mirror_tree(root.right)
    return root


def two_dim_search(array, target):
    if not array:
        return False
    rows, cols = len(array), len(array[0])
    # 左下角开始查找
    row, col = len(array) - 1, 0
    while row >= 0 and col < cols:
        if array[row][col] == target:
            return True
        elif array[row][col] < target:
            col += 1
        else:
            row -= 1
    return False


def count_of_1_bin(number):
    count = 0
    if number < 0:
        number = number & 0xffffffff
    while number:
        count += 1
        number = (number - 1) & number
    return count


def print_tree(root):
    if not root:
        return []
    stack, ans = [root], []
    while stack:
        node = stack.pop(0)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
        ans.append(node.val)
    return ans


def remove_same_node(phead: ListNode):
    if not phead:
        return
    pre_node, pnode = None, phead
    while pnode:
        # 检查是否需要删除
        is_delete = False
        next_node = pnode.next
        if next_node and next_node.val == pnode.val:
            is_delete = True

        # 删除节点逻辑
        if is_delete:
            pre_node = pnode
            pnode = pnode.next
        else:
            to_be_delete = pnode
            node_val = pnode.val
            while to_be_delete and to_be_delete.val == node_val:
                to_be_delete = to_be_delete.next
            if not pre_node:
                phead, pnode = to_be_delete, to_be_delete
                continue
            else:
                pre_node.next = to_be_delete
            pnode = pre_node
    return phead


def is_balance_tree(root):
    """ abs(left_depth - right_depth) <= 1
    """
    if not root:
        return True

    # 当前节点是否符合
    left_depth, right_depth = get_depth(root.left), get_depth(root.right)
    if abs(left_depth - right_depth) > 1:
        return False

    # 左节点和右节点同时满足
    return is_balance_tree(root.left) and is_balance_tree(root.right)


def get_depth(root):
    if not root:
        return 0
    return max(get_depth(root.left), get_depth(root.right)) + 1


def print_from_tail2head(phead: ListNode):
    if not phead:
        return []
    stack = []
    while phead:
        stack.append(phead.val)
        phead = phead.next
    return stack[::-1]


def reverse_list_rec(phead: ListNode):
    if not phead or not phead.next:
        return phead
    node = reverse_list_rec(phead.next)
    phead.next.next = phead
    phead.next = None
    return node


def reverse_list_iter(phead: ListNode):
    reverse_head = None
    pnode = phead
    prev = None
    while pnode:
        pnext = pnode.next
        if not pnext:
            reverse_head = pnode
        pnode.next = prev
        prev = pnode
        pnode = pnext
    return reverse_head


def merge_sort_list(phead1, phead2):
    if not phead1 or not phead2:
        return phead1 or phead2
    pnode = pans = ListNode(-1)
    while phead1 and phead2:
        if phead1.val < phead2.val:
            pnode.next = phead1
            phead1 = phead1.next
        else:
            pnode.next = phead2
            phead2 = phead2.next
        pnode = pnode.next
    pnode.next = phead1 or phead2
    return pans.next


def find_two_sum_number(array, target):
    if not array or array[-2] + array[-1] < target:
        return []
    left, right = 0, len(array) - 1
    while left < right:
        tmp = array[left] + array[right]
        if tmp < target:
            left += 1
        elif tmp > target:
            right -= 1
        else:
            return [array[left], array[right]]
    return []


def char_permutation(chars):
    if not chars:
        return []
    if len(chars) == 1:
        return list(chars)

    chars = list(chars)
    answer = []

    def _helper(ss, tmp):
        if not ss:
            answer.append(tmp)
            return
        a = ss.pop(0)
        for x in ss:
            _helper(ss, tmp+a)
    _helper(chars, "")
    return answer


if __name__ == '__main__':
    s = "abc"
    ans = char_permutation(s)
    print(ans)
