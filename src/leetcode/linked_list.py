# -*- coding: utf-8 -*-

"""
"""


class ListNode:
    def __init__(self, x=None):
        self.val = x
        self.next = None


def add_two_numbers(l1, l2):
    if not l1 or not l2:
        return l1 or l2
    tmp = 0
    pnode = phead = ListNode(-1)
    while l1 or l2 or tmp:
        if l1:
            tmp += l1.val
            l1 = l1.next
        if l2:
            tmp += l2.val
            l2 = l2.next
        pnode.next = ListNode(tmp % 10)
        pnode = pnode.next
        tmp //= 10
    return phead.next


def merge_two_lists(l1, l2):
    if not l1 or not l2:
        return l1 or l2

    pnode = phead = ListNode(-1)
    while l1 and l2:
        if l1.val < l2.val:
            pnode.next = l1
            l1 = l1.next
        else:
            pnode.next = l2
            l2 = l2.next
        pnode = pnode.next
    pnode.next = l1 or l2

    return phead.next


def merge_k_lists(lists):
    if not lists:
        return

    if len(lists) == 1:
        return lists[0]

    if len(lists) == 2:
        return merge_two_lists(lists[0], lists[1])

    mid = len(lists) // 2
    left = lists[:mid]
    right = lists[mid:]

    left_head = merge_k_lists(left)
    right_head = merge_k_lists(right)

    return merge_two_lists(left_head, right_head)


def swap_pairs(head: ListNode) -> ListNode:
    if not head:
        return head





