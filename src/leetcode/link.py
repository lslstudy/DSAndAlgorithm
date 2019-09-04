# -*- coding: utf-8 -*-

""" LinkedList Algorithms
"""


class ListNode:
    def __init__(self, val=None):
        self.val = val
        self.next = None


class DoubleLink:
    def __init__(self, val=None):
        self.val = val
        self.prev = None
        self.next = None


def remove_elements(head: ListNode, target: int) -> ListNode:
    """ given an link and remove elem if elem equal target
    """
    node = answer = ListNode(0)
    while head.next:
        if head.val != target:
            node.next = ListNode(head.val)
            node = node.next
        head = head.next
    return answer.next


class LRUCache:
    """ design a data structure and implements
        LRU(latest recently used)
        hashmap + double linkedlist
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.hashmap = dict()
        self.head = DoubleLink()
        self.tail = DoubleLink()
        self._init_link()

    def _init_link(self):
        self.head.next = self.tail
        self.tail.prev = self.head

    def put(self, key):
        if key not in self.hashmap:
            if self.size >= self.capacity:
                self.size -= 1
                self._remove_last(self.tail.prev)
        else:
            node = self.hashmap.get(key)
        self._move_to_head(key)
        self.size += 1

    def get(self, key) -> int:
        if key not in self.hashmap:
            return -1
        node = self.hashmap.get(key)
        if node.next != self.tail.prev:
            self._remove_last(self.tail.prev)
        else:
            node.prev.next = node.next
            node.next.prev = node.prve
        self._move_to_head(node)
        return node

    def _move_to_head(self, node):
        tmp = self.head.next
        self.head.next = node
        node.next = tmp
        node.prev = self.head

    def _remove_last(self, node):
        node.prev.next = self.tail
        self.tail.prev = node.prev



