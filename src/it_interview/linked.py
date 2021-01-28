# -*- coding: utf-8 -*-

""" 链表操作相关题目
"""

import math


class Node:
    def __init__(self, val=None):
        self.val = val
        self.next = None


class DoubleNode:
    def __init__(self, val=None):
        self.val = val
        self.last = None
        self.next = None


def print_common_part(head1: Node, head2: Node):
    """ 打印两个有序链表的公共部分
    """
    print("Common Part: ")
    while head1 and head2:
        if head1.val < head2.val:
            head1 = head1.next
        elif head1.val > head2.val:
            head2 = head2.next
        else:
            print(head1.val + " ")
            head1 = head1.next
            head2 = head2.next


def remove_last_kth_node(head: Node, k: int):
    """ 删除单链表的倒数第Ｋ个节点
    """
    if not head or k < 1:
        return head
    cur = head
    while cur:
        k -= 1
        cur = cur.next
    if k == 0:
        head = head.next
    if k < 0:
        cur = head
        k += 1
        while k != 0:
            cur = cur.next
            k += 1
        cur.next = cur.next.next
    return head


def remove_double_last_kth_node(head: DoubleNode, k: int):
    """ 删除双链表的倒数第k个节点
    """
    if not head or k < 1:
        return head
    cur = head
    while cur:
        k -= 1
        cur = cur.next
    if k == 0:
        head = head.next
        head.last = None
    if k < 0:
        cur = head
        k += 1
        while k != 0:
            cur = cur.next
            k += 1
        tmp = cur.next.next
        cur.next = tmp
        if tmp:
            tmp.last = cur
    return head


def remove_mid_node(head: Node):
    """ 删除链表的中间节点
    """
    if not head or not head.next:
        return head
    if not head.next.next:
        return head.next
    pre, cur = head, head.next.next
    while cur.next and cur.next.next:
        pre = pre.next
        cur = cur.next.next
    pre.next = pre.next.next
    return head


def remove_node_by_ratio(head: Node, a: int, b: int):
    """ 删除链表的 a/b处的节点
    """
    if not head or a < 1 or a > b:
        return head
    n, cur = 0, head
    while cur:
        n += 1
        cur = cur.next
    n = math.ceil((a * n) // b)  # 向上取整
    if n == 1:
        head = head.next
    if n > 1:
        cur = head
        n -= 1
        while n != 1:
            cur = cur.next
        cur.next = cur.next.next
    return head


def reverse_list(head: Node):
    """ 翻转单链表
    """
    if not head:
        return head
    pre, ne = None, None
    while head:
        ne = head.next
        head.next = pre
        pre = head
        head = ne
    return pre


def reverse_double_list(head: DoubleNode):
    """ 翻转双链表
    """
    if not head:
        return
    pre, nex = None, None
    while head:
        nex = head.next
        head.next = pre
        head.last = nex
        pre = head
        head = nex
    return pre


def reverse_from_to_list(head: Node, f: int, t: int):
    """ 翻转部分链表，翻转f到t部分
    """
    length, node1 = 0, head
    fpre, tpos = None, None
    while node1:
        length += 1
        fpre = node1 if length == f - 1 else fpre
        tpos = node1 if length == t + 1 else tpos
        node1 = node1.next
    if f > t or f < 1 or t > length:
        return head


def josephus_kill(head: Node, m: int):  # O(N*)
    if not head or head.next == head or m < 1:
        return head
    last: Node = head
    while last.next != head:
        last = last.next
    count = 0
    while head != last:
        count += 1
        if count == m:
            last.next = head.next
            count = 0
        else:
            last = last.next
        head = last.next
    return head


def node_is_palindrome(head: Node):
    if not head:
        return True
    stack = []
    cur = head
    while cur:
        stack.append(cur.val)
        cur = cur.next
    while head:
        value = stack.pop(-1)
        if head.val != value:
            return False
        head = head.next
    return True


def node_list_partition(head: Node, pivot: int):
    shead, stail = None, None
    ehead, etail = None, None
    bhead, btail = None, None
    while head:
        nex = head.next
        head.next = None
        if head.val < pivot:
            if shead is None:
                shead = head
                stail = head
            else:
                stail.next = head
                stail = head
        elif head.val == pivot:
            if ehead is None:
                ehead = head
                etail = head
            else:
                etail.next = head
                etail = head
        else:
            if bhead is None:
                bhead = head
                btail = head
            else:
                btail.next = head
                btail = head
        head = nex
    if stail is not None:
        stail.next = ehead
        etail = stail if etail is None else etail
    if etail is not None:
        etail.next = bhead
    return shead if shead is not None else (ehead if ehead is not None else bhead)


def copy_random_list():
    pass


def add_node(node1: Node, node2: Node):
    if not node1 or not node2:
        return node2 or node1
    s1, s2 = [], []
    while node1:
        s1.append(node1.val)
        node1 = node1.next
    while node2:
        s2.append(node2.val)
        node2 = node2.next
    node, pre = None, None
    ca = 0
    while s1 or s2:
        n1 = 0 if not s1 else s1.pop(-1)
        n2 = 0 if not s2 else s2.pop(-1)
        n = n1 + n2 + ca
        pre = node
        node = Node(n % 10)
        node.next = pre
        ca = n // 10
    if ca:
        pre = node
        node = Node(ca)
        node.next = pre
    return node


def get_loop_node(head: Node):
    if not head or not head.next or not head.next.next:
        return None
    fast, slow = head.next.next, head.next
    # 有环
    while slow != fast:
        if not fast.next or not fast.next.next:
            return None
        fast = fast.next.next
        slow = slow.next
    #
    pass


if __name__ == '__main__':
    import requests
    import os
    import json
    headers = {
        "Host": "baike.baidu.com",
        "User-Agent": "Mozilla/5.0(X11;Ubuntu;Linuxx86_64;rv:70.0) Gecko/20100101Firefox/70.0",
        "Accept": "text/html, application/xhtml+xml, application/xml;q=0.9, */*; q=0.8",
        "Accept-Language": "zh-CN,zh;q = 0.8, zh-TW;q = 0.7, zh-HK;q = 0.5, en-US;q = 0.3, en;q=0.2",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "http://baike.baidu.com/fenlei/%E6%94%BF%E6%B2%BB%E4%BA%BA%E7%89%A9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }
    response = requests.get(url="https://baike.baidu.com/item/%E5%88%98%E5%8D%8E%E6%B8%85/2541038", headers=headers)
    print(response.text)

    fw = open("./baike.json", "w+")
    js_obj = {
        "url": response.url,
        "html": response.text,
        "title": "title"
    }
    fw.write(str(js_obj) + os.linesep)
    fw.close()
