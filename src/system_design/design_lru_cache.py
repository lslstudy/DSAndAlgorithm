# -*- coding: utf-8 -*-

""" design an LRU cache:
        1. caching the results of web queries
        2. assume they're valid
        3. can assume this fit memory
"""


class Node(object):

    def __init__(self, results):
        self.results = results
        self.prev = None
        self.next = None


class LinkedList(object):

    def __init__(self):
        self.head = None
        self.tail = None

    def move_to_front(self, node: Node):
        """ move anywhere node to front
        :return:
        """
        self.head = node
        node.prev = self.head
        node.next = self.tail

    def append_to_front(self, node: Node):
        """ add node to LinkedList head
        """
        if self.head is None:
            self.head = node
            self.tail = node
            return
        origin = self.head
        self.head = node
        origin.prev = node
        node.prev = self.head
        node.next = origin

    def remove_from_tail(self):
        """ remove node from LinkedList tail
        :return:
        """
        if self.tail is None:
            return
        last_node = self.tail.prev
        last_node.prev.next = self.tail


class LRUCache(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.size = 0
        self.linked_list = LinkedList()
        self.lookup = dict()

    def get(self, query):
        """ get stored query results from the cache

        Accessing a node updates its position to the front of the LRU cache
        """
        node = self.lookup.get(query)
        if node is None:
            return None
        self.linked_list.move_to_front(node)
        return node.results

    def set(self, results, query):
        """ set the results for the given query key in the cache
        :param results:
        :param query:
        :return:
        """
        node = self.lookup.get(query)
        if node is not None:
            # key exists in the cache, update value
            node.results = results
            self.linked_list.move_to_front(node)
        else:
            # key dose not exists in cache
            if self.size == self.max_size:
                # remove the oldest entry from ths linked list and lookup
                self.lookup.pop(self.linked_list.tail.query, None)
                self.linked_list.remove_from_tail()
            else:
                self.size += 1
            # add the new key and value
            new_node = Node(results=results)
            self.linked_list.append_to_front(new_node)
            self.lookup[query] = new_node


if __name__ == '__main__':
    # a = LinkedList()
    # a.append_to_front(node=Node(10))
    # a.append_to_front(node=Node(11))
    # a.append_to_front(node=Node(12))

    # import sys
    # print(sys.maxsize)
    # print(sys.version)
    # print(sys.platform)
    # import bisect

    import time

