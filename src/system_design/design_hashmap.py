# -*- coding: utf-8 -*-

""" design a hash map:
        1.all keys are integers
        2.assume fit memory
        3.
"""


class Item(object):

    def __init__(self, key, val):
        self.key = key
        self.value = val


class HashMap(object):

    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def _hash_partition(self, key):
        return key % self.size

    def set(self, key, value):
        hash_index = self._hash_partition(key=key)

        # if key exists
        for item in self.table[hash_index]:
            if item.key == key:
                item.value = value
                return

        # key not exists
        self.table[hash_index].append(Item(key, value))

    def get(self, key):
        hash_index = self._hash_partition(key=key)
        for item in self.table[hash_index]:
            if item.key == key:
                return item.value
        raise KeyError("key not found")

    def remove(self, key):
        hash_index = self._hash_partition(key=key)
        for index, item in enumerate(self.table[hash_index]):
            if item.key == key:
                del self.table[hash_index][index]
                return
        raise KeyError("key not found")

