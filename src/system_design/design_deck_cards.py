# -*- coding: utf-8 -*-

""" design a deck of cards

    Is this a generic deck of cards for games like poker and black jack?
        Yes, design a generic deck then extend it to black jack
    Can we assume the deck has 52 cards (2-10, Jack, Queen, King, Ace) and 4 suits?
        Yes
    Can we assume inputs are valid or do we have to validate them?
        Assume they're valid
"""

import sys
import enum

from abc import ABCMeta, abstractmethod


class Suit(enum.Enum):
    HEART = 0
    DIAMOND = 1
    CLUBS = 2
    SPADE = 3


class Card(metaclass=ABCMeta):

    def __init(self, value, suit):
        self.value = value
        self.suit = suit
        self.is_available = True

    @property
    @abstractmethod
    def value(self):
        pass

    @property.setter
    @abstractmethod
    def value(self, other):
        pass


class BlackJackCard(Card):
    def __init__(self, value, suit):
        super(BlackJackCard, self).__init__(value, suit)

    def is_ace(self):
        return self._value == 1

    def is_face_card(self):
        return 10 < self._value <= 13

    @property
    def value(self):
        if self.is_ace():
            return 1
        elif self.is_face_card():
            return 10
        else:
            return self._value

    @value.setter
    def value(self, other):
        if 1 <= other <= 13:
            self._value = other
        else:
            raise ValueError("Invalid card value: {}".format(other))


class Hand(object):

    def __init__(self, cards):
        self.cards = cards

    def add_card(self, card):
        self.cards.append(card)

    def score(self):
        return sum([card.value for card in self.cards if card])


class BlackJackHand(Hand):
    BLACKJACK = 21

    def __init__(self, cards):
        super(BlackJackHand, self).__init__(cards)

    def score(self):
        pass
