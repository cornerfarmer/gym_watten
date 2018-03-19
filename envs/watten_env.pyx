import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum

from gym.envs.classic_control import rendering

import numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import sys
cdef enum Color:
    EICHEL = 0
    GRUEN = 1
    HERZ = 2
    SCHELLN = 3


cdef enum Value:
    SAU = 7
    KOENIG = 6
    OBER = 5
    UNTER = 4
    ZEHN = 3
    NEUN = 2
    ACHT = 1
    SIEBEN = 0


cdef struct Card:
    Color color
    Value value
    int id

cdef struct Player:
    vector[Card*] hand_cards
    int tricks
"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.hand_cards.clear()
        self.tricks = 0

    def get_state(self):
        return self.hand_cards, self.tricks

    def set_state(self, state):
        self.hand_cards = state[0][:]
        self.tricks = state[1]

    def get_trick_array(self):
        if self.tricks == 0:
            return [0, 0]
        elif self.tricks == 1:
            return [1, 0]
        elif self.tricks == 2:
            return [0, 1]
        elif self.tricks == 3:
            return [1, 1]"""

cdef struct State:
    vector[Card*] cards_left
    int current_player
    Card* table_card
    vector[Card*] lastTrick
    vector[Card*] player0_hand_cards
    int player0_tricks
    vector[Card*] player1_hand_cards
    int player1_tricks

cdef struct Observation:
    int hand_cards[4][8][2]
    int tricks[4]

cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& value)
    void random_shuffle[Iter](Iter first, Iter last)

ctypedef vector[Card*] card_vec

cdef class WattenEnv:
    metadata = {'render.modes': ['human', 'rgb_array']}
    cdef int _number_of_cards
    cdef int _number_of_hand_cards
    cdef object action_space
    cdef object observation_space
    cdef int steps
    cdef vector[Card*] cards
    cdef Player* player
    cdef Player[2] player_storage
    cdef vector[Player*] players
    cdef int current_player
    cdef Card* table_card
    cdef object viewer
    cdef vector[Card*] lastTrick
    cdef vector[Card*] cards_left
    cdef Observation obs
    cdef object render_card_trans

    def __cinit__(self):
        self._number_of_cards = 32
        self._number_of_hand_cards = 3
        self.action_space = spaces.Discrete(self._number_of_cards)
        self.observation_space = spaces.Tuple((spaces.Box(0, 1, [4, 8, 2]), spaces.Box(0, 1, [4])))
        self.steps = 0
        for c in [Color.EICHEL, Color.GRUEN]:
            for v in [Value.SAU, Value.KOENIG, Value.OBER, Value.UNTER]:
                card = <Card *>PyMem_Malloc(sizeof(Card))
                card.color = c
                card.value = v
                card.id = self.cards.size()
                self.cards.push_back(card)

        self.current_player = 0
        self.table_card = NULL
        self.viewer = None
        #self.obs.hand_cards = np.zeros([4, 8, 2])
        #self.obs.tricks = np.zeros([4])
        self.lastTrick.resize(2)
        self.render_card_trans = {}

        for i in range(2):
            self.players.push_back(&self.player_storage[i])

    def _seed(self, seed):
        pass

    cdef Observation _step(self, int action):

        reward = self._act(action, self.players[self.current_player])

        return self._obs()

    cdef bool _is_done(self):
        return self.players[0].hand_cards.size() + self.players[1].hand_cards.size() == 0 or self.players[0].tricks == 2 or self.players[1].tricks == 2

    cdef list _act(self, int action, Player* player):
        cdef Card* card
        if action is None:
            card = player.hand_cards[0]
        else:
            card = self.cards[action]

        pos = find(player.hand_cards.begin(), player.hand_cards.end(), card)
        if pos != player.hand_cards.end():
            player.hand_cards.erase(pos)

            if self.table_card is NULL:
                self.table_card = card

                self.current_player = 1 - self.current_player

                return []
            else:

                reward = [0, 0]

                better_player = self._match(self.table_card, card)
                reward[1 - better_player] = 1

                if better_player == 0:
                    self.current_player = 1 - self.current_player

                self.players[self.current_player].tricks += 1
                if self.players[self.current_player].tricks == 2:
                    reward[1 - better_player] += 5

                self.lastTrick[0] = self.table_card
                self.lastTrick[1] = card
                self.table_card = NULL
                return reward
        else:
            if self.table_card is NULL:
                return [-1]
            else:
                return [-1]

    cdef int _match(self, Card* first_card, Card* second_card):
        if self._get_value(first_card, first_card) >= self._get_value(second_card, first_card):
            return 0
        else:
            return 1

    cdef int _get_value(self, Card* card, Card* first_card):
        if False and card.color is Color.HERZ and card.value is Value.KOENIG:
            return 18
        elif False and card.color is Color.SCHELLN and card.value is Value.SIEBEN:
            return 17
        elif False and card.color is Color.EICHEL and card.value is Value.SIEBEN:
            return 16
        if False and card.color is Color.HERZ:
            return card.value + 9
        elif card.color is first_card.color:
            return card.value + 1
        else:
            return 0

    cdef Observation reset(self):
        self.cards_left = self.cards
        random_shuffle(self.cards_left.begin(), self.cards_left.end())

        cdef Player* player
        for player in self.players:
            player.hand_cards.clear()
            player.tricks = 0

            for i in range(self._number_of_hand_cards):
                player.hand_cards.push_back(self.cards_left.back())
                self.cards_left.pop_back()

        self.current_player = 0
        self.table_card = NULL
        self.lastTrick[0] = NULL
        self.lastTrick[1] = NULL

        return self._obs()

    cdef State get_state(self):
        cdef State state
        state.cards_left = self.cards_left
        state.current_player = self.current_player
        state.table_card = self.table_card
        state.lastTrick = self.lastTrick
        state.player0_hand_cards = self.players[0].hand_cards
        state.player0_tricks = self.players[0].tricks
        state.player1_hand_cards = self.players[1].hand_cards
        state.player1_tricks = self.players[1].tricks
        return state

    cdef void set_state(self, State state):
        self.cards_left = state.cards_left
        self.current_player = state.current_player
        self.table_card = state.table_card
        self.lastTrick = state.lastTrick
        self.players[0].hand_cards = state.player0_hand_cards
        self.players[0].tricks = state.player0_tricks
        self.players[1].hand_cards = state.player1_hand_cards
        self.players[1].tricks = state.player1_tricks

    cdef Observation _obs(self):
        cdef Player* player = self.players[self.current_player]

        cdef int i,j,k
        for i in range(4):
            for j in range(8):
                for k in range(2):
                    self.obs.hand_cards[i][j][k] = 0

        for card in player.hand_cards:
            self.obs.hand_cards[<int>card.color][<int>card.value][0] = 1

        if self.table_card is not NULL:
            self.obs.hand_cards[<int>self.table_card.color][<int>self.table_card.value][1] = 1

        #for card in self.players[1 - self.current_player].hand_cards:
        #    self.obs[0][card.color.value][card.value.value][2] = 1

        self.obs.tricks[0] = (player.tricks == 1 or player.tricks == 3)
        self.obs.tricks[1] = (player.tricks == 2 or player.tricks == 3)

        self.obs.tricks[2] = (self.players[1 - self.current_player].tricks == 1 or self.players[1 - self.current_player].tricks == 3)
        self.obs.tricks[3] = (self.players[1 - self.current_player].tricks == 2 or self.players[1 - self.current_player].tricks == 3)

        return self.obs

    cdef Observation regenerate_obs(self):
        return self._obs()

    cdef string _filename_from_card(self, Card* card):
        cdef string filename
        if card.color is Color.EICHEL:
            filename += <char*>"E"
        elif card.color is Color.GRUEN:
            filename += <char*>"G"
        elif card.color is Color.HERZ:
            filename += <char*>"H"
        elif card.color is Color.SCHELLN:
            filename += <char*>"S"

        if card.value is Value.SAU:
            filename += <char*>"A"
        elif card.value is Value.KOENIG:
            filename += <char*>"K"
        elif card.value is Value.OBER:
            filename += <char*>"O"
        elif card.value is Value.UNTER:
            filename += <char*>"U"
        elif card.value is Value.ZEHN:
            filename += <char*>"10"
        elif card.value is Value.NEUN:
            filename += <char*>"9"
        elif card.value is Value.ACHT:
            filename += <char*>"8"
        elif card.value is Value.SIEBEN:
            filename += <char*>"7"
        return filename

    cdef object _create_render_card(self, Card* card, int card_width, int card_height):
        image = rendering.Image("cards/" + self._filename_from_card(card).decode("utf-8") + ".png", card_width, card_height)
        image.attrs.clear()
        return image

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        card_height = 108
        card_width = 60
        cdef Card* card

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            for card in self.cards:
                render_card = self._create_render_card(card, card_width, card_height)
                render_card_trans = rendering.Transform(translation=(0, 0))
                render_card.add_attr(render_card_trans)

                self.render_card_trans[card.id] = render_card_trans
                self.viewer.add_geom(render_card)

        for card in self.cards:
            self.render_card_trans[card.id].set_translation(-card_width, -card_height)

        for p in range(2):
            for i in range(5):
                if i < self.players[p].hand_cards.size():
                    xpos = (i - 2) * (card_width + 20) + screen_width / 2
                    ypos = screen_height if p == 0 else card_height
                    self.render_card_trans[self.players[p].hand_cards[i].id].set_translation(xpos - card_width / 2, ypos - card_height / 2)

        if self.table_card is not NULL:
            self.render_card_trans[self.table_card.id].set_translation(screen_width / 2 - card_width / 4 * 3, screen_height / 2 - card_height / 4)
        elif self.lastTrick[0] is not NULL:
            self.render_card_trans[self.lastTrick[0].id].set_translation(screen_width / 2 - card_width / 4 * 3, screen_height / 2 - card_height / 4)
            self.render_card_trans[self.lastTrick[1].id].set_translation(screen_width / 2 - card_width / 4 * 1, screen_height / 2 + card_height / 4)

        return self.viewer.render()
