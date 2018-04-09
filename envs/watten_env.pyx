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
from libc.stdlib cimport srand
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
import sys

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
    vector[Card*] last_tricks
    vector[Card*] player0_hand_cards
    int player0_tricks
    vector[Card*] player1_hand_cards
    int player1_tricks

cdef struct Observation:
    int hand_cards[4][8][6]
    int tricks[4]

cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& value)
    void random_shuffle[Iter](Iter first, Iter last)

ctypedef vector[Card*] card_vec

cdef class WattenEnv:
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __cinit__(self, bool minimal=False):
        self._number_of_cards = 32
        self._number_of_hand_cards = 3
        self.action_space = spaces.Discrete(self._number_of_cards)
        self.observation_space = spaces.Tuple((spaces.Box(0, 1, [4, 8, 6], dtype=np.float32), spaces.Box(0, 1, [4], dtype=np.float32)))
        self.steps = 0

        if minimal:
            colors = [Color.EICHEL, Color.GRUEN]
            values = [Value.SAU, Value.KOENIG, Value.OBER, Value.UNTER]
        else:
            colors = [Color.EICHEL, Color.GRUEN, Color.HERZ, Color.SCHELLN]
            values = [Value.SAU, Value.KOENIG, Value.OBER, Value.UNTER, Value.ZEHN, Value.NEUN, Value.ACHT, Value.SIEBEN]

        for c in colors:
            for v in values:
                card = <Card *>PyMem_Malloc(sizeof(Card))
                card.color = c
                card.value = v
                card.id = self.cards.size()
                self.cards.push_back(card)

        self.current_player = 0
        self.table_card = NULL
        self.viewer = None
        self.minimal = minimal
        #self.obs.hand_cards = np.zeros([4, 8, 2])
        #self.obs.tricks = np.zeros([4])
        self.render_card_trans = {}

        for i in range(2):
            self.players.push_back(&self.player_storage[i])

    cdef void seed(self, unsigned int seed):
        srand(seed)

    cdef void step(self, int action, Observation* obs=NULL):
        self._act(action, self.players[self.current_player])

        if obs != NULL:
            self._obs(obs)

    cdef bool is_done(self):
        return self.players[0].hand_cards.size() + self.players[1].hand_cards.size() == 0 or self.players[0].tricks == 2 or self.players[1].tricks == 2 or self.invalid_move

    cdef void _act(self, int action, Player* player):
        cdef Card* card
        if action is -1:
            card = player.hand_cards[0]
        else:
            card = self.cards[action]

        pos = find(player.hand_cards.begin(), player.hand_cards.end(), card)
        if pos != player.hand_cards.end():
            player.hand_cards.erase(pos)
            self.invalid_move = False

            if self.table_card is NULL:
                self.table_card = card

                self.current_player = 1 - self.current_player
            else:
                if self.current_player == 1:
                    self.last_tricks.push_back(self.table_card)
                    self.last_tricks.push_back(card)
                else:
                    self.last_tricks.push_back(card)
                    self.last_tricks.push_back(self.table_card)

                better_player = self._match(self.table_card, card)

                if better_player == 0:
                    self.current_player = 1 - self.current_player

                self.players[self.current_player].tricks += 1
                if self.players[self.current_player].tricks == 2:
                    self.last_winner = self.current_player

                self.table_card = NULL
        else:
            self.invalid_move = True
            self.last_winner = 1 - self.current_player

    cdef int _match(self, Card* first_card, Card* second_card):
        if self._get_value(first_card, first_card) >= self._get_value(second_card, first_card):
            return 0
        else:
            return 1

    cdef int _get_value(self, Card* card, Card* first_card):
        if not self.minimal and card.color is Color.HERZ and card.value is Value.KOENIG:
            return 18
        elif not self.minimal and card.color is Color.SCHELLN and card.value is Value.SIEBEN:
            return 17
        elif not self.minimal and card.color is Color.EICHEL and card.value is Value.SIEBEN:
            return 16
        if not self.minimal and card.color is Color.HERZ:
            return card.value + 9
        elif card.color is first_card.color:
            return card.value + 1
        else:
            return 0

    cdef void reset(self, Observation* obs=NULL):
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
        self.last_tricks.clear()
        self.invalid_move = False

        if obs != NULL:
            self._obs(obs)

    cdef State get_state(self):
        cdef State state
        state.cards_left = self.cards_left
        state.current_player = self.current_player
        state.table_card = self.table_card
        state.last_tricks = self.last_tricks
        state.player0_hand_cards = self.players[0].hand_cards
        state.player0_tricks = self.players[0].tricks
        state.player1_hand_cards = self.players[1].hand_cards
        state.player1_tricks = self.players[1].tricks
        return state

    cdef void set_state(self, State* state):
        self.cards_left = state.cards_left
        self.current_player = state.current_player
        self.table_card = state.table_card
        self.last_tricks = state.last_tricks
        self.players[0].hand_cards = state.player0_hand_cards
        self.players[0].tricks = state.player0_tricks
        self.players[1].hand_cards = state.player1_hand_cards
        self.players[1].tricks = state.player1_tricks
        self.invalid_move = False

    cdef void _obs(self, Observation* obs):
        cdef Player* player = self.players[self.current_player]

        cdef int i,j,k
        for i in range(4):
            for j in range(8):
                for k in range(6):
                    obs.hand_cards[i][j][k] = 0

        for card in player.hand_cards:
            obs.hand_cards[<int>card.color][<int>card.value][0] = 1

        if self.table_card is not NULL:
            obs.hand_cards[<int>self.table_card.color][<int>self.table_card.value][1] = 1

        for i in range(max(0, <int>self.last_tricks.size() - 4), self.last_tricks.size()):
            obs.hand_cards[<int>self.last_tricks[i].color][<int>self.last_tricks[i].value][2 + ((self.last_tricks.size() - 1) / 2 - i / 2) * 2 + ((1 - i % 2) if self.current_player == 1 else (i % 2))] = 1

        #for card in self.players[1 - self.current_player].hand_cards:
        #    self.obs[0][card.color.value][card.value.value][2] = 1

        obs.tricks[0] = (player.tricks == 1 or player.tricks == 3)
        obs.tricks[1] = (player.tricks == 2 or player.tricks == 3)

        obs.tricks[2] = (self.players[1 - self.current_player].tricks == 1 or self.players[1 - self.current_player].tricks == 3)
        obs.tricks[3] = (self.players[1 - self.current_player].tricks == 2 or self.players[1 - self.current_player].tricks == 3)


    cdef void regenerate_obs(self, Observation* obs):
        self._obs(obs)

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
        elif self.last_tricks.size() >= 2:
            self.render_card_trans[self.last_tricks[0].id].set_translation(screen_width / 2 - card_width / 4 * 3, screen_height / 2 - card_height / 4)
            self.render_card_trans[self.last_tricks[1].id].set_translation(screen_width / 2 - card_width / 4 * 1, screen_height / 2 + card_height / 4)

        return self.viewer.render()
