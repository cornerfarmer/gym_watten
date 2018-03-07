import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum

from gym.envs.classic_control import rendering

import numpy as np

class Color(Enum):
    EICHEL = 0
    GRUEN = 1
    HERZ = 2
    SCHELLN = 3


class Value(Enum):
    SAU = 7
    KOENIG = 6
    OBER = 5
    UNTER = 4
    ZEHN = 3
    NEUN = 2
    ACHT = 1
    SIEBEN = 0


class Card:
    def __init__(self, color, value, id):
        self.color = color
        self.value = value
        self.id = id

class Player:

    def __init__(self):
        self.reset()

    def reset(self):
        self.hand_cards = []
        self.tricks = 0

    def get_state(self):
        return self.hand_cards[:], self.tricks

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
            return [1, 1]

class WattenEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self._number_of_cards = 32
        self._number_of_hand_cards = 5
        self.action_space = spaces.Discrete(self._number_of_cards)
        self.observation_space = spaces.Tuple((spaces.Box(0, 1, [4, 8, 2]), spaces.Box(0, 1, [4])))
        self.steps = 0
        self.cards = []
        for c in Color:
            for v in Value:
                self.cards.append(Card(c, v, len(self.cards)))
        self.players = [Player(), Player()]
        self.current_player = 0
        self.table_card = None
        self.viewer = None
        self.lastTrick = None
        self.obs = [np.zeros([4, 8, 2]), np.zeros([4])]

    def _seed(self, seed):
        pass

    def _step(self, action):

        reward = self._act(action, self.players[self.current_player])

        return self._obs(), reward, len(reward) > 0 and reward[0] < 0 or self._is_done(), {}

    def _is_done(self):
        return len(self.players[0].hand_cards) + len(self.players[1].hand_cards) == 0 or self.players[0].tricks == 2 or self.players[1].tricks == 2

    def _act(self, action, player):
        if action is None:
            card = player.hand_cards[0]
        else:
            card = self.cards[action]

        if card in player.hand_cards:
            player.hand_cards.remove(card)

            if self.table_card is None:
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

                self.lastTrick = [self.table_card, card]
                self.table_card = None
                return reward
        else:
            if self.table_card is None:
                return [-1]
            else:
                return [-1]

    def _match(self, first_card, second_card):
        if self._get_value(first_card, first_card) >= self._get_value(second_card, first_card):
            return 0
        else:
            return 1

    def _get_value(self, card, first_card):
        if card.color is Color.HERZ and card.value is Value.KOENIG:
            return 18
        elif card.color is Color.SCHELLN and card.value is Value.SIEBEN:
            return 17
        elif card.color is Color.EICHEL and card.value is Value.SIEBEN:
            return 16
        if card.color is Color.HERZ:
            return int(card.value.value) + 9
        elif card.color is first_card.color:
            return int(card.value.value) + 1
        else:
            return 0

    def _reset(self):
        self.cards_left = self.cards[:]
        random.shuffle(self.cards_left)

        for player in self.players:
            player.reset()
            for i in range(self._number_of_hand_cards):
                player.hand_cards.append(self.cards_left.pop())

        self.current_player = 0
        self.table_card = None
        self.lastTrick = None

        return self._obs()

    def get_state(self):
        return self.cards_left[:], self.current_player, self.table_card, None if self.lastTrick is None else self.lastTrick[:], self.players[0].get_state(), self.players[1].get_state()

    def set_state(self, state):
        self.cards_left = state[0][:]
        self.current_player = state[1]
        self.table_card = state[2]
        self.lastTrick = None if state[3] is None else state[3][:]
        self.players[0].set_state(state[4])
        self.players[1].set_state(state[5])

    def _obs(self):
        player = self.players[self.current_player]

        self.obs[0].fill(0)
        for card in player.hand_cards:
            self.obs[0][card.color.value][card.value.value][0] = 1

        if self.table_card is not None:
            self.obs[0][self.table_card.color.value][self.table_card.value.value][1] = 1

        #for card in self.players[1 - self.current_player].hand_cards:
        #    self.obs[0][card.color.value][card.value.value][2] = 1

        self.obs[1][0] = (player.tricks == 1 or player.tricks == 3)
        self.obs[1][1] = (player.tricks == 2 or player.tricks == 3)

        self.obs[1][2] = (self.players[1 - self.current_player].tricks == 1 or self.players[1 - self.current_player].tricks == 3)
        self.obs[1][3] = (self.players[1 - self.current_player].tricks == 2 or self.players[1 - self.current_player].tricks == 3)

        return self.obs

    def regenerate_obs(self):
        return self._obs()

    def _filename_from_card(self, card):
        filename = ""
        if card.color is Color.EICHEL:
            filename += "E"
        elif card.color is Color.GRUEN:
            filename += "G"
        elif card.color is Color.HERZ:
            filename += "H"
        elif card.color is Color.SCHELLN:
            filename += "S"

        if card.value is Value.SAU:
            filename += "A"
        elif card.value is Value.KOENIG:
            filename += "K"
        elif card.value is Value.OBER:
            filename += "O"
        elif card.value is Value.UNTER:
            filename += "U"
        elif card.value is Value.ZEHN:
            filename += "10"
        elif card.value is Value.NEUN:
            filename += "9"
        elif card.value is Value.ACHT:
            filename += "8"
        elif card.value is Value.SIEBEN:
            filename += "7"
        return filename

    def _create_render_card(self, card, card_width, card_height):
        image = rendering.Image("cards/" + self._filename_from_card(card) + ".png", card_width, card_height)
        image.attrs.clear()
        return image

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        card_height = 108
        card_width = 60

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            for card in self.cards:
                render_card = self._create_render_card(card, card_width, card_height)
                render_card_trans = rendering.Transform(translation=(0, 0))
                render_card.add_attr(render_card_trans)

                card.render_card_trans = render_card_trans
                self.viewer.add_geom(render_card)

        for card in self.cards:
            card.render_card_trans.set_translation(-card_width, -card_height)

        for p in range(2):
            for i in range(5):
                if i < len(self.players[p].hand_cards):
                    xpos = (i - 2) * (card_width + 20) + screen_width / 2
                    ypos = screen_height if p == 0 else card_height
                    self.players[p].hand_cards[i].render_card_trans.set_translation(xpos - card_width / 2, ypos - card_height / 2)

        if self.table_card is not None:
            self.table_card.render_card_trans.set_translation(screen_width / 2 - card_width / 4 * 3, screen_height / 2 - card_height / 4)
        elif self.lastTrick is not None:
            self.lastTrick[0].render_card_trans.set_translation(screen_width / 2 - card_width / 4 * 3, screen_height / 2 - card_height / 4)
            self.lastTrick[1].render_card_trans.set_translation(screen_width / 2 - card_width / 4 * 1, screen_height / 2 + card_height / 4)

        return self.viewer.render()
