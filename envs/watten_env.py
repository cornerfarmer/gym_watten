import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from enum import Enum

from gym.envs.classic_control import rendering

class Color(Enum):
    EICHEL = 1
    GRUEN = 2
    HERZ = 3
    SCHELLN = 4


class Value(Enum):
    SAU = 1
    KOENIG = 2
    OBER = 3
    UNTER = 4
    ZEHN = 5
    NEUN = 6
    ACHT = 7
    SIEBEN = 8


class Card:
    def __init__(self, color, value):
        self.color = color
        self.value = value

class Player:

    def __init__(self):
        self.reset()

    def reset(self):
        self.hand_cards = []
        self.tricks = 0

    def get_trick_array(self):
        if self.tricks == 0:
            return [0, 0]
        elif self.tricks == 1:
            return [1, 0]
        elif self.tricks == 2:
            return [0, 1]

class WattenEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self._number_of_cards = 32
        self._number_of_hand_cards = 5
        self.action_space = spaces.Discrete(self._number_of_cards)
        self.observation_space = spaces.Box(0, 1, [self._number_of_cards * 2 + 4])
        self.steps = 0
        self.cards = []
        for c in Color:
            for v in Value:
                self.cards.append(Card(c, v))
        self.players = [Player(), Player()]
        self.current_player = 0
        self.table_card = None
        self.viewer = None
        self.lastTrick = None

    def _step(self, action):

        reward = self._act(action, self.players[self.current_player])

        self.current_player = 1 - self.current_player

        return self._obs(), reward, reward < 0 or self._is_done(), {}

    def _is_done(self):
        return len(self.players[0].hand_cards) + len(self.players[1].hand_cards) == 0

    def _act(self, action, player):
        card = self.cards[action]

        if card in player.hand_cards:
            player.hand_cards.remove(card)

            if self.table_card is None:
                self.table_card = card
            else:
                self.lastTrick = [self.table_card, card]
                self.table_card = None
            return 0
        else:
            return -1

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

    def _obs(self):
        obs = []
        player = self.players[self.current_player]

        for card in self.cards:
            obs.append(1 if card in player.hand_cards else 0)

        for card in self.cards:
            obs.append(1 if card is self.table_card else 0)

        obs.extend(player.get_trick_array())
        obs.extend(self.players[1 - self.current_player].get_trick_array())

        return obs


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
