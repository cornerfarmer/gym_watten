from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

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

cdef struct State:
    vector[Card*] cards_left
    int current_player
    Card* table_card
    vector[Card*] last_tricks
    vector[Card*] player0_hand_cards
    int player0_tricks
    vector[Card*] player1_hand_cards
    int player1_tricks

ctypedef vector[int] hand_card_row
ctypedef vector[hand_card_row] hand_card_set

cdef struct Observation:
    vector[hand_card_set] sets#[4][8][6]
    vector[int] scalars#[4]

cdef class WattenEnv:
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
    cdef vector[Card*] last_tricks
    cdef vector[Card*] cards_left
    cdef object render_card_trans
    cdef int last_winner
    cdef bool invalid_move
    cdef bool minimal
    cdef public int max_number_of_tricks

    cdef void seed(self, unsigned int seed)
    cdef void step(self, int action, Observation* obs=?)
    cdef bool is_done(self)
    cdef void _act(self, int action, Player* player)
    cdef int _match(self, Card* first_card, Card* second_card)
    cdef int _get_value(self, Card* card, Card* first_card)
    cdef void reset(self, Observation* obs=?)
    cdef State get_state(self)
    cdef void set_state(self, State* state)
    cdef void _obs(self, Observation* obs)
    cdef void regenerate_obs(self, Observation* obs)
    cdef string _filename_from_card(self, Card* card)
    cdef object _create_render_card(self, Card* card, int card_width, int card_height)
