# chess environment 
# functionality for representing and operating on the chess environment

# https://github.com/Zeta36/chess-alpha-zero/blob/master/src/chess_zero/env/chess_env.py
# how to proceed?
# we can either use the implementation provided here 
# or write everything in multidimensional arrays in pure functional code
# i really like functional style but getting better at object oriented
# programming cant hurt either. 
# using the prebuilt environment would save a massive amount of time 

# we need a minimal representation that can be fed into the evalnet
# doing it from scratch may be the best way and should allow porting to julia

# i think functional style with multidimensional arrays is the way to go
# what we need to track: 
# the pieces of both players each in their own 8x8 array => 8x8x12
# that allows one hot encoding for every layer
# this is helpful for passing everything to the neural net later on
# castling order 8x8x4
# en passant 8x8x1
# fifty move rule 8x8x1

# this will be the board state in a 8x8x(12 + 4 + 1 + 1) array
# from this we will need to generate a list of all possible legal moves
# in the alphazero paper they use an 8x8x73 array as a list of all legal moves

# lets reiterate what we're trying to do
# train the neural network and mcts to find the best move
# to accomplish this we need to write a chess environment suited for self play

# take board representation 
# get legal moves from representation 
# repeat the following: 
    # for every legal move predict the outcome of the game with a neural net
    # do mcts guided by network params and dirichlet noise to search optimal move
    # update network params to minimize prediction error

# once we have the trained network we can pass it any current board state and 
# find the optimal moves 

# first we need to generate self play games on which we can train the evaluator 
# network
# for this task we need to code a chess engine
# this is going to be more work than i originally anticipated 

##############################################################################
# SETTING UP THE BOARD AND CONSTANTS
# board representation code inspired by this helpful video
# https://www.youtube.com/watch?v=x9sPmLt-EBM&list=PLZ1QII7yudbc-Ky058TEaOstZHVbT-2hg&index=3
# we can use the bitarray library to define 64 bit integers 
from bitarray import bitarray
# from aenum import Enum
from aenum import IntEnum
import numpy as np
# a = bitarray(int(64))
# a.setall(False)
# =============================================================================
# def U64(init):
#     ret = bitarray(64)
#     ret.setall(init)
#     return(ret)
# =============================================================================

class mybitarray(bitarray):
    def __lshift__(self, count):
        return self[count:] + type(self)('0') * count
    def __rshift__(self, count):
        return type(self)('0') * count + self[:-count]
    def __repr__(self):
        return "{}('{}')".format(type(self).__name__,self.to01())

# global constants
MAXGAMEMOVES = 2048
BOARD_SQUARE_NUMBER = 120
PIECES = IntEnum("PIECES", "EMPTY wP wN wB wR wQ wK bP bN bB bR bQ bK", start = 0)
FILES = IntEnum("FILES", "FILE_A FILE_B FILE_C FILE_D FILE_E FILE_F FILE_G FILE_H FILE_NONE", start = 0)
RANKS = IntEnum("RANKS", "RANK_1 RANK_2 RANK_3 RANK_4 RANK_5 RANK_6 RANK_7 RANK_8 RANK_NONE", start = 0)
COLORS = IntEnum("COLORS", "WHITE BLACK BOTH")

# adding castling constant
# we can access the constants through CASTLING.WQCA or SQUARES.H7 for example
class CASTLING(IntEnum, start = 1):
    WKCA = 1; WQCA = 2; BKCA = 4; BQCA = 6;

class SQUARES(IntEnum, start = 21):
    A1 = 21; B1; C1; D1; E1; F1; G1; H1;
    A2 = 31; B2; C2; D2; E2; F2; G2; H2;
    A3 = 41; B3; C3; D3; E3; F3; G3; H3;
    A4 = 51; B4; C4; D4; E4; F4; G4; H4;
    A5 = 61; B5; C5; D5; E5; F5; G5; H5;
    A6 = 71; B6; C6; D6; E6; F6; G6; H6;
    A7 = 81; B7; C7; D7; E7; F7; G7; H7;
    A8 = 91; B8; C8; D8; E8; F8; G8; H8; NO_SQ;
    
    
# this should do the trick 
# board structure 

class BOARD():
    def __init__(self):
        # integer list that represents board state
        self.pieces = [0] * BOARD_SQUARE_NUMBER 
        # white black and both pawn positions
        # bits are 1 if there is a pawn and 0 if there is no pawn 
        self.pawns = [mybitarray([0] * 64), mybitarray([0] * 64),mybitarray([0] * 64)]
        # square numbers of both kings
        self.KingSquares = [0] * 2
        # current side to move
        self.side = [0]
        # en passant square
        self.enPassant = [0]
        # fifty move rule 
        self.fiftyMove = [0]
        # how many half moves are we into the current game
        self.ply = [0]
        # we store the board history in a list
        self.hisPly = [0]
        # position hash 
        # self.posKey = U64(False)
        self.poskey = mybitarray([0])
        # number of pieces on the board (12 different pieces + empty square)
        self.pceNum = [0] * 13
        # number of big pieces (Anything that is not a pawn) by color
        # major = Rooks, Queens
        # minor = Bishops, Knights
        self.bigPieces = [0] * 3
        self.majPieces = [0] * 3
        self.minPieces = [0] * 3
        # we can use 4 bit integer to represent castling permissions
        self.castlePerm = [0]
        # we can index the history with hisPly to get any point in the history
        self.history = [UNDO()] * MAXGAMEMOVES
        # piece list to speed up search later on
        self.pList = np.zeros([13,10])
        # adding a white knight o E1:
        # pLIST[PIECES.wN, 0] = SQUARES.E1
        # adding white knight to D4
        # pList[PIECES.wN, 1] = SQUARES.D4
        
        
# short structure that is going to be needed to construct the board history
class UNDO():
    def __init__(self):
        self.move = [0]
        self.castlePerm = [0]
        self.enPassant = [0]
        self.fiftyMove = [0]
        self.posKey = mybitarray([0])

# Array 120 to array 64 indexing is needed 
# The board has 120 squares so we can generate all possible moves easily
# and then check for their legality later. 
# the legal board squares are 21 to 98 but the indices for the pawn 
# representation run from 0 to 63
# hardcoding the array120 to array64 conversion
# these are going to be the lookup tables for conversion
Sq120ToSq64 = np.zeros(BOARD_SQUARE_NUMBER)
Sq64ToSq120 = np.zeros(64)

# quick function to convert file and rank numbers to board120 numbers
def filerank2square120(file, rank):
    sq = (21 + file) + (rank * 10)
    return(sq)

# function to fill the empty lookup tables
def initsquare120to64():
    square64 = 0
    Sq120ToSq64.fill(65)
    Sq64ToSq120.fill(120)
    for rank in range(RANKS.RANK_8 + 1):
        for file in range(FILES.FILE_H + 1):
            square = filerank2square120(file, rank)
            Sq64ToSq120[square64] = square
            Sq120ToSq64[square] = square64
            square64 = square64 + 1

# initializing the lookup tables
initsquare120to64()

# bitboards, pop, count, set, clear



# TODO: Bitboards Pop and Count
# TODO: Setting and clearing bits
# TODO: Position hashing
# TODO: Position setup
# TODO: Parse FEN notations (maybe for trainingsset)
# TODO: Parse opencv inputs from screenshots
# TODO: Webscrape match data
# TODO: Square attacked?
# TODO: Rank and File Arrays
# TODO: Move encoding and bit setting
# TODO: Move generation 
# TODO: Repetition detection 
# TODO: Selfplay
# TODO: Search
# TODO: Evaluation 
# TODO: Move ordering and picking
