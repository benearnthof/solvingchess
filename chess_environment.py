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
from aenum import Enum
# a = bitarray(int(64))
# a.setall(False)
def U64(init) :
    ret = bitarray(64)
    ret.setall(init)
    return(ret)

# global constants
BOARD_SQUARE_NUMBER = 120
PIECES = Enum("PIECES", "EMPTY wP wN wB wR wQ wK bP bN bB bR bQ bK", start = 0)
FILES = Enum("FILES", "FILE_A FILE_B FILE_C FILE_D FILE_E FILE_F FILE_G FILE_H FILE_NONE", start = 0)
RANKS = Enum("RANKS", "RANK_1 RANK_2 RANK_3 RANK_4 RANK_5 RANK_6 RANK_7 RANK_8 RANK_NONE", start = 0)
COLORS = Enum("COLORS", "WHITE BLACK BOTH")

class SQUARES(Enum, start = 21):
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

class S_BOARD():
    def __init__(self):
        # integer list that represents board state
        self.pieces = [0] * BOARD_SQUARE_NUMBER 
        # white black and both pawn positions
        # bits are 1 if there is a pawn and 0 if there is no pawn 
        self.pawns = [U64(False), U64(False), U64(False)]
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
        self.posKey = U64(False)
        # number of pieces on the board (12 different pieces + empty square)
        self.pceNum = [0] * 13
        # number of big pieces (Anything that is not a pawn) by color
        # major = Rooks, Queens
        # minor = Bishops, Knights
        self.bigPieces = [0] * 3
        self.majPieces = [0] * 3
        self.minPieces = [0] * 3
        

# TODO: undo move structure