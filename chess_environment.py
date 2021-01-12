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
# most of the code to set up the chess environment is taken from the tutorial series
# we can use the bitarray library to define 64 bit integers 
from bitarray import bitarray
# from aenum import Enum
from aenum import IntEnum
import numpy as np
from copy import copy
import sys
# import struct
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
COLORS = IntEnum("COLORS", "WHITE BLACK BOTH", start = 0)
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# lookup tables for piecelists and hardcoded values
PieceBig = np.array([0,0,1,1,1,1,1,0,1,1,1,1,1])
PieceMaj = np.array([0,0,0,0,1,1,1,0,0,0,1,1,1])
PieceMin = np.array([0,0,1,1,0,0,0,0,1,1,0,0,0])
PieceVal = np.array([0, 100, 325, 325, 550, 1000, 50000,100, 325, 325, 550, 1000, 50000])
PieceCol = np.array([COLORS.BOTH, COLORS.WHITE, COLORS.WHITE, COLORS.WHITE, COLORS.WHITE, COLORS.WHITE, COLORS.WHITE, 
                     COLORS.BLACK, COLORS.BLACK, COLORS.BLACK, COLORS.BLACK, COLORS.BLACK, COLORS.BLACK])

# adding castling constant
# we can access the constants through CASTLING.WQCA or SQUARES.H7 for example
class CASTLING(IntEnum, start = 1):
    WKCA = 1; WQCA = 2; BKCA = 4; BQCA = 8;
# we simply call CASTLING with an index in 1 2 4 8 
# and can fight out total castling rights by bitwise or

class SQUARES(IntEnum, start = 21):
    A1 = 21; B1; C1; D1; E1; F1; G1; H1;
    A2 = 31; B2; C2; D2; E2; F2; G2; H2;
    A3 = 41; B3; C3; D3; E3; F3; G3; H3;
    A4 = 51; B4; C4; D4; E4; F4; G4; H4;
    A5 = 61; B5; C5; D5; E5; F5; G5; H5;
    A6 = 71; B6; C6; D6; E6; F6; G6; H6;
    A7 = 81; B7; C7; D7; E7; F7; G7; H7;
    A8 = 91; B8; C8; D8; E8; F8; G8; H8; NO_SQ;
    OFFBOARD = 100;
    
# filerank arrays to lookup filerank of boardsquarenumbers    
FilesBrd = np.zeros(BOARD_SQUARE_NUMBER, dtype = int)
RanksBrd = np.zeros(BOARD_SQUARE_NUMBER, dtype = int)

def initfilerankarrays():
    FilesBrd[:] = SQUARES.OFFBOARD
    RanksBrd[:] = SQUARES.OFFBOARD
    for rank in range(RANKS.RANK_1, RANKS.RANK_8 + 1, 1):
        for file in range(FILES.FILE_A, FILES.FILE_H + 1, 1):
            square = filerank2square120(file, rank)
            FilesBrd[square] = file
            RanksBrd[square] = rank

# helper function to verify that filerankarrays have been initialized correctly    
def printfileranksboard(filesboard):
    for i in range(0, BOARD_SQUARE_NUMBER, 1):
        if i%10 == 0 and i != 0: printf("\n")
        printf("%4d", filesboard[i])
        
# this should do the trick 
# util for board init
def fillempty(pieces):
    index = range(0, 64)
    index = sq120(index)
    pieces[index] = PIECES.EMPTY
    return(pieces)

# board structure 
# resetboard is equal to init empty board 
class BOARD():
    def __init__(self):
        # integer list that represents board state
        # self.pieces = [0] * BOARD_SQUARE_NUMBER 
        self.pieces = np.zeros(BOARD_SQUARE_NUMBER, dtype = int)
        self.pieces.fill(SQUARES.OFFBOARD)
        self.pieces = fillempty(self.pieces)
        # white black and both pawn positions
        # bits are 1 if there is a pawn and 0 if there is no pawn 
        self.pawns = [mybitarray([0] * 64), mybitarray([0] * 64),mybitarray([0] * 64)]
        # square numbers of both kings
        self.KingSquares = np.array([SQUARES.NO_SQ, SQUARES.NO_SQ])
        # current side to move
        self.side = COLORS.BOTH
        # en passant square
        self.enPassant = SQUARES.NO_SQ
        # fifty move rule 
        self.fiftyMove = 0
        # how many half moves are we into the current game
        self.ply = 0
        # we store the board history in a list
        self.hisPly = 0
        # position hash 
        # self.posKey = U64(False)
        self.poskey = np.uint64()
        # number of pieces on the board (12 different pieces + empty square)
        self.pceNum = np.zeros(13, dtype = int)
        # number of big pieces (Anything that is not a pawn) by color
        # major = Rooks, Queens
        # minor = Bishops, Knights
        self.bigPieces = np.zeros(2, dtype = int)
        self.majPieces = np.zeros(2, dtype = int)
        self.minPieces = np.zeros(2, dtype = int)
        # this is the material evaluation, will not be needed for alpha zero
        # and should require floats i think, we'll see later
        self.material = np.zeros(2, dtype = int)
        # we can use 4 bit integer to represent castling permissions
        self.castlePerm = 0
        # we can index the history with hisPly to get any point in the history
        self.history = [UNDO()] * MAXGAMEMOVES
        # piece list to speed up search later on
        self.pList = np.zeros([13,10])
        # adding a white knight o E1:
        # pLIST[PIECES.wN, 0] = SQUARES.E1
        # adding white knight to D4
        # pList[PIECES.wN, 1] = SQUARES.D4
    def updatelistsmaterial(self):
        for i in range(0, BOARD_SQUARE_NUMBER, 1):
            square = i
            piece = self.pieces[i]
            if piece != SQUARES.OFFBOARD and piece != PIECES.EMPTY:
                color = PieceCol[piece]
                if PieceBig[piece] == True: self.bigPieces[color] += 1
                if PieceMin[piece] == True: self.minPieces[color] += 1
                if PieceMaj[piece] == True: self.majPieces[color] += 1
                self.material[color] += PieceVal[piece]
                # piece lists
                self.pList[piece, self.pceNum[piece]] = square
                self.pceNum[piece] += 1
                # setting king squares
                if piece == PIECES.wK: self.KingSquares[COLORS.WHITE] = square
                if piece == PIECES.bK: self.KingSquares[COLORS.BLACK] = square
                # setting pawn bitboards
                if piece == PIECES.wP:
                    setbit(self.pawns[COLORS.WHITE], sq64(square))
                    setbit(self.pawns[COLORS.BOTH], sq64(square))
                elif piece == PIECES.bP:
                    setbit(self.pawns[COLORS.BLACK], sq64(square))
                    setbit(self.pawns[COLORS.BOTH], sq64(square))
        
# short structure that is going to be needed to construct the board history
# TODO: rewrite this as method for board class
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
Sq120ToSq64 = np.zeros(BOARD_SQUARE_NUMBER, dtype = int)
Sq64ToSq120 = np.zeros(64, dtype = int)

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

# utils to make arrays callable
def sq64(square):
    return Sq120ToSq64[square]

def sq120(square):
    return Sq64ToSq120[square]

# util to convert bitarray to int
def ba2int(ba):
    return int.from_bytes(ba, byteorder = ba.endian())

# util to get correct print behavior
def printf(format, *args):
    sys.stdout.write(format % args)
    
# util to get a pseudo 1ULL
def ull1(): 
    ret = mybitarray([0]*64)
    ret[63] = 1
    return ret
    
# bitboard print
def printbitboard(bitboard = mybitarray([0]*64)):
    shiftme = ull1()
    printf("\n")
    for rank in range(RANKS.RANK_8, -1, -1):
        for file in range(FILES.FILE_A, 8, 1):
            square = filerank2square120(file, rank) # 120 based
            square64 = Sq120ToSq64[square]          # 64 based
            # we can check if a square is occupied by looping through bitshifts
            # and a bitwise and with the bitboard to check
            if (sum((shiftme << square64) & bitboard)):
                printf("X")
            else:
                printf("O")
        printf("\n")
    printf("\n\n")
    
# util to check board setup - needs to be expanded in the future
def printboard(board = BOARD()):
    for i in range(0, 120, 1):
        if board.pieces[i] == SQUARES.OFFBOARD:
            printf("\0")
        elif board.pieces[i] == PIECES.EMPTY:
            printf(".")
        elif board.pieces[i] == PIECES.bB:
            printf("b")
        elif board.pieces[i] == PIECES.bK:
            printf("k")
        elif board.pieces[i] == PIECES.bN:
            printf("n")
        elif board.pieces[i] == PIECES.bP:
            printf("p")
        elif board.pieces[i] == PIECES.bQ:
            printf("q")
        elif board.pieces[i] == PIECES.bR:
            printf("r")
        elif board.pieces[i] == PIECES.wB:
            printf("B")
        elif board.pieces[i] == PIECES.wK:
            printf("K")
        elif board.pieces[i] == PIECES.wN:
            printf("N")
        elif board.pieces[i] == PIECES.wP:
            printf("P")
        elif board.pieces[i] == PIECES.wQ:
            printf("Q")
        elif board.pieces[i] == PIECES.wR:
            printf('R')
        else:
            print("default")
        if i % 10 == 9:
            printf("\n")

# other printboard function that rotates the board accordingly
def printboard2(board):
    piecestr = ".PNBRQKpnbrqk"
    sidestr = "wb-"
    # rankstr = "12345678"
    # filestr = "abcdefgh"
    for rank in range(RANKS.RANK_8, RANKS.RANK_1-1, -1):
        printf("%d  ", (rank + 1))
        for file in range(FILES.FILE_A, FILES.FILE_H+1, 1):
            square = filerank2square120(file, rank)
            piece = board.pieces[square]
            printf("%3c", piecestr[piece])
        printf("\n")
    for file in range(FILES.FILE_A, FILES.FILE_H+1, 1):
        if file == 0: printf("   ")
        printf("%3c",(chr(ord('a') + file)))
    printf("\n")
    printf("Side: %c\n", sidestr[board.side])
    printf("En Passant: %d\n", board.enPassant)
    printf("Castle: %c%c%c%c\n", 
           "K" if board.castlePerm & CASTLING.WKCA else "-",
           "Q" if board.castlePerm & CASTLING.WQCA else "-",
           "k" if board.castlePerm & CASTLING.BKCA else "-",
           "q" if board.castlePerm & CASTLING.BQCA else "-",
           )
    printf("Position Key: %11x\n", board.poskey)

# trying out if everything works accordingly
pbb = mybitarray([0] * 64)
printbitboard(pbb)
pbb |= (ull1() << sq64(SQUARES.D2))
printbitboard(pbb)
pbb |= (ull1() << sq64(SQUARES.G2))
printbitboard(pbb)

# bitboard pop and count
def countbits(bitboard = mybitarray([0]*64)): 
    return sum(bitboard)

# removes the first nonzero bit and returns its index
def popbit(bitboard = mybitarray([0]*64)): 
    temp = copy(bitboard)
    temp.reverse()
    index_r = temp.index(1)
    temp[index_r] = 0
    temp.reverse()
    return([index_r, temp])

# trying out if everything works accordingly
pbb = mybitarray([0] * 64)
pbb |= (ull1() << sq64(SQUARES.D2))
pbb |= (ull1() << sq64(SQUARES.D3))
pbb |= (ull1() << sq64(SQUARES.D4))
printbitboard(pbb)
pbb = popbit(pbb)
printbitboard(pbb[1])
pbb = popbit(pbb[1])
printbitboard(pbb[1])

# Setting and clearing bits
def setbit(bitboard = mybitarray([0]*64), index = 0):
    # i think this is needed
    index = 63 - index
    bitboard[index] = 1
    return bitboard

def clearbit(bitboard = mybitarray([0]*64), index = 0):
    index = 63 - index
    bitboard[index] = 0
    return bitboard

# testing if everything works
bitboard = mybitarray([0]*64)
setbit(bitboard, 61)
printbitboard(bitboard)
clearbit(bitboard, 61)
printbitboard(bitboard)

# Position hashing
# =============================================================================
# piecekeys = np.zeros((13, 120))
# sidekey = 0
# castlekeys = np.zeros(16)
# =============================================================================

# initializing hash keys 
def inithashkeys(seed = 0):
    np.random.seed(seed)
    piecekeys = np.random.randint(2**63, size = (13, 120), dtype = np.uint64)
    sidekey = np.random.randint(2**63, size = 1, dtype = np.uint64)
    castlekeys = np.random.randint(2**63, size = 16, dtype = np.uint64)
    ret = {
        "pkey": piecekeys,
        "skey": sidekey, 
        "ckey": castlekeys
        }
    return(ret)

# the idea is to initialize the hashkeys as tables of random integers at the 
# start of the game and then encode the current board state with "bitwise or"
# generating position key
def generateposkey(board, hashkeys):
    finalkey = np.uint64()
    piece = PIECES.EMPTY
    for square in range(0, BOARD_SQUARE_NUMBER, 1):
        piece = board.pieces[square]
        if piece != SQUARES.NO_SQ and piece != PIECES.EMPTY and piece != SQUARES.OFFBOARD:
            assert piece >= PIECES.wP and piece <= PIECES.bK
            finalkey ^= hashkeys["pkey"][piece, square]
    if board.side == COLORS.WHITE:
        finalkey ^= hashkeys["skey"][0]
    if board.enPassant != SQUARES.NO_SQ:
        assert board.enPassant >= 0 and board.enPassant 
        finalkey ^= hashkeys["pkey"][PIECES.EMPTY, board.enPassant]
    assert board.castlePerm >= 0 and board.castlePerm <= 15
    finalkey ^= hashkeys["ckey"][board.castlePerm]
    return(finalkey)
            
# testing positionkey generation with emptyboard and testhashkeys
testboard = BOARD()
testkeys = inithashkeys()
testposkey = generateposkey(testboard, testkeys)
        
# Position setup & FEN parsing
# https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
def parsefen (fen = START_FEN):
    assert type(fen) == str
    board = BOARD()
    rank = RANKS.RANK_8
    file = FILES.FILE_A
    # piece = 0
    # get the side and split the board setup from the rest of the info
    if " w " in fen:
        head = fen.split(" w ")[0]
        tail = fen.split(" w ")[1]
        board.side = COLORS.WHITE
    else: 
        head = fen.split(" b ")[0]
        tail = fen.split(" b ")[1]
        board.side = COLORS.BLACK
    # setup castling rights and rest of info    
    for i in range(0, 4, 1):
        if tail[i] == 'K': 
            board.castlePerm = board.castlePerm | CASTLING.WKCA
        elif tail[i] == 'Q': board.castlePerm = board.castlePerm | CASTLING.WQCA
        elif tail[i] == 'k': board.castlePerm = board.castlePerm | CASTLING.BKCA
        elif tail[i] == 'q': board.castlePerm = board.castlePerm | CASTLING.BQCA
        else: 
            print("default")
    # set en passant square
    tail = tail.split()[1:]
    if tail[0] != '-':
        # use ascii conversion to avoid conversion table
        enpasfile = ord(tail[0][0]) - ord('a')
        enpasrank = ord(tail[0][1]) - ord('1')
        board.enPassant = filerank2square120(enpasfile, enpasrank)
    # set 50 move rule
    board.fiftyMove = int(tail[1])
    # set ply
    board.ply = int(tail[2])
    # setup pieces
    index = 0
    while rank >= RANKS.RANK_1 and index < len(head):
        count = 1
        if head[index] == 'p': 
            piece = PIECES.bP
        elif head[index] == 'r': 
                piece = PIECES.bR
        elif head[index] == 'n': piece = PIECES.bN
        elif head[index] == 'b': piece = PIECES.bB
        elif head[index] == 'k': piece = PIECES.bK
        elif head[index] == 'q': piece = PIECES.bQ
        elif head[index] == 'P': piece = PIECES.wP
        elif head[index] == 'R': piece = PIECES.wR
        elif head[index] == 'N': piece = PIECES.wN
        elif head[index] == 'B': piece = PIECES.wB
        elif head[index] == 'K': piece = PIECES.wK
        elif head[index] == 'Q': piece = PIECES.wQ
        elif head[index] in '12345678': 
            piece = PIECES.EMPTY
            count = int(head[index])
        elif head[index] in " /":
            rank = rank - 1
            file = FILES.FILE_A
            index = index + 1
            continue
        else: 
            printf("FEN error \n")
            return(-1)
        for i in range(0, count, 1):
            square64 = rank * 8 + file
            square120 = sq120(square64)
            if piece != PIECES.EMPTY:
                board.pieces[square120] = piece
            file = file + 1
        index = index + 1
    # set poskey
    board.poskey = generateposkey(board, inithashkeys())
    return(board)

test = parsefen()
test.poskey
printboard(test)
test2 = parsefen("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2 ")
printboard(test2)
# everything seems to work!
# testing printboard2 function
printboard2(test2)
# it works indeed. last move is knight f3, position keys also work
# testing en passant square
test3 = parsefen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
printboard2(test3)

# testing piecelist updating function
test = parsefen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")

# testing by generating a random fen and printing the board
def generatefen():
    chars = list('1KQRBNPkqrbnp')
    pieces = np.random.choice(chars, 64)
    fen = '/'.join([''.join(pieces[i*8:(i+1)*8]) for i in range(8)])
    # fen = fen + ' b KQkq e3 0 1'
    return(fen)

# printing random boards for the fun of it
def funkyboards(n = 5):
    np.random.seed()
    temp = [generatefen() for i in range(0, n, 1)]
    boards = [parsefen(temp[j]) for j in range(0, n, 1)]
    for k in range(0, n, 1):
        printboard2(boards[k])    
       
# =============================================================================
#         board = parsefen(temp)
#         printboard2(board)
# =============================================================================

funkyboards()
# can we generate screenshots of the board from a random fen string?
# https://github.com/Elucidation/tensorflow_chessbot/blob/master/tensorflow_generate_training_data.ipynb
import PIL
import os
from IPython.display import Image, display
import feedparser
import subprocess

url = "http://en.lichess.org/editor/%s"% generatefen()
output_filename = "testA.png"

d = feedparser.parse(url)
link = d.entries[0].link
title = d.entries[0].title

def scrape_url(url, outpath):
    """
    Requires webkit2png to be on the path
    """
    subprocess.call(["webkit2png", "-o", outpath, "-g", "1000", "1260",
                     "-t", "30", url])

scrape_url(url, os.getcwd())

# TODO: Add docstrings 
# TODO: Parse opencv inputs from screenshots
# TODO: Webscrape match data
# TODO: Square attacked?
# TODO: Move encoding and bit setting
# TODO: Move generation
# TODO: Make Move
# TODO: Perft testing 
# TODO: Repetition detection 
# TODO: Selfplay
# TODO: Search
# TODO: Evaluation 
# TODO: Move ordering and picking
