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

