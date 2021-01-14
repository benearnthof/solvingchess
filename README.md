# solvingchess
This project implements AlphaZero for lichess.com  
The general idea is as follows: 
* Detect the board state from a screenshot
* Encode the board state such that it is usable for inference
* Use AlphaZero to infer the best move 
* Execute the best move
* Repeat until the match is over

I'm going to try to implement everything from scratch. 
Currently only the detection of pieces works. 

Current todo list: 
* Add docstrings 
* Automate CV stuff to generate training data for board recognition
* Generate training data for board recognition
* Train board recognition 
* Webscrape match data
* Square attacked?
* Move encoding and bit setting
* Move generation
* Make Move
* Perft testing 
* Repetition detection 
* Selfplay
* Search
* Evaluation 
* Move ordering and picking
