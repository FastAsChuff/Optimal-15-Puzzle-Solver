# Optimal-15-Puzzle-Solver
Optimal 15 Puzzle Solver With Arbitrary Start And End Board States

fifteen3.c
==========

This program find an optimally short path from two arbitrary board states of the famous 'fifteen' sliding square puzzle, if there is a solution, and a 'not reachable' message otherwise. The boards are input as arguments with hex values for the squares and the empty square is zero. The normally solved board state (endboard) is 123456789abcdef0. It uses an Iterative Deepening Depth Limited Search method using the maximum of Linear Conflicts and Manhatten Distance heuristics. Random board pairs will probably be solved in seconds, but there are some problematic state pairs which can not be solved in reasonable time with this algorithm/heuristic. This program's run-time data memory requirement is much less than 1MB.

Copyright:- Simon Goater July 2024

Usage:- ./fifteen3.bin startboard endboard

e.g. ./fifteen3.bin fe169b4c0a73d852 123456789abcdef0

fifteen9b.c
===========

This program performs an exhaustive search and finds an optimally short path from two arbitrary board states of the famous 'fifteen' sliding square puzzle, if there is a solution, and outputs a 'not reachable' message otherwise. The boards are input as arguments with hex values for the squares and the empty square is zero. The normally solved board state (endboard) is 123456789abcdef0. It uses an Iterative Deepening Depth Limited Search method using the maximum of Linear Conflicts and an additive 7-7-1+0 pattern database heuristics.
The pattern database is generated by this program and typically takes up to an hour if it has not been saved previuosly, otherwise, it will be loaded from file. Random board pairs will probably be solved just seconds after the database is built/loaded. Difficult inputs will likely be solved in minutes, but extreme cases may take up to an hour or so. Exhaustive search is computationally expensive, but the results are guaranteed to be optimal, meaning there does not exist a shorter solution than the one returned. Relaxing the requirement for optimality can massively reduce the time to find solutions, but this program is for strictly optimal solutions only, and is optimised for difficult inputs (60+ steps). This program's run-time data memory requirement is approx. 2GB. It also requires approx. 3 GB of storage space.

Copyright:- Simon Goater August 2024

Usage:- ./fifteen9b.bin startboard endboard

e.g. ./fifteen9b.bin 43218765cba90efd 5248a03ed6bc1f97

