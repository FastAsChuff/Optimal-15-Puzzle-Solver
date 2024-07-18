# Optimal-15-Puzzle-Solver
Optimal 15 Puzzle Solver With Arbitrary Start And End Board States

This program find an optimally short path from two arbitrary board states of the famous 'fifteen' sliding square puzzle, if there is a solution, and a 'not reachable' message otherwise. The boards are input as arguments with hex values for the squares and the empty square is zero. The normally solved board state (endboard) is 123456789abcdef0. It uses an Iterative Deepening Depth Limited Search method using the maximum of Linear Conflicts and Manhatten Distance heuristics. Random board pairs will probably be solved in seconds, but there are some problematic state pairs which can not be solved in reasonable time with this algorithm/heuristic. This program's run-time data memory requirement is much less than 1MB.

Copyright:- Simon Goater July 2024

Usage:- ./fifteen3.bin startboard endboard

e.g. ./fifteen3.bin fe169b4c0a73d852 123456789abcdef0
