# Hand-coding solutions for the ARC
_Originally forked from [The Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC).   
The “Abstraction and Reasoning Corpus” (ARC) of Chollet is an interesting new benchmark
for AI research.  
Please refer to the original repo for details about the ARC._

## Scope
This project is created as an assignment for _Programming and Tools for AI (CT5132/CT5148)_ course at [NUIG](http://www.nuigalway.ie/).


For this assignment, a set of tasks from the data/training
directory have been chosen to be solved by a hand-coded solution for each task.  

## Design Strategy
Even that the main focus of this assignment is on manual/hand-coded solutions , a balanced approach on the midway between specialization and generalization has been chosen, The approach taken here leans reuseability of small to medium sized set of general-ish and basic-ish operations (referred to here as OpKit, short for Operations Kit) which are basically an aggressively simplified set of image processing-ish operations (sorry, too many "-ish" in a paragraph :) ). Of course the word 'general' in 'general operations' is relative to the topic or the domain at hand (which is more or less some sort of image and basic shapes processing/manipulation), So consider these set of operations as a simple example of the Domain-Specific Language (DSL) referred to by Francois Chollet in 'On the Measure of Intelligence' (https://arxiv.org/abs/1911.01547).

## Problem Selection
At least 3 relatively easy problems, and at least 3 relatively hard ones, have been chosen, that is so just to test the hypothesis that the same set of operations (OpKit) would be compositable enough to solve problems on a wide range of complexity levels.
Hardness | Problems IDs
---|---
Relatively easy | 67e8384a, 2013d3e2, 6d75e8bb
Relatively hard | 5ad4f10b, c8cbb738, 681b3aeb