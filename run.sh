#!/bin/bash

# This script runs the Python script with arguments for framework and type

# Get the arguments
framework="pytorch" # or, "tensorflow"
type="letter_by_letter" # or, "word_by_word"

# Run the Python script with the provided arguments
python main.py --framework "$framework" --type "$type"