#!/bin/bash

# The file to edit
FILE="training_script.sh"

# Extract the current value of --train_steps, accounting for the space between the argument and its value
# The awk command here splits the line at --train_steps and then prints the next field, which should be the value
CURRENT_STEPS=$(grep -- "--train_steps" $FILE | awk '{for(i=1;i<=NF;i++) if ($i=="--train_steps") {print $(i+1); exit}}')

# Check if we found the current steps, if not exit
if [ -z "$CURRENT_STEPS" ]; then
    echo "Could not find --train_steps in $FILE"
    exit 1
fi

# Add 5500 to the current value
NEW_STEPS=$((CURRENT_STEPS + 5500))

# Replace the old value with the new value in the file
# This uses a more complex sed pattern to account for the space and ensures only digits following --train_steps are matched
sed -i "s/--train_steps \([0-9]*\)/--train_steps $NEW_STEPS/" $FILE

sbatch_output=$(/usr/local/bin/sbatch $FILE 2>&1)
echo "$sbatch_output"
