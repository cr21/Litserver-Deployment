#!/bin/bash

# Create the main output directory
mkdir -p data/merged_food_101

# Loop through all food categories (using test directory as reference)
for category in data/food_101_data/test/*; do
    # Get the base category name
    category_name=$(basename "$category")
    
    # Create category directory in merged folder
    mkdir -p "data/merged_food_101/$category_name"
    
    # Copy images from test directory
    cp -r "data/food_101_data/test/$category_name"/* "data/merged_food_101/$category_name/"
    
    # Copy images from train directory
    cp -r "data/food_101_data/train/$category_name"/* "data/merged_food_101/$category_name/"
    
    # Copy images from val directory
    cp -r "data/food_101_data/val/$category_name"/* "data/merged_food_101/$category_name/"
done

echo "Data merging completed!"
