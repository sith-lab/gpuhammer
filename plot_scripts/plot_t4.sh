output_file="$HAMMER_ROOT/results/fig12_t4/t4.csv"

echo "Bitflip,Model,Top1,Top5,RAD" > "$output_file"

# Iterate over each folder
cd $HAMMER_ROOT/results/fig12_t4/
for folder in D1 D3 B1 B2; do
    # Iterate over each file in the folder
    for file in "$folder"/*.txt; do
        # Extract the row with the highest value in the third column
        max_row=$(awk -F',' 'NR == 1 || $3 > max { max = $3; line = $0 } END { print line }' "$file")
        
        # Append result to CSV file
        echo "$folder,$(basename "$file" .txt),$max_row" >> "$output_file"
    done
done
