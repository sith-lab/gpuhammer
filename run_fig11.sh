# Variables
agg_rowid=30327
bank_id=1

row_step=4
count_iter=50

# Constant values
addr_step=256
iterations=91000
mem_size=50465865728

num_rows=64169      # Number of rows in the row_set (line number - 1)
vic_pat=55          # Victim row data pattern in hex
agg_pat=aa          # Aggressor row data pattern in hex

aggs=(24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8)
warps=(8  8  8  8 10 10  9  9  8  8  7  7  6  6 10 10 8)
threads=(3 3 3  3  2  2  2  2  2  2  2  2  2  2  1 1 1)
delays=(56 56 56 56 56 56 56 56 56 56 56 56 56 56 56 56 56 56)

for ((i=0; i<16; i++)); do
  bank_id=${aggs[$i]}
  num_warp=${warps[$i]}
  num_thread=${threads[$i]}
  delay=${delays[$i]}
  
  # File paths
  mkdir -p "$HAMMER_ROOT/results/fig11/raw_data"
  rowset_file="$HAMMER_ROOT/results/row_sets/ROW_SET_${bank_id}.txt"
  log_file="$HAMMER_ROOT/results/fig11/raw_data/${num_agg}agg_b${bank_id}.log"
  bitflip_file="$HAMMER_ROOT/results/fig11/raw_data/${num_agg}agg_b${bank_id}_count.txt"

  echo Start hammering $num_agg -sided patterns ...

  $HAMMER_ROOT/src/out/build/trr_sampler $rowset_file $((num_agg - 1)) $addr_step $iterations $agg_rowid $row_step $mem_size $num_warp $num_thread $delay 1 $count_iter $num_rows $vic_pat $agg_pat $bitflip_file > $log_file

  sleep 5
done
