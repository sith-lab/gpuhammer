bash $HAMMER_ROOT/util/init_cuda.sh 1800 7600

# Variables
num_agg=24
num_warp=8
num_thread=3
round=1

row_step=4
skip_step=4
count_iter=10

# Constant values
addr_step=256
iterations=46000
mem_size=50465865728
num_rows=64169

banks=(256 2048 2048 2048 5120 6400 6400 6400)
flips=(30329 3543 13057 23029 4371 13635 21801 28498)
delays=(55 58 58 58 58 57 57 57)
vic_pos=(4 0 6 7 0 6 0 6)
vic_num=(1 0 0 1 0 0 0 0)
flip_names=(A1 B1 B2 B3 C1 D1 D2 D3)

# pats=(00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff)
pats=(55 aa)

dirname=$HAMMER_ROOT/results/sample_bitflips

for i in {0..7}; do
    
    bank_offset=${banks[$i]}
    flip_row=${flips[$i]}
    delay=${delays[$i]}
    flip_name=${flip_names[$i]}

    min_rowid=$((flip_row - 94))
    max_rowid=$((flip_row + 5))

    mkdir -p $dirname/$flip_name

    echo "Start hammering bank $bank_offset row $flip_row"

    for j in {0..1}; do

        k=$((1-j))
        vic_pat=0x${pats[$j]}
        agg_pat=0x${pats[$k]}

        expected=$(( (vic_pat >> vic_pos[i]) & 1 ))
        if [ $expected -ne ${vic_num[$i]} ]; then
            # echo "- Skip   Victim: $vic_pat"
            continue
        fi

        # File paths
        rowset_file="$HAMMER_ROOT/results/row_sets/ROW_SET_${bank_offset}.txt"
        log_file="$dirname/$flip_name/${num_agg}agg_b${bank_offset}_flip${flip_row}_${pats[$j]}${pats[$k]}.log"
        bitflip_file="$dirname/$flip_name/${num_agg}agg_b${bank_offset}_count_flip${flip_row}_${pats[$j]}${pats[$k]}.txt"

        echo "- Victim Pattern: $vic_pat, Aggressor Pattern: $agg_pat"

        $HAMMER_ROOT/src/out/build/gpu_hammer $rowset_file $((num_agg - 1)) $addr_step $iterations $min_rowid $max_rowid $row_step $skip_step $mem_size $num_warp $num_thread $delay $round $count_iter $num_rows $vic_pat $agg_pat $bitflip_file > $log_file

        sleep 3

    done

done