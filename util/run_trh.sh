bash "$HAMMER_ROOT/util/init_cuda.sh" 1800 7600

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

dum_rowid=1000

banks=(256 2048 2048 2048 5120 6400 6400 6400)
flips=(30329 3543 13057 23029 4371 13635 21801 28498)
delays=(55 58 58 58 58 57 57 57)
vic_pos=(4 0 6 7 0 6 0 6)
vic_num=(1 0 0 1 0 0 0 0)

# pats=(00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff)
pats=(55 aa)

for i in {0..7}; do
# i=0
# agg_period=55
# dum_period=17

    bank_offset=${banks[$i]}
    flip_row=${flips[$i]}
    delay=${delays[$i]}

    folder="$HAMMER_ROOT/results/trh/${i}"
    mkdir -p $folder

    min_rowid=$((flip_row - 94))
    max_rowid=$((flip_row + 5))

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

        for agg_period in {1..10}; do
            for ((dum_period=1; dum_period<agg_period; dum_period++)); do

                # File paths
                rowset_file="$HAMMER_ROOT/results/row_sets/ROW_SET_${bank_offset}.txt"
                log_file="$folder/agg${agg_period}_dum${dum_period}.log"

                bitflip_file="$folder/agg${agg_period}_dum${dum_period}_bitflip.txt"

                echo "- Hammer Victim: $vic_pat, Aggressor: $agg_pat"

                date > $log_file
                $HAMMER_ROOT/src/out/build/trh  $rowset_file $((num_agg - 1)) $addr_step $iterations $min_rowid $max_rowid $dum_rowid $row_step $skip_step $mem_size $num_warp $num_thread $delay $round $count_iter $num_rows $vic_pat $agg_pat $agg_period $dum_period $bitflip_file >> $log_file
                date >> $log_file

                sleep 3
            done
        done
    done
done