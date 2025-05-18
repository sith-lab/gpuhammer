# Variables
bank_offset=6400
num_agg=24
num_warp=8
num_thread=3
delay=58
victim_row=13635
min_rowid=$(($victim_row - 94))
max_rowid=$victim_row
row_step=4
count_iter=100
addr_step=256
iterations=91000

mkdir -p $HAMMER_ROOT/results/fig12_t4/D1
for model in alexnet vgg resnet dense inception; do
    echo "Processing $model"

    >  "$HAMMER_ROOT/results/fig12_t4/D1/${model}.txt"
    shuf --random-source=<(yes 42) -i 0-4980 -n 50 | while read num; do

        mem_size=$((10723400960 - (256 * $num)))
        printf "$num: ***********************\n"

        rowset_file="$HAMMER_ROOT/results/row_sets/ROW_SET_${bank_offset}.txt"

        echo "Start hammering ..."
        nohup $HAMMER_ROOT/src/out/build/hammer_mem_manage $((46 * (2 ** 30))) >/dev/null 2>&1 &
        sleep 2
        nohup $HAMMER_ROOT/src/out/build/hammer_manual_agg_left $rowset_file $((num_agg - 1)) $addr_step $iterations $min_rowid $max_rowid $row_step $mem_size $num_warp $num_thread $delay 1 $count_iter >/dev/null 2>&1 &
        sleep 5
        python3 $HAMMER_ROOT/util/run_imagenet_models.py $model att D1 $HAMMER_ROOT/results/fig12_t4/ $HAMMER_ROOT/src/out/build/liballoc.so "$HAMMER_ROOT/results/fig12_t4/D1/${model}.txt"
        sleep 5
        > memory_control.txt
        sleep 5
        echo "Hammering done."

    done
done

# rm exploit_control.txt memory_control.txt model_control.txt
