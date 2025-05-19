for val in 0 256 2048 5120 6400; do
  python3 $HAMMER_ROOT/util/run_timing_task.py conf_set --range $((47 * (1 << 30))) --size $((47 * (1<<30))) --it 15 --step 256 --threshold 27 --file $HAMMER_ROOT/results/row_sets/CONF_SET_$val.txt --trgtBankOfs $val
  sleep 3s
  python3 $HAMMER_ROOT/util/run_timing_task.py row_set --size $((47 * (1<<30))) --it 15 --threshold 27 --trgtBankOfs $val --outputFile $HAMMER_ROOT/results/row_sets/ROW_SET_$val.txt $HAMMER_ROOT/results/row_sets/CONF_SET_$val.txt
  sleep 3s
done