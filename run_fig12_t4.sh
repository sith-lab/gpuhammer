cd $HAMMER_ROOT/results/fig12_t4
rm -rf val
mkdir val
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar -C val
rm ILSVRC2012_img_val.tar
python3 $HAMMER_ROOT/util/filter_validation_set.py

bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_B1.sh
bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_B2.sh
bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_D1.sh
bash $HAMMER_ROOT/data_scripts/fig12_t4/run_hammer_manual_D3.sh
rm exploit_control.txt memory_control.txt model_control.txt

bash $HAMMER_ROOT/plot_scripts/plot_t4.sh
python3 $HAMMER_ROOT/plot_scripts/plot_fig12.py