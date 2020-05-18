#!/usr/bin/sh

# Commands to make all stackplots

python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20/ -o plots_ee_2016/ -s "Channel == 1 && chi2 > 5 && chi2 < 30" -e 2016;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20/ -o plots_mumu_2016/ -s "Channel == 0 && chi2 > 5 && chi2 < 30" -e 2016;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20/ -o plots_ee_unblinded_2016/ -s "Channel == 1" -e 2016;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20/ -o plots_mumu_unblinded_2016/ -s "Channel == 0" -e 2016;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20_zPlus/ -o plots_ee_zPlus_2016/ -s "Channel == 1" -e 2016;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20_zPlus/ -o plots_mumu_zPlus_2016/ -s "Channel == 0" -e 2016;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2016/all/mz20mw20_emu/ -o plots_emu_2016/ -s "Channel == 2" -e 2016;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2017/all/mz20mw20/ -o plots_ee_2017/ -s "Channel == 1 && chi2 > 5 && chi2 < 30" -e 2017;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2017/all/mz20mw20/ -o plots_mumu_2017/ -s "Channel == 0 && chi2 > 5 && chi2 < 30" -e 2017;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2017/all/mz20mw20/ -o plots_ee_unblinded_2017/ -s "Channel == 1" -e 2017;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2017/all/mz20mw20/ -o plots_mumu_unblinded_2017/ -s "Channel == 0" -e 2017;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2017/all/mz20mw20_zPlus/ -o plots_ee_zPlus_2017/ -s "Channel == 1" -e 2017;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2017/all/mz20mw20_zPlus/ -o plots_mumu_zPlus_2017/ -s "Channel == 0" -e 2017;
python stackplots.py -i /scratch/data/TopPhysics/mvaDirs/inputs/2017/all/mz20mw20_emu/ -o plots_emu_2017/ -s "Channel == 2" -e 2017;
