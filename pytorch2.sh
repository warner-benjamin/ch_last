python train.py
sleep 10
python train.py --channels_last True
sleep 10
python train.py --channels_last True --pt_compile True
sleep 10
python train.py --subclass True
sleep 10
python train.py --channels_last True --subclass True
sleep 10
python train.py --channels_last True --subclass True --pt_compile True