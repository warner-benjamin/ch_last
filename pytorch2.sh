python train.py --channels_last False --subclass False --pt_compile False
sleep 10
python train.py --channels_last True --subclass False --pt_compile False
sleep 10
python train.py --channels_last True --subclass False --pt_compile True
sleep 10
python train.py --channels_last False --subclass True --pt_compile False
sleep 10
python train.py --channels_last True --subclass True --pt_compile False
sleep 10
python train.py --channels_last True --subclass True --pt_compile True