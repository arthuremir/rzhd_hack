# rzhd_hack

Контакт - telegram @aemirov

Сборка докера:
``` 
cd src/mmsegmentation/docker
docker build . -t hack
```
Запуск докера:
```
docker run -u $(id -u):$(id -g) --shm-size 32G --gpus '"device=0"' --log-driver=none -v /etc/passwd:/etc/passwd -v /path/to/code/:/mmsegmentation/ -it hack /bin/bash
```

Настройка среды внутри докера:
```
cd src/mmsegmentation/
./set.sh
```

Ожидаемая структура каталога данных:
```
-- data
   -- test [1000 изображений]
   -- train
       -- images [8203 изображений]
       -- mask [маски]
```

Подготовка данных:
```
python3 prep_data.py
python3 split_masks.py
```

Запуск обучения сетей:
```
cd src/mmsegmentation/
./tools/dist_train.sh configs/mine/upernet_convnext_xlarge_trains.py 1 --work-dir work_dirs/upernet_convnext_xlarge_trains
./tools/dist_train.sh configs/mine/upernet_convnext_xlarge_rails.py 1 --work-dir work_dirs/upernet_convnext_xlarge_rails
```

Тестирование:
```
python3 tools/test.py configs/mine/upernet_convnext_xlarge_trains.py work_dirs/upernet_convnext_xlarge_trains/best_mIoU_iter_SOME_ITER.pth --opacity 1 --show-dir work_dirs/upernet_convnext_xlarge_trains/pred_maps_trains_tta
python3 tools/test.py configs/mine/upernet_convnext_xlarge_rails.py work_dirs/upernet_convnext_xlarge_rails/best_mIoU_iter_SOME_ITER.pth --opacity 1 --show-dir work_dirs/upernet_convnext_xlarge_rails/pred_maps_rails_tta
```

Объединение предсказаний и подготовка архива:
```
python3 postprocess_merge.py
```
