# Projeto Final - Modelos Preditivos Conexionistas

### Nomes dos alunos:
- Erike Simon
- José Aparecido Silvano de Albuquerque

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|
|--|--|--|
|Classificação de Imagens|YOLOv8|PyTorch|

## Performance

O modelo treinado possui performance de **81.82%**.
![image](https://github.com/aparecidoSilvano/projeto-cesar-redes-neurais/assets/7593828/6a869e3e-d237-419c-adaa-2f70e9f0e48b)

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
    WARNING ⚠️ 'ultralytics.yolo.v8' is deprecated since '8.0.136' and will be removed in '8.1.0'. Please use 'ultralytics.models.yolo' instead.
WARNING ⚠️ 'ultralytics.yolo.utils' is deprecated since '8.0.136' and will be removed in '8.1.0'. Please use 'ultralytics.utils' instead.
Note this warning may be related to loading older models. You can update your model to current structure with:
    import torch
    ckpt = torch.load("model.pt")  # applies to both official and custom models
    torch.save(ckpt, "updated-model.pt")

Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-cls.pt to 'yolov8m-cls.pt'...
/content
100%|██████████| 32.7M/32.7M [00:00<00:00, 48.0MB/s]
New https://pypi.org/project/ultralytics/8.0.199 available 😃 Update with 'pip install -U ultralytics'
Ultralytics YOLOv8.0.186 🚀 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (Tesla T4, 15102MiB)
engine/trainer: task=classify, mode=train, model=yolov8m-cls.pt, data=/content/datasets/New-Animals-Classification-1, epochs=50, patience=50, batch=16, imgsz=64, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=None, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, vid_stride=1, stream_buffer=False, line_width=None, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=runs/classify/train3
train: /content/datasets/New-Animals-Classification-1/train... found 465 images in 4 classes: ERROR ❌️ requires 3 classes, not 4
val: None...
test: /content/datasets/New-Animals-Classification-1/test... found 22 images in 3 classes ✅ 
Overriding model.yaml nc=1000 with nc=3

                   from  n    params  module                                       arguments                     
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
  1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
  2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
  3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
  5                  -1  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
  6                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
  7                  -1  1   2655744  ultralytics.nn.modules.conv.Conv             [384, 768, 3, 2]              
  8                  -1  2   7084032  ultralytics.nn.modules.block.C2f             [768, 768, 2, True]           
  9                  -1  1    989443  ultralytics.nn.modules.head.Classify         [768, 3]                      
YOLOv8m-cls summary: 141 layers, 15776179 parameters, 15776179 gradients
Transferred 228/230 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir runs/classify/train3', view at http://localhost:6006/
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
AMP: checks passed ✅
train: Scanning /content/datasets/New-Animals-Classification-1/train... 475 images, 0 corrupt: 100%|██████████| 475/475 [00:00<?, ?it/s]
albumentations: RandomResizedCrop(p=1.0, height=64, width=64, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1), HorizontalFlip(p=0.5), ColorJitter(p=0.5, brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.30000000000000004, 1.7], hue=[-0.015, 0.015]), Normalize(p=1.0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0), ToTensorV2(always_apply=True, p=1.0, transpose_mask=False)
val: Scanning /content/datasets/New-Animals-Classification-1/test... 22 images, 0 corrupt: 100%|██████████| 22/22 [00:00<?, ?it/s]
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.000714, momentum=0.9) with parameter groups 38 weight(decay=0.0), 39 weight(decay=0.0005), 39 bias(decay=0.0)
Image sizes 64 train, 64 val
Using 2 dataloader workers
Logging results to runs/classify/train3
Starting training for 50 epochs...

      Epoch    GPU_mem       loss  Instances       Size
       1/50     0.679G     0.2584         11         64: 100%|██████████| 30/30 [00:09<00:00,  3.30it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00,  2.86it/s]
                   all      0.591          1

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 8.4ms
Speed: 11.6ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 14.7ms
Speed: 0.6ms preprocess, 14.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 11.9ms
Speed: 0.4ms preprocess, 11.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 7.8ms
Speed: 0.6ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 7.5ms
Speed: 1.2ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 9.7ms
Speed: 0.5ms preprocess, 9.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 8.2ms
Speed: 0.5ms preprocess, 8.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 5.8ms
Speed: 0.5ms preprocess, 5.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 8.0ms
Speed: 0.5ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 12.5ms
Speed: 0.5ms preprocess, 12.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 13.1ms
Speed: 0.4ms preprocess, 13.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       2/50     0.682G     0.1751         11         64: 100%|██████████| 30/30 [00:03<00:00,  7.63it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 27.26it/s]
                   all      0.545          1

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 5.9ms
Speed: 0.5ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 6.4ms
Speed: 0.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 10.5ms
Speed: 0.6ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 8.1ms
Speed: 0.5ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.4ms
Speed: 0.4ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 6.1ms
Speed: 0.4ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 14.2ms
Speed: 0.4ms preprocess, 14.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.0ms
Speed: 0.4ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 5.8ms
Speed: 0.5ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 5.4ms
Speed: 0.9ms preprocess, 5.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 5.8ms
Speed: 1.5ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       3/50     0.661G     0.1061         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.85it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 23.15it/s]
                   all      0.727          1

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 8.1ms
Speed: 0.6ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 6.3ms
Speed: 0.9ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 11.0ms
Speed: 0.5ms preprocess, 11.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 9.7ms
Speed: 0.5ms preprocess, 9.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.36, rhinoceros 0.35, hippopotamus 0.28, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 6.2ms
Speed: 0.6ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.36, rhinoceros 0.35, hippopotamus 0.28, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 7.7ms
Speed: 1.3ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 10.6ms
Speed: 0.5ms preprocess, 10.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 13.8ms
Speed: 0.5ms preprocess, 13.8ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.36, rhinoceros 0.35, hippopotamus 0.28, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 5.5ms
Speed: 0.7ms preprocess, 5.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 6.0ms
Speed: 0.9ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.36, rhinoceros 0.35, hippopotamus 0.28, 8.3ms
Speed: 0.5ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 5.5ms
Speed: 1.9ms preprocess, 5.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.37, rhinoceros 0.35, hippopotamus 0.28, 5.2ms
Speed: 0.4ms preprocess, 5.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       4/50     0.661G    0.06981         11         64: 100%|██████████| 30/30 [00:05<00:00,  5.86it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 21.43it/s]
                   all      0.727          1

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.2ms
Speed: 0.8ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 9.0ms
Speed: 0.4ms preprocess, 9.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.1ms
Speed: 0.4ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.1ms
Speed: 0.6ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 5.3ms
Speed: 0.4ms preprocess, 5.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 5.4ms
Speed: 0.6ms preprocess, 5.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 5.3ms
Speed: 0.3ms preprocess, 5.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 8.6ms
Speed: 0.6ms preprocess, 8.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.5ms
Speed: 2.5ms preprocess, 6.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.9ms
Speed: 0.4ms preprocess, 6.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.5ms
Speed: 0.8ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.1ms
Speed: 0.3ms preprocess, 6.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       5/50     0.661G    0.05651         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.63it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 29.59it/s]
                   all      0.591          1

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 14.6ms
Speed: 0.7ms preprocess, 14.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 12.4ms
Speed: 0.5ms preprocess, 12.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 11.1ms
Speed: 0.6ms preprocess, 11.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 10.9ms
Speed: 0.8ms preprocess, 10.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.3ms
Speed: 3.0ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.7ms
Speed: 3.2ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 11.9ms
Speed: 0.5ms preprocess, 11.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 5.4ms
Speed: 1.0ms preprocess, 5.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 9.8ms
Speed: 0.4ms preprocess, 9.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 5.4ms
Speed: 2.3ms preprocess, 5.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 5.6ms
Speed: 0.5ms preprocess, 5.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 9.4ms
Speed: 0.5ms preprocess, 9.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 5.7ms
Speed: 0.3ms preprocess, 5.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.9ms
Speed: 0.8ms preprocess, 6.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, elephant 0.37, hippopotamus 0.25, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       6/50     0.661G    0.05785         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.56it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 19.64it/s]
                   all      0.727          1

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 11.0ms
Speed: 0.5ms preprocess, 11.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 8.6ms
Speed: 0.4ms preprocess, 8.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 7.6ms
Speed: 0.4ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 8.8ms
Speed: 0.5ms preprocess, 8.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 10.3ms
Speed: 0.5ms preprocess, 10.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.29, 6.0ms
Speed: 1.6ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 8.0ms
Speed: 0.4ms preprocess, 8.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 10.0ms
Speed: 0.4ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 5.8ms
Speed: 1.7ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 6.5ms
Speed: 0.4ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 6.6ms
Speed: 0.4ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.35, rhinoceros 0.35, hippopotamus 0.30, 5.9ms
Speed: 0.6ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       7/50     0.661G    0.07967         11         64: 100%|██████████| 30/30 [00:04<00:00,  7.25it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 24.46it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 6.7ms
Speed: 5.3ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 8.2ms
Speed: 0.6ms preprocess, 8.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.40, elephant 0.37, hippopotamus 0.22, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 9.1ms
Speed: 0.6ms preprocess, 9.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.21, 6.9ms
Speed: 0.7ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 14.1ms
Speed: 0.5ms preprocess, 14.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.40, elephant 0.38, hippopotamus 0.22, 7.4ms
Speed: 0.4ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 10.0ms
Speed: 0.4ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 7.2ms
Speed: 0.4ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 6.6ms
Speed: 0.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 6.6ms
Speed: 0.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 9.8ms
Speed: 0.7ms preprocess, 9.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 12.1ms
Speed: 0.4ms preprocess, 12.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.40, elephant 0.38, hippopotamus 0.22, 6.1ms
Speed: 0.4ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.38, hippopotamus 0.22, 5.5ms
Speed: 0.4ms preprocess, 5.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 5.6ms
Speed: 1.8ms preprocess, 5.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 5.3ms
Speed: 0.4ms preprocess, 5.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, elephant 0.37, hippopotamus 0.22, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       8/50     0.661G    0.07363         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.69it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 42.12it/s]
                   all      0.682          1

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 10.5ms
Speed: 0.8ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.51, rhinoceros 0.33, hippopotamus 0.15, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.52, rhinoceros 0.33, hippopotamus 0.15, 5.8ms
Speed: 0.5ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 5.7ms
Speed: 0.5ms preprocess, 5.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.52, rhinoceros 0.33, hippopotamus 0.15, 7.8ms
Speed: 0.4ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.52, rhinoceros 0.33, hippopotamus 0.15, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.54, rhinoceros 0.32, hippopotamus 0.15, 6.1ms
Speed: 0.6ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.52, rhinoceros 0.32, hippopotamus 0.15, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 6.0ms
Speed: 0.4ms preprocess, 6.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.54, rhinoceros 0.32, hippopotamus 0.15, 5.5ms
Speed: 0.6ms preprocess, 5.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 5.4ms
Speed: 0.7ms preprocess, 5.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 8.7ms
Speed: 0.6ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.52, rhinoceros 0.33, hippopotamus 0.15, 5.5ms
Speed: 0.5ms preprocess, 5.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.53, rhinoceros 0.32, hippopotamus 0.15, 5.8ms
Speed: 1.8ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
       9/50     0.661G    0.08188         11         64: 100%|██████████| 30/30 [00:03<00:00,  7.65it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 14.88it/s]
                   all      0.545          1

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 14.2ms
Speed: 0.5ms preprocess, 14.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, hippopotamus 0.26, rhinoceros 0.17, 11.6ms
Speed: 0.5ms preprocess, 11.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.55, hippopotamus 0.27, rhinoceros 0.17, 14.7ms
Speed: 0.5ms preprocess, 14.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 12.6ms
Speed: 0.5ms preprocess, 12.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, hippopotamus 0.25, rhinoceros 0.18, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.55, hippopotamus 0.27, rhinoceros 0.18, 13.0ms
Speed: 0.5ms preprocess, 13.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 18.8ms
Speed: 0.5ms preprocess, 18.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 13.7ms
Speed: 0.5ms preprocess, 13.7ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 12.7ms
Speed: 0.5ms preprocess, 12.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, hippopotamus 0.25, rhinoceros 0.18, 14.4ms
Speed: 0.5ms preprocess, 14.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, hippopotamus 0.27, rhinoceros 0.17, 15.0ms
Speed: 0.5ms preprocess, 15.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 9.3ms
Speed: 0.4ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 7.1ms
Speed: 5.9ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, hippopotamus 0.25, rhinoceros 0.18, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, hippopotamus 0.25, rhinoceros 0.17, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 9.2ms
Speed: 0.4ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 6.4ms
Speed: 0.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, hippopotamus 0.26, rhinoceros 0.17, 8.0ms
Speed: 0.4ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.56, hippopotamus 0.26, rhinoceros 0.18, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.55, hippopotamus 0.27, rhinoceros 0.18, 11.6ms
Speed: 0.4ms preprocess, 11.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, hippopotamus 0.25, rhinoceros 0.18, 11.7ms
Speed: 0.4ms preprocess, 11.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      10/50     0.661G    0.05785         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.37it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 21.03it/s]
                   all      0.636          1

0: 64x64 elephant 0.72, rhinoceros 0.14, hippopotamus 0.14, 8.0ms
Speed: 0.7ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, hippopotamus 0.15, rhinoceros 0.15, 6.9ms
Speed: 0.4ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, hippopotamus 0.16, rhinoceros 0.15, 6.8ms
Speed: 0.7ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.15, hippopotamus 0.15, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.68, rhinoceros 0.16, hippopotamus 0.16, 12.3ms
Speed: 0.4ms preprocess, 12.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, hippopotamus 0.15, rhinoceros 0.15, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.16, hippopotamus 0.15, 12.1ms
Speed: 0.5ms preprocess, 12.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.71, hippopotamus 0.15, rhinoceros 0.14, 5.9ms
Speed: 0.5ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.71, rhinoceros 0.15, hippopotamus 0.14, 5.8ms
Speed: 0.5ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.16, hippopotamus 0.16, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, hippopotamus 0.15, rhinoceros 0.15, 5.9ms
Speed: 0.6ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, hippopotamus 0.16, rhinoceros 0.16, 8.7ms
Speed: 0.5ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.15, hippopotamus 0.15, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.15, hippopotamus 0.15, 16.1ms
Speed: 0.5ms preprocess, 16.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.72, rhinoceros 0.14, hippopotamus 0.14, 5.8ms
Speed: 0.5ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, hippopotamus 0.15, rhinoceros 0.15, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.16, hippopotamus 0.15, 5.3ms
Speed: 0.3ms preprocess, 5.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.16, hippopotamus 0.15, 5.5ms
Speed: 0.4ms preprocess, 5.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.71, hippopotamus 0.15, rhinoceros 0.14, 7.0ms
Speed: 2.5ms preprocess, 7.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.15, hippopotamus 0.15, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, hippopotamus 0.15, rhinoceros 0.15, 8.1ms
Speed: 0.6ms preprocess, 8.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.16, hippopotamus 0.15, 5.3ms
Speed: 0.3ms preprocess, 5.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      11/50     0.661G    0.03979         11         64: 100%|██████████| 30/30 [00:02<00:00, 10.07it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 27.86it/s]
                   all      0.727          1

0: 64x64 elephant 0.83, hippopotamus 0.10, rhinoceros 0.07, 9.0ms
Speed: 0.6ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.82, hippopotamus 0.10, rhinoceros 0.08, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.09, rhinoceros 0.08, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.09, rhinoceros 0.08, 9.3ms
Speed: 0.4ms preprocess, 9.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.09, rhinoceros 0.07, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.82, hippopotamus 0.10, rhinoceros 0.08, 6.5ms
Speed: 1.8ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.82, hippopotamus 0.10, rhinoceros 0.08, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.82, hippopotamus 0.10, rhinoceros 0.08, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, hippopotamus 0.11, rhinoceros 0.08, 6.0ms
Speed: 1.7ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.10, rhinoceros 0.08, 5.8ms
Speed: 1.8ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.10, rhinoceros 0.08, 5.7ms
Speed: 2.0ms preprocess, 5.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.82, hippopotamus 0.10, rhinoceros 0.08, 6.2ms
Speed: 1.8ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.82, hippopotamus 0.10, rhinoceros 0.08, 10.5ms
Speed: 2.1ms preprocess, 10.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.10, rhinoceros 0.08, 6.1ms
Speed: 1.6ms preprocess, 6.1ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.09, rhinoceros 0.08, 12.3ms
Speed: 0.4ms preprocess, 12.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, hippopotamus 0.10, rhinoceros 0.08, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.09, rhinoceros 0.07, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, hippopotamus 0.11, rhinoceros 0.08, 5.7ms
Speed: 1.7ms preprocess, 5.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.82, hippopotamus 0.10, rhinoceros 0.08, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, hippopotamus 0.10, rhinoceros 0.08, 5.7ms
Speed: 2.5ms preprocess, 5.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, hippopotamus 0.10, rhinoceros 0.09, 5.7ms
Speed: 0.7ms preprocess, 5.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.83, hippopotamus 0.10, rhinoceros 0.08, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      12/50     0.661G    0.04506         11         64: 100%|██████████| 30/30 [00:04<00:00,  7.14it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 18.74it/s]
                   all      0.727          1

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 7.4ms
Speed: 0.6ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 9.2ms
Speed: 0.5ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 9.5ms
Speed: 0.4ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.33, hippopotamus 0.32, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, elephant 0.33, hippopotamus 0.33, 8.0ms
Speed: 0.5ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 9.4ms
Speed: 0.4ms preprocess, 9.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, elephant 0.33, hippopotamus 0.33, 9.1ms
Speed: 0.4ms preprocess, 9.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, elephant 0.33, hippopotamus 0.33, 12.0ms
Speed: 0.5ms preprocess, 12.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 10.8ms
Speed: 0.4ms preprocess, 10.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 9.5ms
Speed: 0.5ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, elephant 0.33, hippopotamus 0.33, 11.4ms
Speed: 0.5ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.32, 22.8ms
Speed: 0.5ms preprocess, 22.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 8.4ms
Speed: 0.6ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.32, 10.1ms
Speed: 0.5ms preprocess, 10.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.32, elephant 0.32, 11.4ms
Speed: 0.5ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, hippopotamus 0.33, elephant 0.31, 9.5ms
Speed: 0.5ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.34, elephant 0.32, 9.3ms
Speed: 0.6ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.33, hippopotamus 0.32, 9.4ms
Speed: 0.5ms preprocess, 9.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, elephant 0.33, hippopotamus 0.33, 9.3ms
Speed: 0.7ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.34, hippopotamus 0.33, elephant 0.33, 7.7ms
Speed: 0.6ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      13/50     0.661G    0.04866         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.07it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 26.63it/s]
                   all      0.727          1

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.19, 6.0ms
Speed: 5.7ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.34, elephant 0.19, 7.2ms
Speed: 0.7ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.33, elephant 0.19, 12.5ms
Speed: 0.4ms preprocess, 12.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.19, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.19, 10.0ms
Speed: 0.4ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.33, elephant 0.18, 7.8ms
Speed: 0.4ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.34, elephant 0.19, 6.7ms
Speed: 0.8ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.18, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.34, elephant 0.19, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.33, elephant 0.19, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.34, elephant 0.19, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.33, elephant 0.18, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.33, elephant 0.19, 7.8ms
Speed: 0.7ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.19, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.34, elephant 0.19, 7.5ms
Speed: 0.6ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.18, 9.3ms
Speed: 0.7ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.34, elephant 0.18, 9.5ms
Speed: 0.4ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.19, 9.8ms
Speed: 0.4ms preprocess, 9.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.18, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.34, elephant 0.18, 5.5ms
Speed: 2.1ms preprocess, 5.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.33, elephant 0.18, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.33, elephant 0.18, 5.7ms
Speed: 1.9ms preprocess, 5.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      14/50     0.661G    0.03415         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.67it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 16.79it/s]
                   all      0.773          1

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.3ms
Speed: 0.7ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 10.8ms
Speed: 0.4ms preprocess, 10.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.33, elephant 0.21, 6.0ms
Speed: 0.4ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 5.6ms
Speed: 0.4ms preprocess, 5.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 5.8ms
Speed: 2.9ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.33, elephant 0.22, 5.8ms
Speed: 3.1ms preprocess, 5.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.33, elephant 0.21, 5.3ms
Speed: 0.4ms preprocess, 5.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.32, elephant 0.21, 9.5ms
Speed: 0.4ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.9ms
Speed: 0.9ms preprocess, 6.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.8ms
Speed: 0.7ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.34, elephant 0.21, 9.8ms
Speed: 0.5ms preprocess, 9.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.33, elephant 0.22, 6.6ms
Speed: 2.0ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.32, elephant 0.21, 7.0ms
Speed: 2.2ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.2ms
Speed: 2.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.32, elephant 0.21, 6.4ms
Speed: 2.3ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.9ms
Speed: 2.1ms preprocess, 6.9ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      15/50     0.661G    0.04488         11         64: 100%|██████████| 30/30 [00:05<00:00,  5.84it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 35.24it/s]
                   all      0.773          1

0: 64x64 rhinoceros 0.58, hippopotamus 0.25, elephant 0.17, 7.6ms
Speed: 0.6ms preprocess, 7.6ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.24, elephant 0.17, 9.1ms
Speed: 0.5ms preprocess, 9.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.24, elephant 0.17, 10.2ms
Speed: 0.6ms preprocess, 10.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.25, elephant 0.17, 10.4ms
Speed: 0.6ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.16, 12.3ms
Speed: 0.5ms preprocess, 12.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.17, 11.0ms
Speed: 0.5ms preprocess, 11.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.17, 13.1ms
Speed: 0.6ms preprocess, 13.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.24, elephant 0.17, 15.1ms
Speed: 0.5ms preprocess, 15.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.17, 14.4ms
Speed: 0.5ms preprocess, 14.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.17, 18.1ms
Speed: 0.5ms preprocess, 18.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.25, elephant 0.17, 11.6ms
Speed: 1.0ms preprocess, 11.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.25, elephant 0.17, 13.6ms
Speed: 2.8ms preprocess, 13.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.26, elephant 0.17, 8.7ms
Speed: 0.6ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.17, 8.3ms
Speed: 0.6ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.25, elephant 0.17, 15.2ms
Speed: 0.6ms preprocess, 15.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.25, elephant 0.17, 11.5ms
Speed: 0.6ms preprocess, 11.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.25, elephant 0.17, 11.2ms
Speed: 0.6ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.16, 10.6ms
Speed: 0.5ms preprocess, 10.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.25, elephant 0.17, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.24, elephant 0.17, 10.5ms
Speed: 0.5ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.24, elephant 0.17, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.24, elephant 0.17, 6.2ms
Speed: 1.0ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      16/50     0.661G    0.03616         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.38it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 40.82it/s]
                   all      0.773          1

0: 64x64 rhinoceros 0.57, hippopotamus 0.23, elephant 0.20, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.20, 9.0ms
Speed: 0.7ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.21, 16.8ms
Speed: 0.9ms preprocess, 16.8ms inference, 5.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.22, elephant 0.20, 8.6ms
Speed: 0.5ms preprocess, 8.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.21, 11.6ms
Speed: 1.2ms preprocess, 11.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.23, elephant 0.21, 11.5ms
Speed: 0.7ms preprocess, 11.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.22, elephant 0.20, 8.1ms
Speed: 1.0ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.23, elephant 0.20, 6.4ms
Speed: 3.3ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.21, 8.9ms
Speed: 0.5ms preprocess, 8.9ms inference, 4.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.56, hippopotamus 0.23, elephant 0.20, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.21, 5.9ms
Speed: 0.6ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.22, elephant 0.21, 6.0ms
Speed: 0.5ms preprocess, 6.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.23, elephant 0.20, 5.8ms
Speed: 0.8ms preprocess, 5.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.20, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.20, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.20, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.21, 5.8ms
Speed: 0.3ms preprocess, 5.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.56, hippopotamus 0.23, elephant 0.21, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.23, elephant 0.21, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.23, elephant 0.21, 5.2ms
Speed: 0.4ms preprocess, 5.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.21, 5.6ms
Speed: 0.8ms preprocess, 5.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.22, elephant 0.21, 8.6ms
Speed: 0.5ms preprocess, 8.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      17/50     0.661G    0.04225         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.69it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 11.58it/s]
                   all      0.773          1

0: 64x64 elephant 0.43, rhinoceros 0.39, hippopotamus 0.18, 6.9ms
Speed: 2.3ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.18, 11.4ms
Speed: 0.5ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.39, hippopotamus 0.18, 13.6ms
Speed: 0.6ms preprocess, 13.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.40, hippopotamus 0.17, 11.9ms
Speed: 0.5ms preprocess, 11.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.18, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.39, hippopotamus 0.18, 18.0ms
Speed: 0.5ms preprocess, 18.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.40, hippopotamus 0.18, 11.7ms
Speed: 0.5ms preprocess, 11.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.41, rhinoceros 0.40, hippopotamus 0.18, 13.6ms
Speed: 0.5ms preprocess, 13.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.39, hippopotamus 0.19, 12.5ms
Speed: 0.5ms preprocess, 12.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.18, 11.3ms
Speed: 0.6ms preprocess, 11.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.18, 16.3ms
Speed: 0.5ms preprocess, 16.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.39, hippopotamus 0.18, 14.8ms
Speed: 0.5ms preprocess, 14.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.19, 12.4ms
Speed: 0.5ms preprocess, 12.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.40, hippopotamus 0.17, 8.7ms
Speed: 2.7ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.40, hippopotamus 0.18, 7.4ms
Speed: 2.8ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.39, hippopotamus 0.18, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.39, hippopotamus 0.18, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.18, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.43, rhinoceros 0.39, hippopotamus 0.18, 8.8ms
Speed: 0.4ms preprocess, 8.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.39, hippopotamus 0.19, 15.4ms
Speed: 0.5ms preprocess, 15.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.17, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.42, rhinoceros 0.40, hippopotamus 0.19, 8.3ms
Speed: 0.5ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      18/50     0.661G    0.02567         11         64: 100%|██████████| 30/30 [00:04<00:00,  6.56it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 29.86it/s]
                   all      0.773          1

0: 64x64 elephant 0.69, rhinoceros 0.20, hippopotamus 0.11, 8.0ms
Speed: 0.7ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 11.5ms
Speed: 0.5ms preprocess, 11.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.19, hippopotamus 0.11, 7.6ms
Speed: 0.7ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.20, hippopotamus 0.11, 8.8ms
Speed: 0.5ms preprocess, 8.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 7.0ms
Speed: 2.3ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.19, hippopotamus 0.11, 6.8ms
Speed: 2.7ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 6.0ms
Speed: 2.1ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.19, hippopotamus 0.12, 7.2ms
Speed: 2.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.19, hippopotamus 0.11, 6.4ms
Speed: 2.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 6.4ms
Speed: 2.2ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.71, rhinoceros 0.19, hippopotamus 0.11, 6.7ms
Speed: 4.1ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 6.1ms
Speed: 2.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 9.2ms
Speed: 3.3ms preprocess, 9.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.71, rhinoceros 0.19, hippopotamus 0.11, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.71, rhinoceros 0.19, hippopotamus 0.11, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.19, hippopotamus 0.11, 7.7ms
Speed: 0.4ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.19, hippopotamus 0.12, 10.6ms
Speed: 0.8ms preprocess, 10.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.69, rhinoceros 0.19, hippopotamus 0.12, 8.7ms
Speed: 0.5ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.70, rhinoceros 0.19, hippopotamus 0.11, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      19/50     0.661G    0.03527         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.13it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 57.05it/s]
                   all      0.773          1

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 6.2ms
Speed: 5.6ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 7.1ms
Speed: 0.7ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 6.6ms
Speed: 2.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 6.5ms
Speed: 0.6ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 8.4ms
Speed: 0.6ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 6.6ms
Speed: 0.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 4.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 6.0ms
Speed: 0.8ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 8.5ms
Speed: 0.6ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 8.4ms
Speed: 4.6ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 5.7ms
Speed: 2.3ms preprocess, 5.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.10, hippopotamus 0.06, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.09, hippopotamus 0.06, 7.4ms
Speed: 0.6ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.84, rhinoceros 0.09, hippopotamus 0.06, 7.3ms
Speed: 0.6ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.85, rhinoceros 0.09, hippopotamus 0.06, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      20/50     0.661G    0.02895         11         64: 100%|██████████| 30/30 [00:04<00:00,  7.10it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 20.68it/s]
                   all      0.818          1

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 7.7ms
Speed: 0.7ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 5.8ms
Speed: 0.8ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.76, rhinoceros 0.16, hippopotamus 0.09, 6.6ms
Speed: 0.4ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.74, rhinoceros 0.16, hippopotamus 0.10, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.76, rhinoceros 0.16, hippopotamus 0.09, 6.2ms
Speed: 0.4ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 12.6ms
Speed: 0.4ms preprocess, 12.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.74, rhinoceros 0.16, hippopotamus 0.10, 9.7ms
Speed: 0.6ms preprocess, 9.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 7.5ms
Speed: 0.4ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.76, rhinoceros 0.16, hippopotamus 0.09, 7.4ms
Speed: 0.3ms preprocess, 7.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 5.3ms
Speed: 2.5ms preprocess, 5.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 6.8ms
Speed: 0.9ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.74, rhinoceros 0.16, hippopotamus 0.09, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 6.2ms
Speed: 0.4ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.74, rhinoceros 0.16, hippopotamus 0.10, 7.0ms
Speed: 0.4ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.74, rhinoceros 0.16, hippopotamus 0.09, 8.9ms
Speed: 0.4ms preprocess, 8.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.75, rhinoceros 0.16, hippopotamus 0.09, 6.5ms
Speed: 0.4ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.74, rhinoceros 0.16, hippopotamus 0.09, 8.0ms
Speed: 0.5ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      21/50     0.661G    0.02584         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.29it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 29.50it/s]
                   all      0.727          1

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 11.6ms
Speed: 0.5ms preprocess, 11.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.06, hippopotamus 0.03, 6.1ms
Speed: 0.8ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.05, hippopotamus 0.03, 6.3ms
Speed: 0.8ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.0ms
Speed: 0.9ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.05, hippopotamus 0.03, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.06, hippopotamus 0.03, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.05, hippopotamus 0.03, 7.7ms
Speed: 0.4ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.5ms
Speed: 0.6ms preprocess, 6.5ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 8.6ms
Speed: 0.5ms preprocess, 8.6ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 8.5ms
Speed: 0.6ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.6ms
Speed: 0.4ms preprocess, 6.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.06, hippopotamus 0.03, 8.6ms
Speed: 0.4ms preprocess, 8.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.06, hippopotamus 0.03, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.91, rhinoceros 0.06, hippopotamus 0.03, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.92, rhinoceros 0.06, hippopotamus 0.03, 6.9ms
Speed: 0.4ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      22/50     0.661G    0.02241         11         64: 100%|██████████| 30/30 [00:04<00:00,  6.01it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 14.68it/s]
                   all      0.773          1

0: 64x64 elephant 0.81, rhinoceros 0.14, hippopotamus 0.05, 13.2ms
Speed: 0.5ms preprocess, 13.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.79, rhinoceros 0.15, hippopotamus 0.05, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.79, rhinoceros 0.15, hippopotamus 0.05, 8.1ms
Speed: 0.5ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.79, rhinoceros 0.15, hippopotamus 0.06, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.4ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, rhinoceros 0.14, hippopotamus 0.05, 8.2ms
Speed: 0.4ms preprocess, 8.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.4ms
Speed: 0.4ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 6.4ms
Speed: 2.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 9.4ms
Speed: 1.1ms preprocess, 9.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, rhinoceros 0.14, hippopotamus 0.05, 7.6ms
Speed: 0.4ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.6ms
Speed: 0.4ms preprocess, 7.6ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, rhinoceros 0.14, hippopotamus 0.05, 6.6ms
Speed: 0.4ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 6.5ms
Speed: 0.6ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.5ms
Speed: 0.8ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.0ms
Speed: 0.4ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.0ms
Speed: 0.4ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, rhinoceros 0.14, hippopotamus 0.05, 6.2ms
Speed: 2.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, rhinoceros 0.14, hippopotamus 0.05, 8.7ms
Speed: 0.5ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.80, rhinoceros 0.15, hippopotamus 0.05, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.81, rhinoceros 0.15, hippopotamus 0.05, 6.9ms
Speed: 0.8ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      23/50     0.661G    0.01867         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.09it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 30.53it/s]
                   all      0.818          1

0: 64x64 elephant 0.57, rhinoceros 0.31, hippopotamus 0.12, 7.6ms
Speed: 0.6ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.59, rhinoceros 0.30, hippopotamus 0.11, 8.0ms
Speed: 0.5ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 7.7ms
Speed: 0.4ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 6.0ms
Speed: 0.5ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 7.9ms
Speed: 0.6ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 8.1ms
Speed: 3.7ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, rhinoceros 0.31, hippopotamus 0.12, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.59, rhinoceros 0.30, hippopotamus 0.12, 8.6ms
Speed: 0.5ms preprocess, 8.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 8.5ms
Speed: 3.5ms preprocess, 8.5ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.59, rhinoceros 0.29, hippopotamus 0.12, 5.9ms
Speed: 0.5ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.59, rhinoceros 0.30, hippopotamus 0.11, 9.3ms
Speed: 0.4ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.60, rhinoceros 0.29, hippopotamus 0.11, 5.4ms
Speed: 0.5ms preprocess, 5.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.59, rhinoceros 0.29, hippopotamus 0.12, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.59, rhinoceros 0.30, hippopotamus 0.11, 10.5ms
Speed: 0.5ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.57, rhinoceros 0.30, hippopotamus 0.13, 5.9ms
Speed: 0.5ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 8.1ms
Speed: 0.7ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 elephant 0.58, rhinoceros 0.30, hippopotamus 0.12, 6.5ms
Speed: 0.4ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      24/50     0.661G    0.02065         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.59it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 35.95it/s]
                   all      0.909          1

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.19, 17.3ms
Speed: 0.5ms preprocess, 17.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.20, 9.9ms
Speed: 0.5ms preprocess, 9.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.29, elephant 0.19, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.29, elephant 0.19, 10.9ms
Speed: 0.5ms preprocess, 10.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.29, elephant 0.20, 12.5ms
Speed: 0.5ms preprocess, 12.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.28, elephant 0.21, 17.2ms
Speed: 0.5ms preprocess, 17.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.20, 10.7ms
Speed: 0.5ms preprocess, 10.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.29, elephant 0.20, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 4.7ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.20, 13.5ms
Speed: 0.5ms preprocess, 13.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.29, elephant 0.19, 11.5ms
Speed: 0.5ms preprocess, 11.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.20, 13.8ms
Speed: 0.5ms preprocess, 13.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.19, 6.0ms
Speed: 0.5ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.28, elephant 0.21, 8.7ms
Speed: 0.4ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.21, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.28, elephant 0.20, 6.0ms
Speed: 0.4ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.28, elephant 0.20, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.29, elephant 0.19, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.29, elephant 0.20, 6.1ms
Speed: 0.4ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.29, elephant 0.19, 5.9ms
Speed: 0.5ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.29, elephant 0.20, 8.4ms
Speed: 0.4ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.28, elephant 0.21, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.28, elephant 0.21, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      25/50     0.661G    0.01177         11         64: 100%|██████████| 30/30 [00:04<00:00,  7.02it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 22.70it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.48, elephant 0.27, hippopotamus 0.25, 7.5ms
Speed: 0.6ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.26, elephant 0.25, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, elephant 0.26, hippopotamus 0.26, 10.7ms
Speed: 0.4ms preprocess, 10.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, elephant 0.26, hippopotamus 0.26, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, elephant 0.27, hippopotamus 0.25, 6.9ms
Speed: 1.9ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.26, elephant 0.25, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.26, elephant 0.26, 13.8ms
Speed: 0.5ms preprocess, 13.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, elephant 0.27, hippopotamus 0.26, 12.0ms
Speed: 0.4ms preprocess, 12.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, elephant 0.26, hippopotamus 0.26, 10.6ms
Speed: 0.5ms preprocess, 10.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.26, elephant 0.26, 9.7ms
Speed: 0.5ms preprocess, 9.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.27, elephant 0.25, 7.2ms
Speed: 0.4ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.26, elephant 0.25, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.27, elephant 0.25, 9.8ms
Speed: 0.4ms preprocess, 9.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.26, elephant 0.25, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.27, elephant 0.25, 5.5ms
Speed: 2.3ms preprocess, 5.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, elephant 0.27, hippopotamus 0.25, 6.4ms
Speed: 2.3ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.27, elephant 0.25, 7.3ms
Speed: 0.8ms preprocess, 7.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.26, elephant 0.26, 5.4ms
Speed: 0.4ms preprocess, 5.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, elephant 0.26, hippopotamus 0.26, 9.2ms
Speed: 0.5ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, elephant 0.27, hippopotamus 0.26, 9.2ms
Speed: 2.0ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.26, elephant 0.25, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.26, elephant 0.26, 9.0ms
Speed: 0.4ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      26/50     0.661G    0.02152         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.84it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 14.20it/s]
                   all      0.773          1

0: 64x64 rhinoceros 0.62, hippopotamus 0.23, elephant 0.15, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.22, elephant 0.17, 10.2ms
Speed: 0.5ms preprocess, 10.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.23, elephant 0.15, 10.3ms
Speed: 0.4ms preprocess, 10.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.23, elephant 0.15, 8.0ms
Speed: 0.6ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.15, 8.6ms
Speed: 0.4ms preprocess, 8.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.23, elephant 0.15, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 12.5ms
Speed: 0.4ms preprocess, 12.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.23, elephant 0.16, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.22, elephant 0.15, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 6.0ms
Speed: 0.4ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.15, 6.6ms
Speed: 0.9ms preprocess, 6.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 6.7ms
Speed: 8.0ms preprocess, 6.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 8.5ms
Speed: 0.4ms preprocess, 8.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 6.8ms
Speed: 1.0ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.23, elephant 0.16, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.22, elephant 0.16, 5.8ms
Speed: 2.9ms preprocess, 5.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.23, elephant 0.15, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.23, elephant 0.15, 8.2ms
Speed: 0.6ms preprocess, 8.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.23, elephant 0.15, 6.8ms
Speed: 1.5ms preprocess, 6.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.23, elephant 0.16, 9.1ms
Speed: 1.2ms preprocess, 9.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      27/50     0.661G    0.01098         11         64: 100%|██████████| 30/30 [00:04<00:00,  7.26it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 20.25it/s]
                   all      0.773          1

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 11.9ms
Speed: 0.5ms preprocess, 11.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.39, elephant 0.10, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 13.5ms
Speed: 0.4ms preprocess, 13.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.40, elephant 0.09, 13.8ms
Speed: 0.5ms preprocess, 13.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.38, elephant 0.09, 9.9ms
Speed: 0.4ms preprocess, 9.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.38, elephant 0.09, 9.7ms
Speed: 0.5ms preprocess, 9.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.38, elephant 0.09, 9.2ms
Speed: 0.5ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 11.4ms
Speed: 0.4ms preprocess, 11.4ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 12.3ms
Speed: 0.5ms preprocess, 12.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.38, elephant 0.09, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 10.5ms
Speed: 0.5ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 12.7ms
Speed: 0.4ms preprocess, 12.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.38, elephant 0.09, 7.4ms
Speed: 0.4ms preprocess, 7.4ms inference, 4.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.40, elephant 0.08, 10.3ms
Speed: 0.5ms preprocess, 10.3ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.38, elephant 0.09, 9.3ms
Speed: 0.5ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.38, elephant 0.09, 9.2ms
Speed: 0.5ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 9.3ms
Speed: 0.5ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.39, elephant 0.08, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.39, elephant 0.09, 8.9ms
Speed: 0.5ms preprocess, 8.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.39, elephant 0.09, 10.2ms
Speed: 0.6ms preprocess, 10.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      28/50     0.661G    0.01191         11         64: 100%|██████████| 30/30 [00:03<00:00,  7.76it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 17.61it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.50, hippopotamus 0.38, elephant 0.12, 9.7ms
Speed: 0.6ms preprocess, 9.7ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.37, elephant 0.12, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.39, elephant 0.12, 6.9ms
Speed: 0.6ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.38, elephant 0.12, 8.3ms
Speed: 0.4ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.39, elephant 0.12, 7.8ms
Speed: 0.4ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.39, elephant 0.12, 11.2ms
Speed: 0.5ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.38, elephant 0.12, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.38, elephant 0.12, 16.6ms
Speed: 0.5ms preprocess, 16.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.38, elephant 0.12, 10.6ms
Speed: 3.7ms preprocess, 10.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.39, elephant 0.12, 8.5ms
Speed: 0.7ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.37, elephant 0.12, 7.2ms
Speed: 0.9ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.38, elephant 0.11, 8.2ms
Speed: 0.7ms preprocess, 8.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.37, elephant 0.12, 6.4ms
Speed: 3.3ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.38, elephant 0.12, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.37, elephant 0.12, 6.4ms
Speed: 1.6ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.38, elephant 0.12, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.39, elephant 0.12, 7.7ms
Speed: 0.9ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.38, elephant 0.12, 7.0ms
Speed: 0.9ms preprocess, 7.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.38, elephant 0.12, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.38, elephant 0.13, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.39, elephant 0.11, 8.7ms
Speed: 0.8ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.38, elephant 0.11, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      29/50     0.661G     0.0109         11         64: 100%|██████████| 30/30 [00:03<00:00,  9.01it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 16.87it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.51, hippopotamus 0.35, elephant 0.15, 8.8ms
Speed: 0.7ms preprocess, 8.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.15, 6.6ms
Speed: 5.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.15, 8.0ms
Speed: 0.6ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.15, 7.7ms
Speed: 0.6ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.35, elephant 0.14, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.35, elephant 0.14, 9.7ms
Speed: 0.7ms preprocess, 9.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.35, elephant 0.14, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.15, 9.8ms
Speed: 0.5ms preprocess, 9.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.14, 8.7ms
Speed: 4.6ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.34, elephant 0.14, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.35, elephant 0.15, 6.0ms
Speed: 0.7ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.14, 5.9ms
Speed: 0.5ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.35, elephant 0.15, 9.6ms
Speed: 0.5ms preprocess, 9.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.35, elephant 0.15, 8.6ms
Speed: 0.5ms preprocess, 8.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.15, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.34, elephant 0.14, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.35, elephant 0.15, 13.0ms
Speed: 0.5ms preprocess, 13.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.35, elephant 0.14, 5.8ms
Speed: 0.5ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.34, elephant 0.16, 7.1ms
Speed: 0.7ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.15, 6.9ms
Speed: 1.2ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.34, elephant 0.15, 7.0ms
Speed: 0.7ms preprocess, 7.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.51, hippopotamus 0.35, elephant 0.14, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      30/50     0.661G    0.00821         11         64: 100%|██████████| 30/30 [00:04<00:00,  6.42it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 33.41it/s]
                   all      0.864          1

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 9.8ms
Speed: 0.5ms preprocess, 9.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.36, elephant 0.18, 14.0ms
Speed: 0.5ms preprocess, 14.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 14.3ms
Speed: 0.4ms preprocess, 14.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.17, 7.6ms
Speed: 0.4ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.36, elephant 0.19, 13.8ms
Speed: 0.4ms preprocess, 13.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.35, elephant 0.17, 11.1ms
Speed: 0.5ms preprocess, 11.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.16, 13.8ms
Speed: 0.5ms preprocess, 13.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.35, elephant 0.17, 9.8ms
Speed: 0.6ms preprocess, 9.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.17, 14.0ms
Speed: 0.6ms preprocess, 14.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.17, 11.4ms
Speed: 0.6ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.17, 12.1ms
Speed: 0.4ms preprocess, 12.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.17, 10.2ms
Speed: 0.6ms preprocess, 10.2ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 12.6ms
Speed: 0.4ms preprocess, 12.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 6.8ms
Speed: 0.4ms preprocess, 6.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.37, elephant 0.16, 8.9ms
Speed: 0.5ms preprocess, 8.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 6.7ms
Speed: 1.0ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.17, 9.6ms
Speed: 0.6ms preprocess, 9.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.36, elephant 0.18, 10.2ms
Speed: 0.4ms preprocess, 10.2ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.16, 7.8ms
Speed: 0.4ms preprocess, 7.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 7.7ms
Speed: 0.4ms preprocess, 7.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.36, elephant 0.17, 8.3ms
Speed: 0.4ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      31/50     0.661G    0.01912         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.59it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 19.97it/s]
                   all      0.727          1

0: 64x64 rhinoceros 0.37, elephant 0.32, hippopotamus 0.31, 6.5ms
Speed: 1.6ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.31, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.31, 8.1ms
Speed: 0.5ms preprocess, 8.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.31, elephant 0.31, 11.1ms
Speed: 2.4ms preprocess, 11.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.30, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.32, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.32, elephant 0.30, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.8ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.32, elephant 0.30, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.30, 13.7ms
Speed: 0.4ms preprocess, 13.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.32, elephant 0.30, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.33, elephant 0.30, 8.2ms
Speed: 0.5ms preprocess, 8.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.33, elephant 0.29, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.31, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.31, elephant 0.30, 8.9ms
Speed: 0.5ms preprocess, 8.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.31, 8.0ms
Speed: 0.4ms preprocess, 8.0ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.35, elephant 0.35, hippopotamus 0.30, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.31, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.30, 8.0ms
Speed: 0.9ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.37, hippopotamus 0.32, elephant 0.31, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.32, elephant 0.30, 6.0ms
Speed: 0.4ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.32, elephant 0.31, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.38, hippopotamus 0.32, elephant 0.30, 7.0ms
Speed: 0.7ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      32/50     0.661G    0.01386         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.55it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 21.16it/s]
                   all      0.773          1

0: 64x64 hippopotamus 0.47, rhinoceros 0.32, elephant 0.22, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.33, elephant 0.21, 6.4ms
Speed: 0.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.33, elephant 0.21, 7.4ms
Speed: 0.6ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.32, elephant 0.23, 13.8ms
Speed: 0.5ms preprocess, 13.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.33, elephant 0.21, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.32, elephant 0.21, 7.4ms
Speed: 0.4ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.33, elephant 0.23, 8.3ms
Speed: 0.6ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.33, elephant 0.21, 10.8ms
Speed: 0.4ms preprocess, 10.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.33, elephant 0.21, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.32, elephant 0.21, 12.6ms
Speed: 0.4ms preprocess, 12.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.32, elephant 0.21, 6.1ms
Speed: 0.4ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.32, elephant 0.21, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.32, elephant 0.22, 6.7ms
Speed: 0.4ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.33, elephant 0.22, 6.8ms
Speed: 0.4ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.44, rhinoceros 0.31, elephant 0.25, 6.4ms
Speed: 1.1ms preprocess, 6.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.33, elephant 0.20, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.32, elephant 0.21, 7.1ms
Speed: 1.0ms preprocess, 7.1ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.33, elephant 0.22, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.33, elephant 0.21, 8.0ms
Speed: 0.5ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.32, elephant 0.22, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.33, elephant 0.22, 7.4ms
Speed: 0.4ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.32, elephant 0.21, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      33/50     0.661G   0.008211         11         64: 100%|██████████| 30/30 [00:05<00:00,  5.25it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 20.36it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.47, hippopotamus 0.32, elephant 0.22, 8.6ms
Speed: 0.5ms preprocess, 8.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.32, elephant 0.21, 11.7ms
Speed: 0.5ms preprocess, 11.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.21, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.20, 9.0ms
Speed: 0.4ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.20, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 12.3ms
Speed: 0.6ms preprocess, 12.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.20, 10.3ms
Speed: 0.5ms preprocess, 10.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.20, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.20, 8.6ms
Speed: 0.4ms preprocess, 8.6ms inference, 4.9ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.34, elephant 0.20, 10.0ms
Speed: 0.4ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.32, elephant 0.22, 9.9ms
Speed: 0.4ms preprocess, 9.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.21, 11.7ms
Speed: 0.4ms preprocess, 11.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.20, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.20, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.34, elephant 0.20, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.20, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.32, elephant 0.21, 5.8ms
Speed: 0.3ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.33, elephant 0.21, 6.6ms
Speed: 0.9ms preprocess, 6.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.19, 9.0ms
Speed: 1.1ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.33, elephant 0.20, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.44, hippopotamus 0.32, elephant 0.24, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.32, elephant 0.20, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      34/50     0.661G   0.006629         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.89it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 30.28it/s]
                   all      0.773          1

0: 64x64 hippopotamus 0.46, rhinoceros 0.39, elephant 0.15, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.37, elephant 0.18, 9.3ms
Speed: 0.5ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.39, elephant 0.15, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.38, elephant 0.16, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.39, elephant 0.15, 8.3ms
Speed: 0.5ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.39, elephant 0.15, 8.0ms
Speed: 0.5ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 10.9ms
Speed: 0.5ms preprocess, 10.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.40, elephant 0.16, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.39, elephant 0.15, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.39, elephant 0.16, 13.2ms
Speed: 0.5ms preprocess, 13.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 6.2ms
Speed: 1.2ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 6.2ms
Speed: 0.4ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.39, elephant 0.15, 11.1ms
Speed: 0.5ms preprocess, 11.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 11.7ms
Speed: 0.4ms preprocess, 11.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.46, rhinoceros 0.39, elephant 0.15, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.40, elephant 0.15, 6.9ms
Speed: 0.6ms preprocess, 6.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.39, elephant 0.16, 9.0ms
Speed: 0.9ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.45, rhinoceros 0.39, elephant 0.16, 7.1ms
Speed: 0.4ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 6.4ms
Speed: 0.5ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 hippopotamus 0.47, rhinoceros 0.38, elephant 0.15, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      35/50     0.661G    0.01125         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.47it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 44.56it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.45, elephant 0.31, hippopotamus 0.24, 11.5ms
Speed: 0.5ms preprocess, 11.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.32, hippopotamus 0.23, 12.2ms
Speed: 0.4ms preprocess, 12.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.44, elephant 0.31, hippopotamus 0.24, 16.2ms
Speed: 0.4ms preprocess, 16.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, elephant 0.37, hippopotamus 0.21, 11.2ms
Speed: 0.4ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.44, elephant 0.34, hippopotamus 0.22, 11.5ms
Speed: 0.4ms preprocess, 11.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.32, hippopotamus 0.22, 21.8ms
Speed: 0.4ms preprocess, 21.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, elephant 0.32, hippopotamus 0.22, 11.8ms
Speed: 0.5ms preprocess, 11.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.34, hippopotamus 0.22, 12.6ms
Speed: 0.4ms preprocess, 12.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.32, hippopotamus 0.23, 16.9ms
Speed: 0.5ms preprocess, 16.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.31, hippopotamus 0.23, 15.5ms
Speed: 0.8ms preprocess, 15.5ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.32, hippopotamus 0.23, 11.2ms
Speed: 0.4ms preprocess, 11.2ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.32, hippopotamus 0.24, 8.5ms
Speed: 2.9ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.31, hippopotamus 0.24, 5.9ms
Speed: 0.5ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, elephant 0.32, hippopotamus 0.23, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.33, hippopotamus 0.22, 5.8ms
Speed: 0.4ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.31, hippopotamus 0.23, 11.2ms
Speed: 0.4ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.32, hippopotamus 0.23, 11.0ms
Speed: 0.5ms preprocess, 11.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, elephant 0.33, hippopotamus 0.22, 6.0ms
Speed: 0.4ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, elephant 0.31, hippopotamus 0.23, 7.1ms
Speed: 0.4ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, elephant 0.31, hippopotamus 0.23, 9.3ms
Speed: 0.4ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, elephant 0.32, hippopotamus 0.22, 8.4ms
Speed: 0.5ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, elephant 0.30, hippopotamus 0.24, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      36/50     0.661G    0.01599         11         64: 100%|██████████| 30/30 [00:05<00:00,  5.90it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 15.59it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.45, hippopotamus 0.35, elephant 0.19, 7.2ms
Speed: 1.0ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.43, hippopotamus 0.36, elephant 0.21, 14.3ms
Speed: 0.5ms preprocess, 14.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.36, elephant 0.17, 13.0ms
Speed: 0.4ms preprocess, 13.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.36, elephant 0.19, 14.1ms
Speed: 1.9ms preprocess, 14.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.37, elephant 0.18, 13.2ms
Speed: 4.0ms preprocess, 13.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.44, hippopotamus 0.38, elephant 0.18, 9.2ms
Speed: 3.3ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.35, elephant 0.19, 6.1ms
Speed: 0.4ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.38, elephant 0.17, 6.6ms
Speed: 0.4ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.35, elephant 0.18, 10.0ms
Speed: 0.4ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.36, elephant 0.19, 14.7ms
Speed: 0.4ms preprocess, 14.7ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.37, elephant 0.18, 6.1ms
Speed: 0.4ms preprocess, 6.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.37, elephant 0.17, 7.9ms
Speed: 0.4ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.37, elephant 0.18, 5.5ms
Speed: 0.4ms preprocess, 5.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.36, elephant 0.18, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.37, elephant 0.18, 9.4ms
Speed: 0.5ms preprocess, 9.4ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.37, elephant 0.18, 8.0ms
Speed: 0.4ms preprocess, 8.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.36, elephant 0.18, 6.1ms
Speed: 0.3ms preprocess, 6.1ms inference, 0.4ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.36, elephant 0.18, 5.9ms
Speed: 0.4ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.36, elephant 0.18, 7.0ms
Speed: 0.7ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.37, elephant 0.18, 9.1ms
Speed: 0.7ms preprocess, 9.1ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.35, elephant 0.19, 7.2ms
Speed: 0.4ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.45, hippopotamus 0.38, elephant 0.17, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      37/50     0.661G    0.01231         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.33it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 20.34it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.41, hippopotamus 0.40, elephant 0.18, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.40, elephant 0.18, 6.6ms
Speed: 0.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.41, hippopotamus 0.38, elephant 0.21, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.40, elephant 0.18, 8.3ms
Speed: 0.5ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.40, elephant 0.18, 13.2ms
Speed: 0.5ms preprocess, 13.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.43, hippopotamus 0.39, elephant 0.18, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.39, elephant 0.19, 8.5ms
Speed: 0.4ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.39, elephant 0.19, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.43, hippopotamus 0.39, elephant 0.18, 7.4ms
Speed: 1.9ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.39, elephant 0.18, 9.7ms
Speed: 0.5ms preprocess, 9.7ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.43, hippopotamus 0.40, elephant 0.18, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.40, elephant 0.18, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.39, elephant 0.18, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.40, elephant 0.18, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.43, hippopotamus 0.38, elephant 0.19, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.43, hippopotamus 0.39, elephant 0.18, 9.1ms
Speed: 0.5ms preprocess, 9.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.40, elephant 0.18, 6.9ms
Speed: 0.7ms preprocess, 6.9ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.43, hippopotamus 0.38, elephant 0.18, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.39, elephant 0.18, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.38, elephant 0.20, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.40, elephant 0.18, 8.3ms
Speed: 0.6ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.42, hippopotamus 0.39, elephant 0.19, 6.7ms
Speed: 0.7ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      38/50     0.661G   0.005945         11         64: 100%|██████████| 30/30 [00:03<00:00,  7.82it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 23.50it/s]
                   all      0.864          1

0: 64x64 rhinoceros 0.47, hippopotamus 0.43, elephant 0.10, 14.1ms
Speed: 0.6ms preprocess, 14.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.43, elephant 0.11, 21.6ms
Speed: 0.5ms preprocess, 21.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.42, elephant 0.11, 13.6ms
Speed: 0.6ms preprocess, 13.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.41, elephant 0.12, 11.9ms
Speed: 0.5ms preprocess, 11.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.42, elephant 0.10, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.42, elephant 0.11, 12.7ms
Speed: 0.4ms preprocess, 12.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.42, elephant 0.11, 11.4ms
Speed: 0.4ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.43, elephant 0.10, 9.5ms
Speed: 0.5ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.42, elephant 0.11, 16.6ms
Speed: 0.5ms preprocess, 16.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.41, elephant 0.11, 12.9ms
Speed: 0.5ms preprocess, 12.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.42, elephant 0.11, 15.6ms
Speed: 0.4ms preprocess, 15.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.41, elephant 0.11, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.42, elephant 0.10, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.41, elephant 0.11, 9.3ms
Speed: 0.6ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.41, elephant 0.11, 11.8ms
Speed: 0.9ms preprocess, 11.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.42, elephant 0.10, 7.8ms
Speed: 0.4ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.42, elephant 0.11, 8.5ms
Speed: 0.9ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.41, elephant 0.10, 8.4ms
Speed: 0.5ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.42, elephant 0.11, 6.1ms
Speed: 0.5ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.41, elephant 0.11, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.42, elephant 0.11, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.41, elephant 0.11, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      39/50     0.661G   0.007497         11         64: 100%|██████████| 30/30 [00:04<00:00,  6.98it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 19.33it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.50, hippopotamus 0.36, elephant 0.14, 6.0ms
Speed: 0.5ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.37, elephant 0.14, 11.4ms
Speed: 0.4ms preprocess, 11.4ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.37, elephant 0.14, 10.0ms
Speed: 0.5ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.37, elephant 0.14, 8.7ms
Speed: 1.5ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.37, elephant 0.14, 7.2ms
Speed: 0.4ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.36, elephant 0.14, 7.1ms
Speed: 0.4ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.36, elephant 0.15, 8.9ms
Speed: 0.5ms preprocess, 8.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.37, elephant 0.14, 10.1ms
Speed: 0.4ms preprocess, 10.1ms inference, 3.4ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.38, elephant 0.14, 13.5ms
Speed: 0.5ms preprocess, 13.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.36, elephant 0.14, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.37, elephant 0.14, 7.7ms
Speed: 0.4ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.37, elephant 0.16, 8.1ms
Speed: 0.4ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.37, elephant 0.15, 6.2ms
Speed: 1.9ms preprocess, 6.2ms inference, 0.5ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.38, elephant 0.14, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.50, hippopotamus 0.36, elephant 0.14, 7.1ms
Speed: 0.4ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.36, elephant 0.15, 6.1ms
Speed: 0.6ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.37, elephant 0.14, 7.0ms
Speed: 0.4ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.49, hippopotamus 0.36, elephant 0.15, 7.2ms
Speed: 0.4ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.37, elephant 0.14, 6.2ms
Speed: 0.4ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.15, 6.9ms
Speed: 1.0ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.38, elephant 0.14, 7.8ms
Speed: 0.4ms preprocess, 7.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.37, elephant 0.14, 7.2ms
Speed: 1.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      40/50     0.661G   0.007117         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.31it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 15.89it/s]
                   all      0.773          1

0: 64x64 rhinoceros 0.58, hippopotamus 0.32, elephant 0.09, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.32, elephant 0.09, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.09, 10.8ms
Speed: 0.5ms preprocess, 10.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.32, elephant 0.09, 11.2ms
Speed: 0.5ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.10, 7.3ms
Speed: 0.4ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.34, elephant 0.09, 6.6ms
Speed: 0.4ms preprocess, 6.6ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.34, elephant 0.10, 8.7ms
Speed: 0.5ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.09, 12.2ms
Speed: 0.8ms preprocess, 12.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.09, 9.5ms
Speed: 0.5ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.09, 6.4ms
Speed: 0.4ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.09, 6.6ms
Speed: 0.4ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.09, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.32, elephant 0.09, 8.7ms
Speed: 0.4ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.32, elephant 0.10, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.32, elephant 0.09, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.32, elephant 0.09, 6.5ms
Speed: 0.4ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.32, elephant 0.10, 14.6ms
Speed: 0.5ms preprocess, 14.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.32, elephant 0.10, 7.8ms
Speed: 1.6ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.34, elephant 0.09, 9.2ms
Speed: 0.6ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.32, elephant 0.10, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.32, elephant 0.10, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.33, elephant 0.09, 14.7ms
Speed: 0.4ms preprocess, 14.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)
Closing dataloader mosaic

      Epoch    GPU_mem       loss  Instances       Size
      41/50     0.661G   0.006492         11         64: 100%|██████████| 30/30 [00:05<00:00,  5.76it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 18.41it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.46, hippopotamus 0.39, elephant 0.14, 7.8ms
Speed: 0.7ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.37, elephant 0.15, 14.2ms
Speed: 0.5ms preprocess, 14.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.40, elephant 0.14, 6.9ms
Speed: 3.6ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.38, elephant 0.14, 11.4ms
Speed: 0.5ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.38, elephant 0.14, 10.5ms
Speed: 0.5ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.14, 8.5ms
Speed: 0.6ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.39, elephant 0.15, 9.9ms
Speed: 0.5ms preprocess, 9.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.38, elephant 0.14, 10.1ms
Speed: 0.5ms preprocess, 10.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.14, 11.1ms
Speed: 0.4ms preprocess, 11.1ms inference, 2.6ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.37, elephant 0.14, 10.5ms
Speed: 0.5ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.37, elephant 0.14, 8.3ms
Speed: 3.5ms preprocess, 8.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.48, hippopotamus 0.37, elephant 0.14, 10.8ms
Speed: 0.4ms preprocess, 10.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.14, 9.8ms
Speed: 0.4ms preprocess, 9.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.15, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 0.9ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.39, elephant 0.15, 9.9ms
Speed: 0.5ms preprocess, 9.9ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.15, 10.4ms
Speed: 1.4ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.39, elephant 0.14, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.15, 10.4ms
Speed: 1.6ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.46, hippopotamus 0.38, elephant 0.16, 13.2ms
Speed: 0.6ms preprocess, 13.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.39, elephant 0.15, 10.8ms
Speed: 0.6ms preprocess, 10.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.15, 17.8ms
Speed: 0.6ms preprocess, 17.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.47, hippopotamus 0.38, elephant 0.14, 14.6ms
Speed: 0.6ms preprocess, 14.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      42/50     0.661G   0.006824         11         64: 100%|██████████| 30/30 [00:04<00:00,  7.27it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 20.56it/s]
                   all      0.864          1

0: 64x64 rhinoceros 0.54, hippopotamus 0.35, elephant 0.12, 11.0ms
Speed: 0.6ms preprocess, 11.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.35, elephant 0.12, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.36, elephant 0.12, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.36, elephant 0.11, 12.3ms
Speed: 0.5ms preprocess, 12.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.35, elephant 0.11, 6.5ms
Speed: 0.6ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.55, hippopotamus 0.34, elephant 0.11, 7.7ms
Speed: 0.6ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.35, elephant 0.12, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.36, elephant 0.12, 7.3ms
Speed: 0.7ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.35, elephant 0.12, 7.8ms
Speed: 0.6ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.35, elephant 0.11, 7.1ms
Speed: 1.8ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.36, elephant 0.11, 10.9ms
Speed: 0.5ms preprocess, 10.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.34, elephant 0.11, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.35, elephant 0.12, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.55, hippopotamus 0.33, elephant 0.12, 11.3ms
Speed: 0.6ms preprocess, 11.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.35, elephant 0.12, 9.6ms
Speed: 0.5ms preprocess, 9.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, hippopotamus 0.35, elephant 0.13, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.35, elephant 0.11, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.55, hippopotamus 0.34, elephant 0.11, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.35, elephant 0.12, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.35, elephant 0.12, 6.9ms
Speed: 0.4ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.34, elephant 0.12, 7.6ms
Speed: 0.6ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.34, elephant 0.12, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      43/50     0.661G    0.00298         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.33it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 39.38it/s]
                   all      0.864          1

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 24.0ms
Speed: 0.7ms preprocess, 24.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 10.6ms
Speed: 0.6ms preprocess, 10.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.69, hippopotamus 0.22, elephant 0.10, 7.8ms
Speed: 1.0ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.21, elephant 0.10, 14.3ms
Speed: 0.6ms preprocess, 14.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 10.6ms
Speed: 0.5ms preprocess, 10.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 11.4ms
Speed: 0.5ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.69, hippopotamus 0.21, elephant 0.10, 8.1ms
Speed: 0.5ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 15.5ms
Speed: 0.5ms preprocess, 15.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.11, 11.3ms
Speed: 0.5ms preprocess, 11.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 11.1ms
Speed: 0.5ms preprocess, 11.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.69, hippopotamus 0.21, elephant 0.10, 14.5ms
Speed: 1.3ms preprocess, 14.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.67, hippopotamus 0.23, elephant 0.10, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.69, hippopotamus 0.21, elephant 0.10, 9.9ms
Speed: 0.5ms preprocess, 9.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 12.3ms
Speed: 0.6ms preprocess, 12.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 12.0ms
Speed: 0.6ms preprocess, 12.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.21, elephant 0.10, 11.6ms
Speed: 0.5ms preprocess, 11.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 12.6ms
Speed: 0.5ms preprocess, 12.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 9.4ms
Speed: 0.6ms preprocess, 9.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.67, hippopotamus 0.23, elephant 0.10, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.68, hippopotamus 0.22, elephant 0.10, 5.9ms
Speed: 0.6ms preprocess, 5.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.69, hippopotamus 0.21, elephant 0.10, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.67, hippopotamus 0.23, elephant 0.10, 6.1ms
Speed: 0.6ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      44/50     0.661G   0.007111         11         64: 100%|██████████| 30/30 [00:05<00:00,  5.55it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 23.31it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.24, elephant 0.13, 8.2ms
Speed: 0.5ms preprocess, 8.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.24, elephant 0.15, 8.7ms
Speed: 0.5ms preprocess, 8.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.25, elephant 0.14, 6.6ms
Speed: 0.7ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.25, elephant 0.13, 8.4ms
Speed: 0.5ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.24, elephant 0.13, 15.9ms
Speed: 0.6ms preprocess, 15.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 11.2ms
Speed: 0.6ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 14.7ms
Speed: 1.0ms preprocess, 14.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.24, elephant 0.13, 7.5ms
Speed: 0.4ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.25, elephant 0.13, 11.8ms
Speed: 0.6ms preprocess, 11.8ms inference, 0.3ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.25, elephant 0.14, 13.1ms
Speed: 0.7ms preprocess, 13.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 6.4ms
Speed: 0.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 9.5ms
Speed: 0.5ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 12.0ms
Speed: 0.6ms preprocess, 12.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.24, elephant 0.13, 6.3ms
Speed: 0.6ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.13, 6.4ms
Speed: 0.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 10.0ms
Speed: 0.6ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 8.1ms
Speed: 0.5ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.23, elephant 0.13, 6.9ms
Speed: 0.6ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.24, elephant 0.14, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.23, elephant 0.14, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.25, elephant 0.14, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      45/50     0.661G   0.004443         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.35it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 24.31it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 7.9ms
Speed: 0.7ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.18, elephant 0.12, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.72, hippopotamus 0.16, elephant 0.12, 13.1ms
Speed: 0.5ms preprocess, 13.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 8.2ms
Speed: 0.5ms preprocess, 8.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.70, hippopotamus 0.18, elephant 0.12, 11.4ms
Speed: 0.5ms preprocess, 11.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.11, 12.7ms
Speed: 0.5ms preprocess, 12.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 11.6ms
Speed: 0.5ms preprocess, 11.6ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.72, hippopotamus 0.16, elephant 0.12, 13.7ms
Speed: 0.5ms preprocess, 13.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 11.6ms
Speed: 0.5ms preprocess, 11.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.16, elephant 0.13, 10.4ms
Speed: 0.5ms preprocess, 10.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 14.8ms
Speed: 0.6ms preprocess, 14.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 7.4ms
Speed: 0.6ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.70, hippopotamus 0.17, elephant 0.13, 7.5ms
Speed: 0.6ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.72, hippopotamus 0.16, elephant 0.12, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.70, hippopotamus 0.18, elephant 0.12, 7.6ms
Speed: 0.4ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 7.7ms
Speed: 0.5ms preprocess, 7.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.71, hippopotamus 0.17, elephant 0.12, 7.4ms
Speed: 0.4ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.72, hippopotamus 0.17, elephant 0.11, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.72, hippopotamus 0.16, elephant 0.12, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      46/50     0.661G   0.006824         11         64: 100%|██████████| 30/30 [00:03<00:00,  8.06it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 23.13it/s]
                   all      0.864          1

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 29.7ms
Speed: 0.6ms preprocess, 29.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.24, elephant 0.22, 24.7ms
Speed: 0.6ms preprocess, 24.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 19.3ms
Speed: 0.6ms preprocess, 19.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 25.1ms
Speed: 0.6ms preprocess, 25.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.24, elephant 0.23, 19.4ms
Speed: 0.5ms preprocess, 19.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.24, elephant 0.22, 14.8ms
Speed: 0.6ms preprocess, 14.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 19.6ms
Speed: 0.6ms preprocess, 19.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 16.4ms
Speed: 0.6ms preprocess, 16.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.25, elephant 0.21, 26.7ms
Speed: 0.6ms preprocess, 26.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.24, elephant 0.22, 26.9ms
Speed: 0.6ms preprocess, 26.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 11.2ms
Speed: 0.6ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.24, elephant 0.21, 12.5ms
Speed: 0.6ms preprocess, 12.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.25, elephant 0.21, 10.6ms
Speed: 0.5ms preprocess, 10.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.25, elephant 0.22, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 6.9ms
Speed: 0.5ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.52, elephant 0.24, hippopotamus 0.24, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 6.5ms
Speed: 0.5ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.26, elephant 0.22, 6.4ms
Speed: 1.6ms preprocess, 6.4ms inference, 2.8ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.25, elephant 0.22, 6.3ms
Speed: 0.6ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.54, hippopotamus 0.24, elephant 0.22, 6.6ms
Speed: 0.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.53, hippopotamus 0.25, elephant 0.22, 6.6ms
Speed: 0.7ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      47/50     0.661G   0.006694         11         64: 100%|██████████| 30/30 [00:04<00:00,  6.56it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 41.80it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.59, hippopotamus 0.28, elephant 0.13, 7.4ms
Speed: 0.6ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.30, elephant 0.13, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.29, elephant 0.13, 6.5ms
Speed: 0.9ms preprocess, 6.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.28, elephant 0.14, 7.4ms
Speed: 0.6ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.28, elephant 0.13, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.60, hippopotamus 0.27, elephant 0.13, 10.5ms
Speed: 0.5ms preprocess, 10.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.28, elephant 0.13, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.27, elephant 0.13, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.28, elephant 0.14, 7.8ms
Speed: 0.5ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.29, elephant 0.14, 10.0ms
Speed: 0.5ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.28, elephant 0.13, 18.5ms
Speed: 0.6ms preprocess, 18.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.29, elephant 0.13, 11.0ms
Speed: 0.5ms preprocess, 11.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.28, elephant 0.13, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.29, elephant 0.13, 16.3ms
Speed: 0.5ms preprocess, 16.3ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.28, elephant 0.13, 13.3ms
Speed: 0.5ms preprocess, 13.3ms inference, 2.7ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.60, hippopotamus 0.28, elephant 0.13, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.59, hippopotamus 0.28, elephant 0.13, 6.7ms
Speed: 0.6ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.29, elephant 0.13, 7.3ms
Speed: 0.7ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.28, elephant 0.14, 6.6ms
Speed: 0.7ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.28, elephant 0.15, 7.2ms
Speed: 0.6ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.58, hippopotamus 0.28, elephant 0.14, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.57, hippopotamus 0.29, elephant 0.14, 6.7ms
Speed: 0.6ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      48/50     0.661G   0.006607         11         64: 100%|██████████| 30/30 [00:03<00:00,  7.86it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 15.93it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.05, 6.4ms
Speed: 0.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.05, 6.0ms
Speed: 0.5ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.05, 6.8ms
Speed: 0.5ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.19, elephant 0.06, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.06, 7.3ms
Speed: 0.5ms preprocess, 7.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.05, 11.0ms
Speed: 0.5ms preprocess, 11.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.05, 7.0ms
Speed: 0.4ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.06, 6.6ms
Speed: 0.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.17, elephant 0.06, 10.2ms
Speed: 0.6ms preprocess, 10.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.17, elephant 0.05, 20.2ms
Speed: 0.4ms preprocess, 20.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.17, elephant 0.05, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.06, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.05, 6.7ms
Speed: 1.4ms preprocess, 6.7ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.06, 16.8ms
Speed: 0.6ms preprocess, 16.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.05, 17.8ms
Speed: 0.6ms preprocess, 17.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.06, 6.4ms
Speed: 0.6ms preprocess, 6.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.06, 10.0ms
Speed: 0.5ms preprocess, 10.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.06, 9.0ms
Speed: 0.6ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.06, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.17, elephant 0.05, 6.0ms
Speed: 0.4ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.18, elephant 0.05, 6.8ms
Speed: 0.7ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.18, elephant 0.06, 9.2ms
Speed: 0.5ms preprocess, 9.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      49/50     0.661G   0.003647         11         64: 100%|██████████| 30/30 [00:04<00:00,  6.09it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 20.57it/s]
                   all      0.818          1

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.16, elephant 0.08, 10.1ms
Speed: 0.5ms preprocess, 10.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.15, elephant 0.07, 8.4ms
Speed: 1.6ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.15, elephant 0.07, 8.4ms
Speed: 0.5ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.15, elephant 0.07, 15.8ms
Speed: 0.6ms preprocess, 15.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.15, elephant 0.08, 10.9ms
Speed: 0.5ms preprocess, 10.9ms inference, 4.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 16.4ms
Speed: 0.6ms preprocess, 16.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 8.6ms
Speed: 0.6ms preprocess, 8.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 10.9ms
Speed: 1.4ms preprocess, 10.9ms inference, 0.8ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.78, hippopotamus 0.15, elephant 0.07, 6.8ms
Speed: 0.6ms preprocess, 6.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 15.8ms
Speed: 0.6ms preprocess, 15.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 14.6ms
Speed: 0.5ms preprocess, 14.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.78, hippopotamus 0.15, elephant 0.07, 18.5ms
Speed: 0.6ms preprocess, 18.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.15, elephant 0.07, 13.7ms
Speed: 1.8ms preprocess, 13.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.76, hippopotamus 0.16, elephant 0.07, 12.0ms
Speed: 0.5ms preprocess, 12.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.77, hippopotamus 0.16, elephant 0.07, 7.2ms
Speed: 0.5ms preprocess, 7.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.78, hippopotamus 0.15, elephant 0.07, 9.3ms
Speed: 0.5ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.78, hippopotamus 0.15, elephant 0.07, 7.5ms
Speed: 0.6ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.78, hippopotamus 0.15, elephant 0.07, 8.4ms
Speed: 0.6ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

      Epoch    GPU_mem       loss  Instances       Size
      50/50     0.661G   0.004417         11         64: 100%|██████████| 30/30 [00:04<00:00,  7.49it/s]
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 20.92it/s]
                   all      0.864          1

0: 64x64 rhinoceros 0.61, hippopotamus 0.30, elephant 0.09, 13.0ms
Speed: 0.6ms preprocess, 13.0ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.08, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 8.1ms
Speed: 0.8ms preprocess, 8.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.08, 7.4ms
Speed: 0.5ms preprocess, 7.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.28, elephant 0.10, 8.4ms
Speed: 0.7ms preprocess, 8.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.64, hippopotamus 0.28, elephant 0.08, 7.5ms
Speed: 0.5ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.09, 11.2ms
Speed: 2.2ms preprocess, 11.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 6.9ms
Speed: 3.3ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.09, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.64, hippopotamus 0.27, elephant 0.09, 6.6ms
Speed: 1.8ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 6.1ms
Speed: 0.9ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.30, elephant 0.09, 11.3ms
Speed: 0.9ms preprocess, 11.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.09, 6.3ms
Speed: 0.5ms preprocess, 6.3ms inference, 0.6ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.09, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.09, 6.6ms
Speed: 0.6ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.08, 5.8ms
Speed: 0.5ms preprocess, 5.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 6.1ms
Speed: 0.6ms preprocess, 6.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.64, hippopotamus 0.28, elephant 0.08, 8.5ms
Speed: 0.5ms preprocess, 8.5ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 6.7ms
Speed: 0.5ms preprocess, 6.7ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.30, elephant 0.08, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.09, 7.6ms
Speed: 0.5ms preprocess, 7.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.08, 6.0ms
Speed: 0.5ms preprocess, 6.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

50 epochs completed in 0.092 hours.
Optimizer stripped from runs/classify/train3/weights/last.pt, 31.7MB
Optimizer stripped from runs/classify/train3/weights/best.pt, 31.7MB

Validating runs/classify/train3/weights/best.pt...
Ultralytics YOLOv8.0.186 🚀 Python-3.10.12 torch-2.0.1+cu118 CUDA:0 (Tesla T4, 15102MiB)
YOLOv8m-cls summary (fused): 103 layers, 15766499 parameters, 0 gradients
WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.
train: /content/datasets/New-Animals-Classification-1/train... found 471 images in 4 classes: ERROR ❌️ requires 3 classes, not 4
val: None...
test: /content/datasets/New-Animals-Classification-1/test... found 22 images in 3 classes ✅ 
               classes   top1_acc   top5_acc: 100%|██████████| 1/1 [00:00<00:00, 62.81it/s]
                   all      0.909          1
Speed: 0.0ms preprocess, 0.4ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to runs/classify/train3

0: 64x64 rhinoceros 0.64, hippopotamus 0.28, elephant 0.08, 7.5ms
Speed: 0.6ms preprocess, 7.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 6.6ms
Speed: 0.5ms preprocess, 6.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.30, elephant 0.09, 6.2ms
Speed: 0.4ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.09, 7.1ms
Speed: 0.5ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.09, 6.9ms
Speed: 0.6ms preprocess, 6.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.30, elephant 0.08, 6.3ms
Speed: 0.4ms preprocess, 6.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 30.4ms
Speed: 0.5ms preprocess, 30.4ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.64, hippopotamus 0.28, elephant 0.08, 7.1ms
Speed: 0.6ms preprocess, 7.1ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.09, 9.0ms
Speed: 0.5ms preprocess, 9.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.08, 8.6ms
Speed: 0.5ms preprocess, 8.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.08, 9.3ms
Speed: 3.1ms preprocess, 9.3ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.64, hippopotamus 0.27, elephant 0.09, 8.1ms
Speed: 0.5ms preprocess, 8.1ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.09, 7.0ms
Speed: 0.6ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.08, 6.2ms
Speed: 0.5ms preprocess, 6.2ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.61, hippopotamus 0.30, elephant 0.09, 9.8ms
Speed: 0.5ms preprocess, 9.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 7.0ms
Speed: 0.5ms preprocess, 7.0ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.29, elephant 0.08, 9.5ms
Speed: 0.4ms preprocess, 9.5ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.28, elephant 0.10, 9.1ms
Speed: 0.5ms preprocess, 9.1ms inference, 0.2ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.09, 8.8ms
Speed: 0.5ms preprocess, 8.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.63, hippopotamus 0.28, elephant 0.09, 12.6ms
Speed: 0.6ms preprocess, 12.6ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 7.8ms
Speed: 1.7ms preprocess, 7.8ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)

0: 64x64 rhinoceros 0.62, hippopotamus 0.29, elephant 0.09, 7.9ms
Speed: 0.5ms preprocess, 7.9ms inference, 0.1ms postprocess per image at shape (1, 3, 64, 64)
Results saved to runs/classify/train3
  ```
</details>

### Evidências do treinamento

Nessa seção você deve colocar qualquer evidência do treinamento, como por exemplo gráficos de perda, performance, matriz de confusão etc.

- Matrix de confusão:
![confusion_matrix](https://github.com/aparecidoSilvano/projeto-cesar-redes-neurais/assets/7593828/53af44cb-189b-47e0-a202-1acd36fc2e8c)

- Resultados:

![download](https://github.com/aparecidoSilvano/projeto-cesar-redes-neurais/assets/7593828/c2d88726-2cf9-4c23-8e75-8055ff6feb2d)

![download (1)](https://github.com/aparecidoSilvano/projeto-cesar-redes-neurais/assets/7593828/94b4b66a-2a7e-49b4-899a-0eaefa7fa796)

![results](https://github.com/aparecidoSilvano/projeto-cesar-redes-neurais/assets/7593828/7ab3344a-c6a7-4313-a3d2-79816ba617f8)

## Google Colab
[Link para o colab]([https://colab.research.google.com/drive/16jPZSnBCyxJLdHlKRHi4d_IgVzmhVsbC?usp=sharing](https://colab.research.google.com/drive/144BjnEWdPnjpnEUxlYqrx16Wk2Y6sp_k?usp=sharing))

## Roboflow

[Link para o Roboflow]( https://universe.roboflow.com/cesar-school-recife/new-animals-classification )

## HuggingFace

[Link para o Huggingface](https://huggingface.co/spaces/silvanoalbuquerque/YOLO-V8_ANIMALS_CLASSIFICATION )
