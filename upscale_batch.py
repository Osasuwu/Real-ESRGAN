import os
import gc
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
import cv2
import torch
from tqdm import tqdm

# Модель (realesr-general-x4v3 — лёгкая, меньше VRAM)
# Скачай файл веса и положи в папку "weights":
# https://github.com/xinntao/Real-ESRGAN/releases
model_name = os.path.join('weights', 'realesr-general-x4v3.pth')
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

# Проверка GPU
use_cuda = torch.cuda.is_available()
if use_cuda:
    print(f'GPU обнаружен: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('GPU не обнаружен, используется CPU (будет медленно)')

# Инициализация upscaler
upsampler = RealESRGANer(
    scale=2,  # Для 1080p→2K
    model_path=model_name,
    model=model,
    tile=0,  # 0 = без тайлинга (обработка целиком). Если VRAM не хватит, верни 400
    tile_pad=10,
    pre_pad=0,
    half=use_cuda,  # FP16 для GPU
    gpu_id=0 if use_cuda else None
)

input_folder = 'inputs'
output_folder = 'outputs'

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not os.path.isfile(model_name):
    raise FileNotFoundError(f'Не найден файл модели: {model_name}. Скачай его и помести в папку weights/')

if not images:
    print(f'Положи изображения (.png/.jpg) в папку {input_folder} и запусти скрипт снова.')
else:
    print(f'\n Найдено {len(images)} изображений для апскейла x2\n')

    for idx, filename in enumerate(images, 1):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            print(f'\n⚠ Ошибка чтения: {filename}')
            continue

        # Проверка разрешения входного изображения
        h, w = img.shape[:2]
        max_dim = max(h, w)

        # Динамический выбор scale для достижения ~2K (2560x1440 min)
        if max_dim <= 1920:  # FullHD и ниже
            scale = 2  # FullHD → 4K
            scale_reason = 'FullHD→4K (x2)'
        elif max_dim < 3840:  # 2K-4K
            scale = 1  # Уже 2K+, не апскейлим
            scale_reason = f'уже 2K+ ({max_dim}p), без изменений'
        else:  # 8K и выше
            scale = 1  # Уже 8K, просто копируем
            scale_reason = f'8K+ ({max_dim}p), без изменений'

        try:
            if scale == 1:
                # Копируем без изменений
                output = img
            else:
                # Апскейлим с динамическим масштабом
                upsampler_temp = RealESRGANer(
                    scale=scale,
                    model_path=model_name,
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=use_cuda,
                    gpu_id=0 if use_cuda else None
                )
                output, _ = upsampler_temp.enhance(img, outscale=scale)

            output_path = os.path.join(output_folder, f'upscaled_{filename}')
            cv2.imwrite(output_path, output)

            # Информация о результате
            out_h, out_w = output.shape[:2]
            tqdm.write(f'  [{idx}/{len(images)}] {filename} ({h}x{w} → {out_h}x{out_w}) [{scale_reason}]')

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'\n❌ CUDA OOM на {filename} ({h}x{w}). Пропускаю.')
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f'\n❌ Ошибка обработки {filename}: {e}')

        # Очистка памяти каждые 10 изображений
        if idx % 10 == 0:
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

    print(f'\n✓ Готово! Обработано {len(images)} изображений → {output_folder}/')
