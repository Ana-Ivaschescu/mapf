import rasterio
from rasterio.enums import Resampling
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

def read_tif_to_array(tif_path):
    with rasterio.open(tif_path) as src:
        raster_array = src.read()

    return raster_array, src

def create_and_save_tiles(raster_array, src, tile_size, output_folder):
    height, width = raster_array.shape[1:]

    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = raster_array[:, i:i+tile_size, j:j+tile_size]
            save_path = os.path.join(output_folder, f'tile_{i}_{j}.tif')

            with rasterio.open(
                save_path,
                'w',
                driver='GTiff',
                height=tile.shape[1],
                width=tile.shape[2],
                count=raster_array.shape[0],
                dtype=raster_array.dtype,
                crs=src.crs,
                transform=rasterio.windows.transform(window=((i, i+tile_size), (j, j+tile_size)), transform=src.transform),
                compress="jpeg",
            ) as dst:
                dst.write(tile)


# Tile size (500 by 500)
tile_size = 500

directory_path = r'C:\mapformer\data/HRSCD/images/'
out_dir = r'C:\mapformer\data/preprocessed/500'
for year in ['2006', '2012']:
    for d in ['D14', 'D35']:
        files = os.listdir(osp.join(directory_path, year, d))
        for file in tqdm(files):
            name, suffix = file.split('.')
            file_path = osp.join(directory_path, year, d, file) 
            raster_array, src = read_tif_to_array(file_path)
            os.makedirs(osp.join(out_dir, 'images', year, d, name), exist_ok=True)
            output_folder = osp.join(out_dir, 'images', year, d, name)
            create_and_save_tiles(raster_array, src, tile_size, output_folder)