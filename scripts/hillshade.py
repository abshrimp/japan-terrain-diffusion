import os
import sys
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


def calculate_hillshade(elevation, azimuth=315, angle_altitude=45):
    """元コードと完全に同じ陰影図"""

    dx, dy = np.gradient(elevation)

    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(angle_altitude)

    shaded = (
        np.sin(altitude_rad) * np.sin(slope)
        + np.cos(altitude_rad)
        * np.cos(slope)
        * np.cos(azimuth_rad - aspect)
    )

    hillshade = 255 * (shaded + 1) / 2

    return np.clip(hillshade, 0, 255).astype(np.uint8)


def create_colored_hillshade(dem):
    """元コードと同じカラー陰影図"""

    ls = LightSource(
        azdeg=315,
        altdeg=45,
    )

    rgb = ls.shade(
        dem,
        cmap=plt.cm.terrain,
        vert_exag=2.0,
        blend_mode="overlay",
    )

    return rgb


def main(input_file):
    gray_output = input_file.replace('.tif', '_hs.png')
    color_output = input_file.replace('.tif', '_colored_hillshade.png')

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # GeoTIFF読込
    with rasterio.open(input_file) as src:
        dem = src.read(1)

    dem = np.nan_to_num(dem)

    print(f"Loaded: {input_file}")
    print(f"Shape : {dem.shape}")
    print(f"Min   : {dem.min():.2f} m")
    print(f"Max   : {dem.max():.2f} m")

    # 白黒陰影図
    hillshade = calculate_hillshade(dem)
    Image.fromarray(hillshade).save(gray_output)
    print(f"Saved: {gray_output}")

    # カラー陰影図
    rgb = create_colored_hillshade(dem)
    plt.imsave(color_output, rgb)
    print(f"Saved: {color_output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hillshade.py <input_dem.tif>")
    else:
        main(sys.argv[1])