import os
import glob
import fire
from PIL import Image, ImageDraw
from collections import Counter
import matplotlib.pyplot as plt


def check_paletted_images(folder="data"):
    """
    Checks all PNG files in the given folder to see if they are paletted,
    if they share the same palette, counts all the pixel values (palette indices),
    and draws a bar chart. Also, stores the palette in the 'output' directory
    if all files have the same palette and renders a square PNG image of the palette.
    
    :param folder: The folder to check for PNG files.
    """
    png_files = glob.glob(os.path.join(folder, "*.png"))
    if not png_files:
        print("No PNG files found in the folder.")
        return
    
    palettes = []
    pixel_counters = Counter()
    
    for file in png_files:
        with Image.open(file) as img:
            if img.mode == "P":
                palette = get_palette(img)
                palettes.append((file, palette))
                print(f"{file} is paletted.")
                
                # Count pixel values (palette indices)
                pixel_data = list(img.getdata())
                pixel_counters.update(pixel_data)
            else:
                print(f"{file} is not paletted.")
                return

    # Check if all paletted images have the same palette
    reference_palette = palettes[0][1]  # The palette of the first image
    all_same_palette = all(palette == reference_palette for _, palette in palettes)
    
    if all_same_palette:
        print("All paletted images have the same palette.")
        save_palette(reference_palette, "output/palette.pal")
        render_palette_image(reference_palette, "output/palette.png")
    else:
        print("Not all paletted images share the same palette.")
    
    # Draw bar chart of pixel values (palette indices)
    draw_bar_chart(pixel_counters, "output/pixel_value_counts.png")


def get_palette(image):
    """
    Returns the palette of a paletted image as a list.
    
    :param image: The PIL Image object.
    :return: List representing the image palette.
    """
    return image.getpalette()


def save_palette(palette, output_path):
    """
    Saves the palette to a file.
    
    :param palette: The palette list to save.
    :param output_path: The file path to save the palette.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for i in range(0, len(palette), 3):
            f.write(f"{palette[i]}, {palette[i+1]}, {palette[i+2]}\n")


def draw_bar_chart(counter, output_path):
    """
    Draws a bar chart of the pixel value counts and saves it to a file.
    
    :param counter: Counter object with pixel values as keys and counts as values.
    :param output_path: The file path to save the chart.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(counter.keys(), counter.values())
    plt.xlabel("Palette Index")
    plt.ylabel("Count")
    plt.title("Pixel Value Counts")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Bar chart saved to {output_path}")


def render_palette_image(palette, output_path):
    """
    Renders a square PNG image of the palette and saves it to a file.
    
    :param palette: The palette list to render.
    :param output_path: The file path to save the palette image.
    """
    num_colors = len(palette) // 3
    size = int(num_colors**0.5)
    if size * size < num_colors:
        size += 1
    
    palette_image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(palette_image)
    
    for i in range(num_colors):
        color = (palette[3 * i], palette[3 * i + 1], palette[3 * i + 2])
        x = i % size
        y = i // size
        draw.point((x, y), fill=color)
    
    palette_image = palette_image.resize((256, 256), Image.NEAREST)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    palette_image.save(output_path)
    print(f"Palette image saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(check_paletted_images)
