import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import get_model
from dataset import PalettedImageDataset
from PIL import Image
import random
import imageio
import time
import datetime
import moviepy.editor as mpy

def temperature_sample(logits, temperature=1.0):
    """
    Applies temperature sampling to logits to generate a probability distribution.
    
    :param logits: The logits output by the model.
    :param temperature: The temperature to use for sampling.
    :return: The sampled index.
    """
    logits = logits / temperature
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, 1).item()

def apply_cellular_automaton(model, initial_image, square_size, num_steps, temperature=1.0, output_folder="output"):
    """
    Applies the trained model to the initial image using cellular automaton rules,
    saves each step to a folder, and creates a GIF.
    
    :param model: The trained model.
    :param initial_image: The initial image as a NumPy array of palette indices.
    :param square_size: The size of the square region around each pixel.
    :param num_steps: The number of steps to apply the cellular automaton.
    :param temperature: The temperature to use for sampling.
    :param output_folder: The folder to save each step's image.
    :return: The resulting image as a NumPy array of palette indices.
    """
    os.makedirs(output_folder, exist_ok=True)
    model.eval()
    half_size = square_size // 2
    height, width = initial_image.shape

    # Load the palette for visualization
    palette_file = os.path.join("data", 'palette.pal')
    with open(palette_file, 'r') as f:
        palette = [tuple(map(int, line.strip().split(','))) for line in f]

    images = []
    for step in range(num_steps):
        new_image = initial_image.copy()
        for y in range(height):
            for x in range(width):
                region = []
                for i in range(-half_size, half_size + 1):
                    for j in range(-half_size, half_size + 1):
                        if i == 0 and j == 0:
                            continue
                        y_idx = (y + i) % height
                        x_idx = (x + j) % width
                        region.append(initial_image[y_idx, x_idx])
                region = torch.tensor(region, dtype=torch.long).unsqueeze(0)  # Shape: (1, (square_size^2)-1)
                with torch.no_grad():
                    logits = model(region)
                    new_image[y, x] = temperature_sample(logits, temperature)
        initial_image = new_image

        # Convert the resulting image to RGB for visualization
        resulting_image_rgb = np.array([palette[idx] for row in initial_image for idx in row]).reshape((height, width, 3))
        
        # Save the image step
        image_path = os.path.join(output_folder, f"step_{step:03d}.png")
        imageio.imwrite(image_path, resulting_image_rgb)
        images.append(resulting_image_rgb)

    # Datetime as YYYYMMDD_HHMM
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Create a GIF from the saved images
    gif_path = os.path.join(output_folder, f"cellular_automaton_{time_stamp}.gif")
    imageio.mimsave(gif_path, images, fps=5)
    print(f"GIF saved to {gif_path}")

    # Also create a PNG of the final image
    final_image_path = os.path.join(output_folder, f"final_image_{time_stamp}.png")
    imageio.imwrite(final_image_path, resulting_image_rgb)
    print(f"Final image saved to {final_image_path}")

    # Create a video from the saved images.
    video_path = os.path.join(output_folder, f"cellular_automaton_{time_stamp}.mp4")
    clip = mpy.ImageSequenceClip(images, fps=5)
    clip.write_videofile(video_path)
    print(f"Video saved to {video_path}")

    return initial_image

def load_model_from_checkpoint(config, checkpoint_path):
    """
    Loads the model from the checkpoint, handling any key mismatches.
    
    :param config: Configuration dictionary for the model.
    :param checkpoint_path: Path to the checkpoint file.
    :return: The loaded model.
    """
    model = get_model(config)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix from keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v  # Remove 'model.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model

def main():
    # Load the model
    config = {
        "square_size": 5,
        "embedding_size": 64,
        "num_heads": 4,
        "num_encoders": 2,
        "num_classes": 256,
        "learning_rate": 0.001,
        "data_folder": "data",
        "batch_size": 32,
        "num_workers": 4
    }
    checkpoint_path = "logs/best-checkpoint.ckpt"
    model = load_model_from_checkpoint(config, checkpoint_path)

    # Create a random initial image
    dataset = PalettedImageDataset(folder=config["data_folder"], square_length=config["square_size"])
    initial_image = random.choice(dataset.images)
    height, width = initial_image.shape

    # Randomly set pixels to 0.
    for x in range(width):
        for y in range(height):
            if random.random() < 0.1:
                initial_image[y, x] = 0

    # Create a noisy initial image.
    #height, width = 100, 100
    #initial_image = np.random.randint(256, size=(height, width))

    # Apply the cellular automaton
    num_steps = 100
    temperature = 1.0
    output_folder = "output"
    resulting_image = apply_cellular_automaton(model, initial_image, config["square_size"], num_steps, temperature, output_folder)

    # Load the palette for visualization
    palette_file = os.path.join(config["data_folder"], 'palette.pal')
    with open(palette_file, 'r') as f:
        palette = [tuple(map(int, line.strip().split(','))) for line in f]

    # Convert the resulting image to RGB for visualization
    resulting_image_rgb = np.array([palette[idx] for row in resulting_image for idx in row]).reshape((height, width, 3))

    # Display the resulting image
    plt.imshow(resulting_image_rgb)
    plt.title(f"Resulting Image after {num_steps} steps")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
