"""
Generates synthetic endoscopic polyp sample frames and ground truth masks
for immediate live demonstration, smoke-testing, and client evaluation.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_sample_endoscopy_data(output_dir: str = "data/sample_images"):
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        {
            "name": "sample_polyp_sessile",
            "bg_color": (150, 45, 55),
            "polyp_pos": (128, 120, 65, 50),
            "polyp_color": (195, 80, 85),
            "has_polyp": True
        },
        {
            "name": "sample_polyp_elevated",
            "bg_color": (135, 40, 45),
            "polyp_pos": (160, 150, 45, 45),
            "polyp_color": (185, 75, 75),
            "has_polyp": True
        },
        {
            "name": "sample_normal_mucosa",
            "bg_color": (140, 42, 50),
            "polyp_pos": None,
            "polyp_color": None,
            "has_polyp": False
        }
    ]

    generated_files = []

    for s in samples:
        # 1. Base endoscopic image (256x256)
        img = Image.new("RGB", (256, 256), color=s["bg_color"])
        draw = ImageDraw.Draw(img)

        # Draw endoscopic lighting vignette and mucosal folds
        for r in range(128, 0, -8):
            alpha = int(255 * (r / 128))
            draw.ellipse([128 - r, 128 - r, 128 + r, 128 + r], outline=(s["bg_color"][0] + 15, s["bg_color"][1] + 10, s["bg_color"][2] + 10))

        # Add mucosal capillary texture
        mucosa = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, 12, mucosa.shape)
        mucosa = np.clip(mucosa + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(mucosa)

        # 2. Binary mask
        mask = Image.new("L", (256, 256), color=0)
        mask_draw = ImageDraw.Draw(mask)

        if s["has_polyp"] and s["polyp_pos"]:
            cx, cy, rx, ry = s["polyp_pos"]
            # Draw irregular lesion boundary
            pts = []
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                rad = 1.0 + np.random.uniform(-0.15, 0.15)
                px = cx + rx * rad * np.cos(angle)
                py = cy + ry * rad * np.sin(angle)
                pts.append((px, py))

            mask_draw.polygon(pts, fill=255)

            # Draw lesion texture on original image
            img_draw = ImageDraw.Draw(img)
            img_draw.polygon(pts, fill=s["polyp_color"])
            # Add specular reflection (endoscopic light reflection)
            img_draw.ellipse([cx - 10, cy - 10, cx + 5, cy + 2], fill=(245, 230, 230))

        # Smooth edges slightly
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

        img_path = os.path.join(output_dir, f"{s['name']}.jpg")
        mask_path = os.path.join(output_dir, f"{s['name']}_mask.png")

        img.save(img_path, "JPEG", quality=95)
        mask.save(mask_path, "PNG")

        generated_files.append((img_path, mask_path))

    return generated_files


if __name__ == "__main__":
    files = generate_sample_endoscopy_data()
    print(f"Generated {len(files)} sample endoscopic evaluation frames.")
