import cv2
import numpy as np
from PIL import Image

def apply_clahe(image: Image.Image) -> Image.Image:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel of LAB space."""
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    img_enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img_enhanced_rgb = cv2.cvtColor(img_enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_enhanced_rgb)

def adjust_gamma(image: Image.Image, gamma: float = 1.0) -> Image.Image:
    """Adjust image gamma value."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    img_np = np.array(image.convert('RGB'))
    img_gamma = cv2.LUT(img_np, table)
    
    return Image.fromarray(img_gamma)

def apply_white_balance(image: Image.Image) -> Image.Image:
    """Apply simple Gray-World white balance algorithm to correct lighting color shifts."""
    img_np = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    try:
        # Requires opencv-contrib-python
        wb = cv2.xphoto.createSimpleWB()
        balanced_bgr = wb.balanceWhite(img_bgr)
    except AttributeError:
        # Fallback manual gray-world algorithm
        balanced_bgr = img_bgr.copy().astype(np.float32)
        avg_b = np.mean(balanced_bgr[:, :, 0])
        avg_g = np.mean(balanced_bgr[:, :, 1])
        avg_r = np.mean(balanced_bgr[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        
        balanced_bgr[:, :, 0] = np.clip(balanced_bgr[:, :, 0] * (avg_gray / (avg_b + 1e-5)), 0, 255)
        balanced_bgr[:, :, 1] = np.clip(balanced_bgr[:, :, 1] * (avg_gray / (avg_g + 1e-5)), 0, 255)
        balanced_bgr[:, :, 2] = np.clip(balanced_bgr[:, :, 2] * (avg_gray / (avg_r + 1e-5)), 0, 255)
        balanced_bgr = balanced_bgr.astype(np.uint8)
        
    img_enhanced_rgb = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_enhanced_rgb)

if __name__ == "__main__":
    print("lighting_enhance module loaded. Ready to apply CLAHE, Gamma correction, or White Balance.")
