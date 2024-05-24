import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random
import glob
import cv2
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class ReplaceGreenScreenBackground:
    def __init__(self, background_images_path):
        self.background_images = glob.glob(str(background_images_path))
    
    def __call__(self, img, transform):
        try:
            # Convert PIL Image to NumPy array
            img_np = np.array(img)
            
            # Convert the image from RGB to HSV color space
            hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            
            # Define the green color range in HSV
            lower_green = np.array([37, 40, 40])
            upper_green = np.array([80, 255, 255])
            
            # Create a mask for the green background
            mask = cv2.inRange(hsv_img, lower_green, upper_green)
            
            # Apply morphological operations to smooth the mask
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((13, 13), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Invert the mask
            mask_inv = cv2.bitwise_not(mask)
            
            # Load a random background image
            bg_img = Image.open(random.choice(self.background_images))
            bg_img = bg_img.resize(img.size)
            bg_np = np.array(bg_img)
            bg_np = cv2.GaussianBlur(bg_np, (15, 15), 0)
            
            # Combine the background image and the original image using the mask
            fg = cv2.bitwise_and(img_np, img_np, mask=mask_inv)
            bg = cv2.bitwise_and(bg_np, bg_np, mask=mask)
            if bg.ndim == 2:
                bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
            if fg.ndim == 2:
                fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2RGB)
            combined_img = cv2.add(fg, bg)
            
            # Convert NumPy array back to PIL Image
            img = transform(Image.fromarray(combined_img))
            
            return img
        except:
            overwatch.info("Error in ReplaceGreenScreenBackground, returning original image")
            return transform(img)


class ReplaceBackgroundWithSegmentation:
    def __init__(self, background_images_path, device='cpu'):
        self.background_images = glob.glob(background_images_path)
        self.device = device
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, img, transform):
        try:
            # Ensure the image is in RGB format
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Preprocess the image
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            # Generate the segmentation mask
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
            
            # Verify if the output is valid
            if output.shape[0] != 21:  # Expecting 21 classes for COCO model
                raise ValueError("Unexpected output shape from the model")
            
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
            # Create a mask for the background
            mask = output_predictions == 0
            
            # If no foreground detected, return original image (for debugging)
            if np.sum(mask) == 0:
                print("No foreground detected")
                return img
            
            # Load a random background image
            bg_img = Image.open(random.choice(self.background_images))
            bg_img = bg_img.resize(img.size)
            bg_np = np.array(bg_img)
            
            # Replace the background
            img_np = np.array(img)
            img_np[mask] = bg_np[mask]
            
            # Convert NumPy array back to PIL Image
            img = Image.fromarray(img_np)
            
            return transform(img)
        except:
            return transform(img)



if __name__ == '__main__':
    # Example usage in a transforms pipeline
    background_images_path = 'data/download/llava-v1.5-instruct/coco/train2017/*.jpg'
    foreground_images_path = 'data/download/mmwand/*/*/_rgb*.png'
    foreround_images = glob.glob(foreground_images_path)

    custom_transforms = transforms.Compose([
        ReplaceGreenScreenBackground(background_images_path),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load your image
    foreround_image = random.choice(foreround_images)
    print(f"Processing image: {foreround_image}")
    img = Image.open(foreround_image)

    # Apply the transforms
    transformed_img = custom_transforms(img)

    # To verify, let's visualize the transformed image using matplotlib (optional)
    import matplotlib.pyplot as plt

    plt.imshow(transformed_img.permute(1, 2, 0))
    plt.savefig('replaced_green_background.png')
    plt.show()
