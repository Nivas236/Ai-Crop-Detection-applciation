"""
Direct API access to the Plant Disease Dataset using Kaggle API
"""

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class PlantDiseaseDataLoader:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        self.dataset_name = "vipoooool/new-plant-diseases-dataset"
        
    def get_available_classes(self):
        """Get list of available plant disease classes"""
        # Classes are pre-defined based on the dataset structure
        # No need to fetch metadata as we know the structure
        return [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot",
            "Peach___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Strawberry___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy"
        ]

    def load_sample_images(self, class_name, num_samples=10):
        """Load sample images from a specific class using Kaggle API"""
        try:
            # Create a temporary directory for downloads
            temp_dir = os.path.join(os.getcwd(), 'temp_downloads')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download only the specific class folder
            path = f"New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/{class_name}"
            self.api.dataset_download_files(
                self.dataset_name,
                path=path,
                quiet=True,
                unzip=True,
                target_dir=temp_dir
            )
            
            # Get list of downloaded image files
            class_path = os.path.join(temp_dir, path)
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
            
            if not image_files:
                raise ValueError(f"No images found for class {class_name}")
            
            images = []
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img = load_img(img_path, target_size=(224, 224))
                img_array = img_to_array(img)
                images.append(img_array)
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
            return np.array(images)
        
        except Exception as e:
            print(f"Error loading images for {class_name}: {str(e)}")
            return None

    def get_training_batch(self, class_names, batch_size=32):
        """Get a batch of training images for specified classes"""
        images = []
        labels = []
        
        # Calculate samples per class to maintain balance
        samples_per_class = max(1, batch_size // len(class_names))
        
        # Select a subset of classes for this batch
        num_classes_per_batch = min(len(class_names), batch_size)
        selected_classes = np.random.choice(class_names, size=num_classes_per_batch, replace=False)
        
        for i, class_name in enumerate(selected_classes):
            try:
                class_images = self.load_sample_images(class_name, num_samples=samples_per_class)
                if class_images is not None and len(class_images) > 0:
                    images.extend(class_images)
                    
                    # Create one-hot encoded labels
                    label = np.zeros(len(class_names))
                    label[class_names.index(class_name)] = 1
                    labels.extend([label] * len(class_images))
            except Exception as e:
                print(f"Error loading class {class_name}: {str(e)}")
                continue
        
        if len(images) == 0 or len(labels) == 0:
            # If no images were loaded, return dummy data
            dummy_image = np.zeros((224, 224, 3))
            dummy_label = np.zeros(len(class_names))
            dummy_label[0] = 1  # First class as default
            return np.array([dummy_image]), np.array([dummy_label])
        
        # Shuffle the data
        combined = list(zip(images, labels))
        np.random.shuffle(combined)
        images, labels = zip(*combined)
        
        # Convert to numpy arrays
        images_array = np.array(images)
        labels_array = np.array(labels)
        
        # Ensure we return exactly batch_size items
        if len(images_array) > batch_size:
            return images_array[:batch_size], labels_array[:batch_size]
        elif len(images_array) < batch_size:
            # Pad with duplicates if needed
            num_to_pad = batch_size - len(images_array)
            indices = np.random.choice(len(images_array), size=num_to_pad)
            images_array = np.concatenate([images_array, images_array[indices]])
            labels_array = np.concatenate([labels_array, labels_array[indices]])
        
        return images_array, labels_array

# Example usage:
if __name__ == "__main__":
    loader = PlantDiseaseDataLoader()
    
    # Get available classes
    classes = loader.get_available_classes()
    print("Available classes:")
    for cls in classes:
        print(f"- {cls}")
    
    # Example: Load samples from specific classes
    selected_classes = [
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy"
    ]
    
    print("\nLoading sample images for selected classes...")
    images, labels = loader.get_training_batch(selected_classes, batch_size=8)
    print(f"Loaded {len(images)} images with shape: {images.shape}")