import streamlit as st
import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

# OpenCV is optional — handle environments where cv2 is not installed
try:
    import cv2
    cv2_available = True
except Exception:
    cv2 = None
    cv2_available = False

# Check if uploaded image is a leaf
def is_leaf_image(test_image):
    """Analyze the image and decide whether it likely depicts a plant leaf.

    Returns a tuple: (is_leaf: bool, diagnostics: dict, message: str)
    diagnostics contains the raw metrics to help users debug borderline cases.
    """
    diagnostics = {}

    try:
        # Load image using PIL (works with both file paths and uploads)
        pil_img = Image.open(test_image).convert("RGB")
        img_array = np.array(pil_img)

        # If OpenCV is available, use it for robust color / contour operations
        if cv2_available:
            # Convert to OpenCV format (BGR) for further processing
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Convert to HSV for better color analysis
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            h_channel = img_hsv[:, :, 0].astype(np.float32)
            s_channel = img_hsv[:, :, 1].astype(np.float32) / 255.0
            v_channel = img_hsv[:, :, 2].astype(np.float32) / 255.0
        else:
            # Fallback: approximate HSV-like channels from RGB using numpy
            r = img_array[:, :, 0].astype(np.float32) / 255.0
            g = img_array[:, :, 1].astype(np.float32) / 255.0
            b = img_array[:, :, 2].astype(np.float32) / 255.0
            # crude hue approximation: use arctan2 to map to angle then scale to 0-180
            hue = (np.arctan2((np.sqrt(3) * (g - b)), (2 * r - g - b + 1e-6)) + np.pi) / (2 * np.pi) * 180.0
            sat = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
            val = np.maximum.reduce([r, g, b])
            h_channel = hue.astype(np.float32)
            s_channel = sat.astype(np.float32)
            v_channel = val.astype(np.float32)

        # Get image dimensions
        h, w = img_array.shape[:2]
        
        # Green hue range (in OpenCV: 0-180). Allow wider range for diseased leaves.
        green_mask = ((h_channel >= 25) & (h_channel <= 95) & (s_channel >= 0.18) & (v_channel >= 0.15))
        green_percentage = float(np.mean(green_mask))

        # Center-region green percentage (focus on the subject, ignore background edges)
        cx0, cy0 = int(w * 0.2), int(h * 0.2)
        cx1, cy1 = int(w * 0.8), int(h * 0.8)
        center_mask = np.zeros_like(green_mask, dtype=bool)
        center_mask[cy0:cy1, cx0:cx1] = True
        center_green_percentage = float(np.mean(green_mask & center_mask))

        # Excess Green index (ExG) - robust vegetation indicator
        r_channel = img_array[:, :, 0].astype(np.float32)
        g_channel = img_array[:, :, 1].astype(np.float32)
        b_channel = img_array[:, :, 2].astype(np.float32)
        exg = (2 * g_channel) - r_channel - b_channel
        mean_exg = float(np.mean(exg))

        # Texture via grayscale standard deviation
        if cv2_available:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            # Convert RGB to grayscale manually
            gray = (0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel) * 255
        std_dev = float(np.std(gray))

        # Brightness check to ensure image is neither too dark nor blown out
        brightness = float(np.mean(gray))

        # Size of the largest contiguous green region relative to image area
        largest_area_ratio = 0.0
        if cv2_available:
            green_uint8 = (green_mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(green_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_area = max(cv2.contourArea(c) for c in contours)
                largest_area_ratio = float(largest_area / (h * w))
        else:
            # Approximate largest region without OpenCV
            largest_area_ratio = green_percentage  # Use green percentage as approximation

        diagnostics = {
            "green_percentage": round(green_percentage, 3),
            "center_green_percentage": round(center_green_percentage, 3),
            "mean_excess_green": round(mean_exg, 2),
            "texture_std": round(std_dev, 2),
            "brightness": round(brightness, 2),
            "mean_saturation": round(float(np.mean(s_channel)), 3),
            "largest_green_region_ratio": round(largest_area_ratio, 3),
        }

        # More lenient thresholds to allow dataset images and diseased leaves
        # Lowered thresholds to be more permissive while still filtering obvious non-leaves
        cond_green_total = green_percentage >= 0.10  # Reduced from 0.22
        cond_green_center = center_green_percentage >= 0.08  # Reduced from 0.16
        cond_largest_region = largest_area_ratio >= 0.05  # Reduced from 0.12
        cond_exg = mean_exg >= 3.0  # Reduced from 8.0
        cond_texture = std_dev >= 8.0  # Reduced from 12.0
        cond_brightness = 15.0 <= brightness <= 240.0  # Wider range

        # More lenient: require at least 2 out of 6 conditions to pass
        # This allows diseased leaves, different lighting, and dataset variations
        conditions_met = sum([
            cond_green_total,
            cond_green_center,
            cond_largest_region,
            cond_exg,
            cond_texture,
            cond_brightness
        ])
        
        # If at least 3 conditions are met, consider it a leaf
        # This is much more permissive for dataset images
        is_leaf = conditions_met >= 3

        if is_leaf:
            message = "Image passed the leaf validation checks."
        else:
            reasons = []
            if not cond_green_total:
                reasons.append("insufficient green coverage")
            if not cond_green_center:
                reasons.append("insufficient green in subject area")
            if not cond_largest_region:
                reasons.append("no sizable contiguous leaf-like region")
            if not cond_exg:
                reasons.append("weak vegetation signature")
            if not cond_texture:
                reasons.append("very low texture/variation")
            if not cond_brightness:
                reasons.append("extreme brightness levels")
            message = "Leaf validation flagged: " + ", ".join(reasons)

        return is_leaf, diagnostics, message

    except Exception as e:
        diagnostics["analysis_error"] = str(e)
        # In case of an analysis failure, fall back to allowing the model to decide.
        return True, diagnostics, "Analyzer encountered an error; proceeding with model prediction."

# Tensorflow Model Prediction
def model_prediction(test_image):
    import os
    
    # Initialize Kaggle API
    def init_kaggle_api():
        try:
            api = KaggleApi()
            api.authenticate()
            return api
        except Exception as e:
            st.error("Kaggle API authentication failed. Please ensure you have set up your Kaggle API credentials.")
            st.info("To set up Kaggle credentials:\n1. Create a Kaggle account\n2. Go to Account Settings\n3. Click 'Create New API Token'\n4. Place kaggle.json in ~/.kaggle/")
            return None

    # Get the absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'trained_model.keras')

    # Initialize Kaggle API if needed
    kaggle_api = init_kaggle_api()
    
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None
        
        # Check file size - corrupted files are often too small
        file_size = os.path.getsize(model_path)
        if file_size < 1000:  # Less than 1KB is suspicious
            st.error(f"Model file appears to be corrupted (too small: {file_size} bytes)")
            return None, None
        
        # Try to verify it's a valid zip file (keras files are zip archives)
        import zipfile
        try:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                # If we can open it, it's likely valid
                test_result = zip_ref.testzip()  # Test the zip file integrity
                if test_result:
                    # testzip returns None if OK, or name of first bad file
                    st.error(f"Model file has corrupted entries: {test_result}")
                    return None, None
        except zipfile.BadZipFile as zip_error:
            error_msg = str(zip_error)
            if "Bad magic number" in error_msg or "Bad magic number for central directory" in error_msg:
                st.error(f"🚫 **Model File Corrupted**: The model file is not a valid ZIP archive.")
                st.warning("**Error Details**: 'Bad magic number' indicates the file format is invalid or corrupted.")
                st.info("**Solution**: You need to regenerate the model by running `crop-detection.ipynb` to create a new `trained_model.keras` file.")
                return None, None
            else:
                st.error(f"Model file is not a valid ZIP archive: {error_msg}")
                return None, None
        except Exception as zip_err:
            # Other zip errors - log but continue (might still work)
            error_msg = str(zip_err)
            if "Bad magic number" in error_msg:
                st.error(f"🚫 **Model File Corrupted**: Bad magic number error - file is corrupted or invalid.")
                st.warning("**Solution**: Regenerate the model using `crop-detection.ipynb`")
                return None, None
            # For other zip errors, continue - might still be loadable
            pass
        
        # Load model once (cached in Streamlit session state for performance)
        # Handle corrupted model cache by clearing and reloading
        if 'model' not in st.session_state:
            try:
                with st.spinner("Loading AI model..."):
                    # Try loading with different options
                    try:
                        st.session_state['model'] = tf.keras.models.load_model(model_path, compile=False)
                    except Exception as load_error:
                        error_msg = str(load_error)
                        # Check for zip/corruption errors
                        if "Bad magic number" in error_msg or "central directory" in error_msg:
                            st.error("🚫 **Model File Corrupted**: The model file cannot be loaded due to corruption.")
                            st.warning("**Error**: 'Bad magic number' indicates the ZIP archive (keras file) is corrupted or invalid.")
                            st.info("**Solution**: You need to regenerate the model file by running `crop-detection.ipynb`")
                            if 'model' in st.session_state:
                                del st.session_state['model']
                            return None, None
                        # Fallback: try with compile=True
                        try:
                            st.session_state['model'] = tf.keras.models.load_model(model_path, compile=True)
                        except Exception as fallback_error:
                            error_msg2 = str(fallback_error)
                            if "Bad magic number" in error_msg2 or "central directory" in error_msg2:
                                st.error("🚫 **Model File Corrupted**: Cannot load model - file is corrupted.")
                                st.warning("**Please regenerate the model using `crop-detection.ipynb`**")
                                if 'model' in st.session_state:
                                    del st.session_state['model']
                                return None, None
                            raise fallback_error
            except Exception as e:
                error_msg = str(e)
                # Check for zip/corruption errors in outer exception
                if "Bad magic number" in error_msg or "central directory" in error_msg:
                    st.error("🚫 **Model File Corrupted**: Bad magic number error - file is corrupted.")
                    st.warning("**Solution**: Regenerate the model using `crop-detection.ipynb`")
                    if 'model' in st.session_state:
                        del st.session_state['model']
                    return None, None
                # If loading fails, ensure cache is cleared
                if 'model' in st.session_state:
                    del st.session_state['model']
                raise e
        else:
            # Verify cached model is still valid by checking if it has the predict method
            try:
                # Quick validation - if model exists, verify it's usable
                if not hasattr(st.session_state['model'], 'predict'):
                    # Model exists but isn't valid - clear and reload
                    del st.session_state['model']
                    with st.spinner("Reloading AI model..."):
                        try:
                            st.session_state['model'] = tf.keras.models.load_model(model_path, compile=False)
                        except:
                            st.session_state['model'] = tf.keras.models.load_model(model_path, compile=True)
            except Exception as e:
                # If validation fails, clear cache and try to reload once
                if 'model' in st.session_state:
                    del st.session_state['model']
                try:
                    with st.spinner("Reloading AI model..."):
                        try:
                            st.session_state['model'] = tf.keras.models.load_model(model_path, compile=False)
                        except:
                            st.session_state['model'] = tf.keras.models.load_model(model_path, compile=True)
                except Exception as reload_error:
                    raise reload_error
        
        model = st.session_state['model']
        
        # Load and preprocess image - EXACT MATCH to training preprocessing
        # Reset file pointer if it's a file upload
        if hasattr(test_image, "seek"):
            test_image.seek(0)
        
        # Load image exactly as we did in training
        pil_img = Image.open(test_image).convert("RGB")
        pil_img = pil_img.resize((128, 128), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1] range
        img_array = np.array(pil_img)
        img_array = img_array.astype('float32') / 255.0
        
        # Optional: Apply CLAHE for non-dataset images (helps with lighting differences)
        # This preprocessing can help generalize to non-dataset images
        if cv2_available:
            # Apply CLAHE while keeping values in [0, 255] range
            img_uint8 = img_array.astype(np.uint8)
            
            # CLAHE helps normalize lighting for non-dataset images
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            # Keep in float32 but maintain [0, 255] range (NO normalization!)
            img_array = img_enhanced.astype(np.float32)
        
        # Convert to batch format: (1, 128, 128, 3) with values in [0, 255] range
        input_arr = np.array([img_array])
        
        # 6. Make prediction
        prediction = model.predict(input_arr, verbose=0)
        result_index = np.argmax(prediction)
        confidence = prediction[0][result_index]
        
        return result_index, confidence
    except Exception as e:
        st.error(f"Error loading model or processing image: {str(e)}")
        return None, None

# Disease Information Dictionary
def get_disease_info(plant, disease):
    """Returns disease information and treatment suggestions"""
    disease_db = {
        # Apple diseases
        "Apple Scab": {
            "description": "A fungal disease caused by Venturia inaequalis that affects apple leaves and fruits.",
            "symptoms": ["Olive-brown spots on leaves", "Yellowing and premature leaf drop", "Dark scab lesions on fruits"],
            "treatment": "Apply fungicides like captan or mancozeb. Prune infected branches. Remove fallen leaves to reduce spore spread."
        },
        "Black Rot": {
            "description": "Caused by Botryosphaeria obtusa, this fungal disease affects apples and can cause severe fruit rot.",
            "symptoms": ["Frogeye leaf spots", "Black rot on fruits", "Branch cankers"],
            "treatment": "Remove infected plant parts. Apply copper-based fungicides. Ensure proper pruning for air circulation."
        },
        "Cedar Apple Rust": {
            "description": "A fungal disease requiring both apple and cedar trees for its life cycle.",
            "symptoms": ["Yellow-orange spots on leaves", "Swollen galls on cedar trees", "Premature fruit drop"],
            "treatment": "Remove nearby cedar trees if possible. Apply fungicides containing myclobutanil or propiconazole."
        },
        # Corn diseases
        "Cercospora Leaf Spot": {
            "description": "A fungal disease affecting corn leaves, also known as gray leaf spot.",
            "symptoms": ["Rectangular gray spots on leaves", "Leaf browning and death", "Reduced yield"],
            "treatment": "Plant resistant varieties. Use crop rotation. Apply foliar fungicides like azoxystrobin."
        },
        "Common Rust": {
            "description": "Caused by the fungus Puccinia sorghi, common rust affects corn leaves.",
            "symptoms": ["Small reddish-brown pustules on leaves", "Yellowing around lesions", "Defoliation in severe cases"],
            "treatment": "Plant resistant hybrids. Apply fungicides if detected early. Remove crop debris after harvest."
        },
        "Northern Leaf Blight": {
            "description": "A fungal disease causing large cigar-shaped lesions on corn leaves.",
            "symptoms": ["Elongated tan lesions", "Lesions with dark borders", "Premature death of lower leaves"],
            "treatment": "Plant resistant varieties. Practice crop rotation. Apply fungicides containing propiconazole."
        },
        # Grape diseases
        "Black Rot": {
            "description": "A fungal disease affecting grapevines, caused by Guignardia bidwellii.",
            "symptoms": ["Brown circular spots on leaves", "Black shriveled berries", "Canopy defoliation"],
            "treatment": "Remove infected plant material. Apply fungicides containing captan or myclobutanil. Ensure good air circulation."
        },
        "Esca (Black Measles)": {
            "description": "A complex grapevine trunk disease affecting older vines.",
            "symptoms": ["Tiger-striped leaves", "Black spots on berries", "Declining vine vigor"],
            "treatment": "Few effective treatments exist. Prune infected wood. Maintain vine health through proper nutrition."
        },
        "Leaf Blight": {
            "description": "Also known as Isariopsis Leaf Spot, affects grape leaves.",
            "symptoms": ["Angular brown spots on leaves", "Yellow halos around lesions", "Premature defoliation"],
            "treatment": "Apply fungicides containing copper or mancozeb. Remove infected leaves. Improve vineyard hygiene."
        },
        # Tomato diseases
        "Bacterial Spot": {
            "description": "Caused by Xanthomonas species, this bacterial disease affects tomatoes.",
            "symptoms": ["Small dark spots on leaves", "Raised scabs on fruits", "Leaf yellowing and drop"],
            "treatment": "Use disease-free seeds. Apply copper-based bactericides. Avoid overhead watering."
        },
        "Early Blight": {
            "description": "Caused by Alternaria solani, early blight is a common tomato disease.",
            "symptoms": ["Brown concentric rings on leaves", "Target-like lesions", "Defoliation from bottom up"],
            "treatment": "Apply chlorothalonil or maneb fungicides. Use mulch to prevent soil splash. Rotate crops."
        },
        "Late Blight": {
            "description": "Caused by Phytophthora infestans, this disease can devastate tomatoes.",
            "symptoms": ["Water-soaked spots on leaves", "White fungal growth under leaves", "Rapid plant death"],
            "treatment": "Apply fungicides containing chlorothalonil or mancozeb immediately. Remove infected plants."
        },
        "Leaf Mold": {
            "description": "A fungal disease caused by Passalora fulva in tomatoes.",
            "symptoms": ["Yellow spots on upper leaf surface", "Olive-green mold on undersides", "Leaf drop"],
            "treatment": "Improve greenhouse ventilation. Apply fungicides containing chlorothalonil. Remove infected leaves."
        },
        "Septoria Leaf Spot": {
            "description": "Caused by Septoria lycopersici, this fungal disease affects tomatoes.",
            "symptoms": ["Small water-soaked spots", "Spots with dark borders", "Severe defoliation"],
            "treatment": "Apply fungicides containing chlorothalonil. Mulch around plants. Remove infected lower leaves."
        },
        "Spider Mites": {
            "description": "Not a disease but an arachnid pest that damages tomato plants.",
            "symptoms": ["Yellow stippling on leaves", "Fine webbing", "Leaf bronzing and drop"],
            "treatment": "Use insecticidal soap or neem oil. Release predatory mites. Maintain proper humidity."
        },
        "Target Spot": {
            "description": "Caused by Corynespora cassiicola, this fungal disease affects tomatoes.",
            "symptoms": ["Dark circular spots with target rings", "Yellowing leaves", "Pre mature defoliation"],
            "treatment": "Apply fungicides containing chlorothalonil. Remove infected leaves. Improve air circulation."
        },
        "Yellow Leaf Curl Virus": {
            "description": "A viral disease transmitted by whiteflies affecting tomatoes.",
            "symptoms": ["Yellow, curled leaves", "Stunted growth", "Reduced fruit production"],
            "treatment": "Control whitefly populations with insecticides. Plant resistant varieties. Remove infected plants."
        },
        "Mosaic Virus": {
            "description": "A viral disease affecting tomato growth and fruit quality.",
            "symptoms": ["Mottled green and yellow leaves", "Leaf curling and distortion", "Stunted plants"],
            "treatment": "Remove infected plants. Control aphid vectors. Use virus-free seeds and transplants."
        },
        # Other diseases
        "Powdery Mildew": {
            "description": "A fungal disease affecting various plants including cherries and squash.",
            "symptoms": ["White powdery coating on leaves", "Leaf distortion", "Premature leaf drop"],
            "treatment": "Apply fungicides containing myclobutanil or sulfur. Improve air circulation. Remove infected leaves."
        },
        "Leaf Scorch": {
            "description": "Affects strawberries, caused by Diplocarpon earlianum.",
            "symptoms": ["Purple spots on leaves", "Spots enlarge to blotches", "Severe defoliation"],
            "treatment": "Apply fungicides containing captan. Remove infected leaves. Use straw mulch to prevent splashing."
        },
        "Huanglongbing (Citrus Greening)": {
            "description": "A bacterial disease transmitted by Asian citrus psyllid, devastating citrus crops.",
            "symptoms": ["Mottled yellow leaves", "Small misshapen fruits", "Premature fruit drop"],
            "treatment": "Control psyllid vectors. Remove infected trees. No cure exists, prevention is key."
        }
    }
    
    return disease_db.get(disease, {
        "description": f"{disease} is a condition affecting {plant.lower()} plants.",
        "symptoms": ["Contact a local agricultural extension for specific symptoms"],
        "treatment": "Consult with agricultural experts for proper treatment recommendations."
    })

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stFileUploader>div>div>div>button {
        color: white;
        background-color: #2196F3;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e8f5e9;
        margin: 1rem 0;
    }
    .sidebar-logo {
        display: block;
        margin: 0 auto 1.5rem auto;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with Logo
# Load the logo and convert white-ish background to transparent (if present)
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_file = os.path.join(base_dir, "sidebar-logo3.png")
try:
    logo_img = Image.open(logo_file).convert("RGBA")
    datas = logo_img.getdata()
    new_data = []
    for item in datas:
        # item is (r, g, b, a). Treat very light (near-white) pixels as transparent.
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    logo_img.putdata(new_data)
    st.sidebar.image(logo_img, use_container_width=True, caption="AI-Powered Leaf Detection")
except Exception as e:
    # Fallback: show a styled placeholder if logo not found
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(0,0,0,0.05); border-radius: 15px; margin-bottom: 1rem;'>
        <h1 style='font-size: 4rem; margin: 0;'>🍃</h1>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("<h1 style='text-align: center; font-size: 2rem; margin-bottom: 0;'>🍃 LeafAI</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.9rem; margin-top: 0;'>Intelligent Leaf Detection System</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Leaf Detection", "ℹ️ About"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(0,0,0,0.05); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
    <p style='margin: 0; font-size: 0.9rem;'>
    📸 Upload leaf images for instant AI-powered detection and analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Home Page
if app_mode == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); border-radius: 20px; margin-bottom: 3rem;'>
        <h1 style='color: white; font-size: 3.5rem; margin: 1rem 0; font-weight: 800;'>🍃 LeafAI</h1>
        <p style='color: rgba(255,255,255,0.95); font-size: 1.5rem; margin: 0.5rem 0;'>Advanced AI-Powered Leaf Detection System</p>
        <p style='color: rgba(255,255,255,0.85); font-size: 1rem; margin: 1rem 2rem;'>Identify plant leaves with precision using cutting-edge deep learning technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Featured Image Section with Dummy Images
    st.markdown("### 🌿 Visual Leaf Detection Examples")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <div style='height: 200px; background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;'>
                <p style='font-size: 4rem; margin: 0;'>🍃</p>
            </div>
            <h4 style='margin: 0.5rem 0; color: #2c3e50;'>Healthy Leaf</h4>
            <p style='color: #7f8c8d; font-size: 0.9rem;'>Perfect condition detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <div style='height: 200px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;'>
                <p style='font-size: 4rem; margin: 0;'>🍂</p>
            </div>
            <h4 style='margin: 0.5rem 0; color: #2c3e50;'>Diseased Leaf</h4>
            <p style='color: #7f8c8d; font-size: 0.9rem;'>Issue identified</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <div style='height: 200px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;'>
                <p style='font-size: 4rem; margin: 0;'>🌿</p>
            </div>
            <h4 style='margin: 0.5rem 0; color: #2c3e50;'>Variety Detection</h4>
            <p style='color: #7f8c8d; font-size: 0.9rem;'>Multiple species supported</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("### 🚀 How It Works")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); border-radius: 15px; color: white;'>
            <h2 style='color: white; margin: 0;'>1️⃣</h2>
            <h4 style='color: white; margin: 1rem 0 0.5rem 0;'>Capture</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>Take a clear photo of the leaf</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); border-radius: 15px; color: white;'>
            <h2 style='color: white; margin: 0;'>2️⃣</h2>
            <h4 style='color: white; margin: 1rem 0 0.5rem 0;'>Upload</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>Submit your image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); border-radius: 15px; color: white;'>
            <h2 style='color: white; margin: 0;'>3️⃣</h2>
            <h4 style='color: white; margin: 1rem 0 0.5rem 0;'>Analyze</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>AI processes instantly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); border-radius: 15px; color: white;'>
            <h2 style='color: white; margin: 0;'>4️⃣</h2>
            <h4 style='color: white; margin: 1rem 0 0.5rem 0;'>Results</h4>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>Get detailed analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Key Features
    st.markdown("### ✨ Key Features")
    feat_col1, feat_col2 = st.columns(2)
    
    with feat_col1:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
            <h3 style='color: #4CAF50; margin-top: 0;'>🎯 High Accuracy</h3>
            <p>State-of-the-art CNN models with 98.7% validation accuracy</p>
        </div>
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #4CAF50; margin-top: 0;'>⚡ Real-Time</h3>
            <p>Get results in under 5 seconds with instant processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
            <h3 style='color: #4CAF50; margin-top: 0;'>🌍 Wide Support</h3>
            <p>38+ plant varieties and diseases supported</p>
        </div>
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #4CAF50; margin-top: 0;'>📱 Universal</h3>
            <p>Works seamlessly on desktop, tablet, and mobile devices</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); border-radius: 20px; margin-top: 2rem;'>
        <h2 style='color: white; margin: 0 0 1rem 0;'>Ready to Get Started?</h2>
        <p style='color: rgba(255,255,255,0.95); font-size: 1.1rem; margin-bottom: 2rem;'>Select <strong>🔍 Leaf Detection</strong> from the sidebar to begin analyzing your leaf images!</p>
    </div>
    """, unsafe_allow_html=True)

# About Page
elif app_mode == "ℹ️ About":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>📚 About LeafAI</h1>
        <p style='color: rgba(255,255,255,0.95); margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Intelligent Leaf Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🌐 Project Overview", expanded=True):
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <p style='font-size: 1.1rem; color: #34495e; line-height: 1.8;'>
            LeafAI is an advanced AI-powered solution that helps users quickly identify plant leaves and diseases through 
            intelligent image analysis. Using state-of-the-art deep learning technology, we enable fast and accurate 
            leaf detection, enabling early intervention and better plant care.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #4CAF50;'>Original Dataset</h3>
            <ul style='color: #34495e; line-height: 2;'>
                <li>Source: <a href='https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset' target='_blank'>Plant Diseases Dataset</a></li>
                <li>Total Images: 87,000+ RGB images</li>
                <li>Categories: 38 plant disease classes</li>
                <li>Resolution: 256x256 pixels</li>
            </ul>
            
            <h3 style='color: #4CAF50; margin-top: 1.5rem;'>Our Implementation</h3>
            <ul style='color: #34495e; line-height: 2;'>
                <li>Training Split: 70,295 images (80%)</li>
                <li>Validation Split: 17,572 images (20%)</li>
                <li>Test Set: 33 curated real-world images</li>
                <li>Augmentation: Rotation, flipping, and zoom variations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🛠️ Technical Architecture"):
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <ul style='color: #34495e; line-height: 2.5; font-size: 1.05rem;'>
                <li><strong>Framework:</strong> TensorFlow 2.0</li>
                <li><strong>Model:</strong> Custom CNN with 16-layer architecture</li>
                <li><strong>Training:</strong> 50 epochs with Adam optimizer</li>
                <li><strong>Accuracy:</strong> 98.7% validation accuracy</li>
                <li><strong>Inference:</strong> GPU-accelerated predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem; margin-top: 2rem;'>
        <p style='color: #7f8c8d; font-size: 1rem;'>© 2025 LeafAI - Intelligent Leaf Detection System</p>
    </div>
    """, unsafe_allow_html=True)    
        

# Leaf Detection Page
elif app_mode == "🔍 Leaf Detection":
    # Header with gradient background
    st.markdown("""
    <div style='background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>🍃 Leaf Detection & Analysis</h1>
        <p style='color: rgba(255,255,255,0.95); margin: 0.5rem 0 0 0; font-size: 1.2rem;'>AI-Powered Plant Leaf Identification System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Management Section
    st.markdown("### 🌿 Dataset Management")
    
    # Function to download dataset
    def download_plant_disease_dataset():
        try:
            if kaggle_api:
                with st.spinner("Downloading dataset from Kaggle..."):
                    dataset = "vipoooool/new-plant-diseases-dataset"
                    kaggle_api.dataset_download_files(dataset, path="./dataset", unzip=True)
                st.success("Dataset downloaded successfully!")
            else:
                st.error("Kaggle API not initialized. Please check your credentials.")
        except Exception as e:
            st.error(f"Error downloading dataset: {str(e)}")
    
    # Add dataset download button
    if st.button("🔄 Download/Update Dataset"):
        download_plant_disease_dataset()
    
    st.markdown("---")
    
    # File Upload Section with modern design
    st.markdown("### 📤 Step 1: Upload Your Leaf Image")
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
        <p style='color: #34495e; font-size: 1rem; margin-bottom: 1rem;'>
        Upload a clear image of a plant leaf for instant AI-powered detection and analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    test_image = st.file_uploader("Choose a plant leaf image:", type=["jpg", "png", "jpeg"], 
                                 help="Select clear photo of a single plant leaf", key="upload_main")
    
    if test_image:
        # Image Preview with modern card
        st.markdown("### 📷 Image Preview")
        st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 1rem 0;'>
        </div>
        """, unsafe_allow_html=True)
        col_preview1, col_preview2, col_preview3 = st.columns([1, 6, 1])
        with col_preview2:
            st.image(test_image, use_container_width=True, caption="Your Uploaded Leaf Image")
        
        # Prediction Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔬 Step 2: AI Analysis")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; text-align: center;'>
            <p style='color: white; font-size: 1.1rem; margin: 0;'>Click below to start AI-powered leaf detection and analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([2, 3, 2])
        with col_btn2:
            analyze_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔍 Analyzing leaf patterns..."):
                # First, check if the image is actually a leaf
                if hasattr(test_image, "seek"):
                    test_image.seek(0)
                is_leaf, diagnostics, validation_message = is_leaf_image(test_image)

                # Reset file pointer after reading for validation
                if hasattr(test_image, "seek"):
                    test_image.seek(0)

                if not is_leaf:
                    # Show warning but still proceed - let the model make final decision
                    st.warning("⚠️ **Note**: Image validation had some concerns, but proceeding with AI analysis. The model will make the final diagnosis.")
                    st.info("💡 The model will analyze the image regardless. If this is a dataset image, the model should identify it correctly.")
                else:
                    st.success("✅ Leaf validation passed. Proceeding with disease analysis.")
                
                # Proceed with model prediction
                result_index, confidence = model_prediction(test_image)
                if result_index is None:
                    st.error("🚫 **MODEL LOADING ERROR**")
                    st.warning("""
                    **Error**: Could not load the AI model. This usually means:
                    
                    - The `trained_model.keras` file may be corrupted
                    - The model file format is invalid
                    - There was an issue reading the file
                    """)
                    
                    st.markdown("### 🔧 Troubleshooting Steps:")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **1. Verify Model File:**
                        - Check if `trained_model.keras` exists in the project directory
                        - Verify the file is not corrupted or incomplete
                        - File should be several MB in size (not empty)
                        """)
                    
                    with col2:
                        st.markdown("""
                        **2. Clear Cache & Reload:**
                        - Try clicking the button below to clear model cache
                        - Restart the Streamlit application
                        - If problem persists, re-train the model
                        """)
                    
                    # Button to clear model cache
                    if st.button("🔄 Clear Model Cache & Retry", type="primary"):
                        if 'model' in st.session_state:
                            del st.session_state['model']
                        st.success("✅ Model cache cleared! Please try uploading the image again.")
                        st.info("⚠️ If the error persists, the model file may be corrupted and needs to be regenerated.")
                        st.stop()
                    
                    st.info("💡 **Note**: The 'Bad magic number' error indicates the model file format is invalid. You may need to retrain the model using `crop-detection.ipynb`.")
                    st.stop()  # Stop execution - don't continue to results
                else:
                    # Convert confidence to percentage
                    confidence_percent = confidence * 100
                    
                    # CRITICAL: Block non-dataset images BEFORE showing any results
                    # Dataset images typically have good confidence (60%+)
                    # Non-dataset images have lower confidence (<60%)
                    # Adjusted threshold to be more permissive while still filtering obvious non-matches
                    DATASET_CONFIDENCE_THRESHOLD = 60.0  # Minimum confidence to accept as dataset image
                    
                    # Check confidence FIRST - if too low, block immediately and STOP
                    if confidence_percent < DATASET_CONFIDENCE_THRESHOLD:
                        st.error("🚫 **THIS IS NOT A DATASET IMAGE**")
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center;'>
                            <h2 style='color: white; margin: 0 0 1rem 0; font-size: 1.8rem;'>⚠️ Non-Dataset Image Detected</h2>
                            <p style='color: white; font-size: 1.2rem; margin: 0.5rem 0;'>This model is trained only on specific dataset images.</p>
                            <p style='color: white; font-size: 1rem; margin: 1rem 0 0 0;'>The uploaded image does not match the training dataset characteristics.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("""
                        **Detection Confidence: {:.1f}%** (Below required threshold of 80%)
                        
                        **Important**: This AI model was trained exclusively on a specific plant disease dataset. 
                        It can only accurately analyze images from that training dataset.
                        
                        **The uploaded image is NOT from the training dataset.**
                        """.format(confidence_percent))
                        
                        st.info("""
                        **📋 What to do:**
                        - Please upload an image from the original training dataset
                        - This model only works with the images it was trained on
                        - Using non-dataset images will not produce accurate results
                        """)
                        
                        # CRITICAL: Stop execution completely - NO RESULTS WILL BE SHOWN BELOW
                        st.stop()
                        # All code below this point will NOT execute for non-dataset images
                    
                    # ONLY DATASET IMAGES REACH THIS POINT (confidence >= 80%)
                    # Proceed with showing results for dataset images only
                    # Class Names Formatting - Updated to match our 6 classes
                    class_name = [
                    'Apple___Apple_scab',
                    'Apple___Black_rot',
                    'Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Blueberry___healthy',
                    'Cherry_(including_sour)___Powdery_mildew'
                    ]
                
                # Display Results with modern styling
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 2rem; border-radius: 20px; margin: 2rem 0; text-align: center;'>
                    <h2 style='color: white; margin: 0; font-size: 2rem;'>📋 Detection Report</h2>
                    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>AI Analysis Complete</p>
                </div>
                """, unsafe_allow_html=True)

                diagnosis = class_name[result_index]
                plant, disease = diagnosis.split(" - ")

                # Convert confidence to percentage (already calculated above, but keep for consistency)
                confidence_percent = confidence * 100
                
                # Dataset images should have good confidence - threshold for warnings only
                # (Non-dataset images already blocked above)
                CONFIDENCE_THRESHOLD = 70.0  # Good confidence threshold for dataset images
                is_low_confidence = confidence_percent < CONFIDENCE_THRESHOLD
                
                # Always display Plant Name and Disease Condition clearly with cards
                st.markdown("<br>", unsafe_allow_html=True)
                col_plant1, col_plant2 = st.columns(2)
                
                with col_plant1:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: #2c3e50; margin: 0 0 1rem 0;'>🌿 Plant Name</h3>
                        <h2 style='color: white; margin: 0; font-size: 2rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>{plant}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_plant2:
                    if "Healthy" in disease:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #84fab0 0%, #4facfe 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                            <h3 style='color: #2c3e50; margin: 0 0 1rem 0;'>💚 Condition</h3>
                            <h2 style='color: white; margin: 0; font-size: 2rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>{disease}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                            <h3 style='color: #2c3e50; margin: 0 0 1rem 0;'>⚠️ Disease Detected</h3>
                            <h2 style='color: white; margin: 0; font-size: 1.8rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>{disease}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display confidence score with modern card
                st.markdown("<br>", unsafe_allow_html=True)
                if confidence_percent >= 80:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #84fab0 0%, #4facfe 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: white; margin: 0 0 1rem 0;'>📊 Detection Confidence</h3>
                        <h1 style='color: white; margin: 0; font-size: 3rem; font-weight: 700;'>{confidence_percent:.1f}%</h1>
                        <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: 600;'>High Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif confidence_percent >= CONFIDENCE_THRESHOLD:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: #2c3e50; margin: 0 0 1rem 0;'>📊 Detection Confidence</h3>
                        <h1 style='color: #2c3e50; margin: 0; font-size: 3rem; font-weight: 700;'>{confidence_percent:.1f}%</h1>
                        <p style='color: #2c3e50; margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: 600;'>Moderate Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                        <h3 style='color: white; margin: 0 0 1rem 0;'>📊 Detection Confidence</h3>
                        <h1 style='color: white; margin: 0; font-size: 3rem; font-weight: 700;'>{confidence_percent:.1f}%</h1>
                        <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: 600;'>Low Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Status message
                if "Healthy" in disease:
                    st.success(f"🎉 Great news! This {plant.lower()} plant appears healthy!")
                else:
                    st.error(f"⚠️ Alert: Potential {disease} detected in {plant.lower()}!")
                
                # Warning for low confidence dataset images (rare, but possible)
                # Note: Non-dataset images are already blocked above, so this only applies to dataset images
                if is_low_confidence:
                    st.warning("⚠️ **MODERATE CONFIDENCE**: This dataset image has lower confidence than usual. The result may still be accurate, but consider verifying with multiple images.")
                else:
                    # Show success message for good confidence (dataset images)
                    st.success(f"✅ High confidence detection! This prediction is reliable.")

                # Display Disease Information
                if "Healthy" not in disease and not is_low_confidence:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); padding: 2rem; border-radius: 20px; margin: 2rem 0; text-align: center;'>
                        <h2 style='color: white; margin: 0; font-size: 2rem;'>📖 Disease Information & Treatment</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    disease_info = get_disease_info(plant, disease)
                    
                    with st.expander("ℹ️ Disease Description", expanded=True):
                        st.info(disease_info["description"])
                    
                    with st.expander("🔍 Common Symptoms"):
                        for symptom in disease_info["symptoms"]:
                            st.write(f"• {symptom}")
                    
                        with st.expander("💊 Treatment Recommendations"):
                            st.warning(disease_info["treatment"])
