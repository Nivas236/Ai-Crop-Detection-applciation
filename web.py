import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2

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

        # Convert to OpenCV format (BGR) for further processing
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Convert to HSV for better color analysis
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        h_channel = img_hsv[:, :, 0].astype(np.float32)
        s_channel = img_hsv[:, :, 1].astype(np.float32) / 255.0
        v_channel = img_hsv[:, :, 2].astype(np.float32) / 255.0

        # Green hue range (in OpenCV: 0-180). Allow wider range for diseased leaves.
        green_mask = ((h_channel >= 25) & (h_channel <= 95) & (s_channel >= 0.18) & (v_channel >= 0.15))
        green_percentage = float(np.mean(green_mask))

        # Center-region green percentage (focus on the subject, ignore background edges)
        h, w = img_hsv.shape[:2]
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
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))

        # Brightness check to ensure image is neither too dark nor blown out
        brightness = float(np.mean(gray))

        # Size of the largest contiguous green region relative to image area
        green_uint8 = (green_mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(green_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area_ratio = 0.0
        if contours:
            largest_area = max(cv2.contourArea(c) for c in contours)
            largest_area_ratio = float(largest_area / (h * w))

        diagnostics = {
            "green_percentage": round(green_percentage, 3),
            "center_green_percentage": round(center_green_percentage, 3),
            "mean_excess_green": round(mean_exg, 2),
            "texture_std": round(std_dev, 2),
            "brightness": round(brightness, 2),
            "mean_saturation": round(float(np.mean(s_channel)), 3),
            "largest_green_region_ratio": round(largest_area_ratio, 3),
        }

        # Thresholds: require meaningful, central, contiguous green to avoid background-only greenery
        # Relaxed to better accept diseased/yellowed leaves while blocking objects
        cond_green_total = green_percentage >= 0.18
        cond_green_center = center_green_percentage >= 0.10
        cond_largest_region = largest_area_ratio >= 0.08
        cond_exg = mean_exg >= 5.0
        cond_texture = std_dev >= 11.0
        cond_brightness = 25.0 <= brightness <= 230.0

        # Require center green and a contiguous region, plus one other vegetation signal
        vegetation_signals = [cond_exg, cond_texture, cond_green_total]
        is_leaf = cond_green_center and cond_largest_region and any(vegetation_signals) and cond_brightness

        # Fallback acceptance for severely diseased/dried leaves: allow low-green
        # images if there is strong texture and a sizable contiguous region
        if (not is_leaf) and cond_largest_region and cond_texture and cond_brightness and (center_green_percentage >= 0.06 or mean_exg >= 3.0):
            is_leaf = True

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
    
    # Get the absolute path to the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'trained_model.keras')
    
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None
            
        model = tf.keras.models.load_model(model_path)
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch
        prediction = model.predict(input_arr)
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
    st.sidebar.image(logo_img, use_container_width=True, caption="AI-Powered Crop Protection")
except Exception as e:
    # Fallback: show the raw file path (Streamlit will still attempt to render it)
    st.sidebar.image(logo_file, use_container_width=True, caption="AI-Powered Crop Protection")
    st.sidebar.write(f"Logo processing failed: {e}")

st.sidebar.title("AgriGuard AI")
app_mode = st.sidebar.radio("Navigate", ["Home", "About", "Crop Disease Recognition"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Upload plant leaf images for quick disease diagnosis")

# Home Page
if app_mode == "Home":
    st.header("🌿 Smart Crop Disease Recognition System")
    st.markdown("---")
    
    # Center the image
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        image_path = "homeIMG.jpg"
        st.image(image_path, use_container_width=True, caption="Healthy Crops, Better Harvest")
    
    st.markdown("""
    ### Welcome to Agricultural AI Guardian!
    **Our mission**: Empower farmers with instant plant disease detection using advanced AI technology. 
    Upload a leaf image and get instant diagnosis to protect your crops effectively.

    🚀 **How It Works**
    1. **Capture** - Take a clear photo of the suspect plant leaf
    2. **Upload** - Visit **Crop Disease Recognition** page to submit your image
    3. **Analyze** - Our AI processes the image using deep learning
    4. **Results** - Get instant diagnosis and management tips

    ✨ **Key Benefits**
    - 🎯 95% Accuracy: State-of-the-art convolutional neural networks
    - ⚡ Real-time Results: Diagnosis in under 5 seconds
    - 🌍 38+ Plant Varieties Supported: From apples to tomatoes
    - 📱 Mobile-friendly: Works seamlessly on all devices

    ### Getting Started
    👉 Select **Crop Disease Recognition** from the sidebar to begin your analysis!
    """)

# About Page
elif app_mode == "About":
    st.header("📚 About This Project")
    st.markdown("---")
    
    with st.expander("🌐 Project Overview", expanded=True):
        st.markdown("""
        This AI-powered solution helps farmers quickly identify plant diseases through leaf image analysis, 
        enabling early intervention and reducing crop losses.
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        #### Original Dataset
        - Source: [Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
        - Total Images: 87,000+ RGB images
        - Categories: 38 plant disease classes
        - Resolution: 256x256 pixels
        
        #### Our Implementation
        - Training Split: 70,295 images (80%)
        - Validation Split: 17,572 images (20%)
        - Test Set: 33 curated real-world images
        - Augmentation: Rotation, flipping, and zoom variations
        """)
    
    with st.expander("🛠️ Technical Architecture"):
        st.markdown("""
        - **Framework**: TensorFlow 2.0
        - **Model**: Custom CNN with 16-layer architecture
        - **Training**: 50 epochs with Adam optimizer
        - **Accuracy**: 98.7% validation accuracy
        - **Inference**: GPU-accelerated predictions
        """)
    st.write("© 2025 AI Crop Detection Application")    
        

# Prediction Page
elif app_mode == "Crop Disease Recognition":
    st.header(" Crop Disease Analysis 🔍")
    st.markdown("---")
    
    # File Upload Section
    st.subheader("📤 Step 1: Upload Leaf Image")
    test_image = st.file_uploader("Choose a plant leaf image:", type=["jpg", "png", "jpeg"], 
                                 help="Select clear photo of a single plant leaf")
    
    if test_image:
        # Image Preview
        st.subheader("Image Preview 📷")
        with st.expander("Click to view uploaded image", expanded=True):
            st.image(test_image, use_container_width=True, caption="Uploaded Leaf Image")
        
        # Prediction Section
        st.subheader("Step 2: Disease Diagnosis")
        if st.button(" Start Analysis 🚀", type="primary"):
            with st.spinner("🔍 Analyzing leaf patterns..."):
                # First, check if the image is actually a leaf
                if hasattr(test_image, "seek"):
                    test_image.seek(0)
                is_leaf, diagnostics, validation_message = is_leaf_image(test_image)

                # Reset file pointer after reading for validation
                if hasattr(test_image, "seek"):
                    test_image.seek(0)

                if not is_leaf:
                    st.error("🚫 This image does not appear to be a plant leaf.")
                    st.info("**Recommendations:**\n- Capture the leaf in natural lighting\n- Ensure the leaf dominates the frame\n- Avoid heavy shadows or background clutter\n- Retake the photo from a clearer angle")
                    st.stop()
                else:
                    st.success("Leaf validation passed. Proceeding with disease analysis.")
                
                # If it passes leaf check, proceed with prediction
                result_index, confidence = model_prediction(test_image)
                if result_index is None:
                    st.error("Could not process the image. Please ensure the model file is present and try again.")
                    st.info("If the problem persists, please contact support.")
                else:
                    # Class Names Formatting
                    class_name = [
                    'Apple - Apple Scab',
                    'Apple - Black Rot',
                    'Apple - Cedar Apple Rust',
                    'Apple - Healthy',
                    'Blueberry - Healthy',
                    'Cherry - Powdery Mildew',
                    'Cherry - Healthy',
                    'Corn - Cercospora Leaf Spot',
                    'Corn - Common Rust',
                    'Corn - Northern Leaf Blight',
                    'Corn - Healthy',
                    'Grape - Black Rot',
                    'Grape - Esca (Black Measles)',
                    'Grape - Leaf Blight',
                    'Grape - Healthy',
                    'Orange - Huanglongbing (Citrus Greening)',
                    'Peach - Bacterial Spot',
                    'Peach - Healthy',
                    'Bell Pepper - Bacterial Spot',
                    'Bell Pepper - Healthy',
                    'Potato - Early Blight',
                    'Potato - Late Blight',
                    'Potato - Healthy',
                    'Raspberry - Healthy',
                    'Soybean - Healthy',
                    'Squash - Powdery Mildew',
                    'Strawberry - Leaf Scorch',
                    'Strawberry - Healthy',
                    'Tomato - Bacterial Spot',
                    'Tomato - Early Blight',
                    'Tomato - Late Blight',
                    'Tomato - Leaf Mold',
                    'Tomato - Septoria Leaf Spot',
                    'Tomato - Spider Mites',
                    'Tomato - Target Spot',
                    'Tomato - Yellow Leaf Curl Virus',
                    'Tomato - Mosaic Virus',
                    'Tomato - Healthy'
                ]
                
                # Display Results
                st.markdown("---")
                st.subheader("📋 Diagnosis Report")
                
                diagnosis = class_name[result_index]
                plant, disease = diagnosis.split(" - ")
                
                # Convert confidence to percentage
                confidence_percent = confidence * 100
                
                # Check for low confidence (unknown/out-of-dataset detection)
                CONFIDENCE_THRESHOLD = 50.0
                is_low_confidence = confidence_percent < CONFIDENCE_THRESHOLD
                
                # Display confidence score
                if confidence_percent >= 80:
                    st.metric("Detection Confidence", f"{confidence_percent:.1f}%", delta="High", delta_color="normal")
                elif confidence_percent >= CONFIDENCE_THRESHOLD:
                    st.metric("Detection Confidence", f"{confidence_percent:.1f}%", delta="Moderate", delta_color="off")
                else:
                    st.metric("Detection Confidence", f"{confidence_percent:.1f}%", delta="Low", delta_color="inverse")
                
                # Warning for low confidence
                if is_low_confidence:
                    st.warning("⚠️ **UNKNOWN LEAF DETECTED**: The uploaded leaf image may not match any plant in our database. This could mean:\n\n"
                              "- The plant species is not supported yet\n"
                              "- The image quality may be poor or unclear\n"
                              "- The leaf may be from an unusual angle or damaged\n\n"
                              "**Recommendation**: Please consult with a local agricultural expert for accurate identification.")
                
                if "Healthy" in disease:
                    st.success(f"🎉 Great news! This {plant.lower()} plant appears healthy!")
                else:
                    st.error(f"⚠️ Alert: Potential {disease} detected in {plant.lower()}!")
                
                # Result Card
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color:#2e7d32;"> Plant: {plant}</h3>
                    <h3 style="color:#d32f2f;">Condition: {disease}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display Disease Information
                if "Healthy" not in disease and not is_low_confidence:
                    st.markdown("---")
                    st.subheader("📖 Disease Information & Treatment")
                    disease_info = get_disease_info(plant, disease)
                    
                    with st.expander("ℹ️ Disease Description", expanded=True):
                        st.info(disease_info["description"])
                    
                    with st.expander("🔍 Common Symptoms"):
                        for symptom in disease_info["symptoms"]:
                            st.write(f"• {symptom}")
                    
                    with st.expander("💊 Treatment Recommendations"):
                        st.warning(disease_info["treatment"])
