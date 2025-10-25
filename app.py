import os
import io
import json
import base64
import streamlit as st
from PIL import Image
import pdfplumber
from google.cloud import vision
from google import generativeai as genai

# ======================================================
# إعداد واجهة Streamlit
# ======================================================
st.set_page_config(page_title="AI PDF Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 AI PDF Analyzer – OCR + Gemini Smart Tagging")
st.caption("تحليل ملفات PDF تلقائيًا عبر Google Vision OCR + Gemini AI")

# ======================================================
# إعداد Google Vision (من secrets)
# ======================================================
@st.cache_resource
def setup_google_vision_client():
    try:
        key_b64 = st.secrets["GOOGLE_VISION_KEY_B64"]
        key_bytes = base64.b64decode(key_b64)
        with open("google_vision.json", "wb") as f:
            f.write(key_bytes)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_vision.json"
        return vision.ImageAnnotatorClient()
    except Exception as e:
        st.error(f"❌ خطأ في تهيئة Google Vision: {e}")
        return None

# ======================================================
# OCR - استخراج النص من PDF أو صورة
# ======================================================
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200):
    """تحويل كل صفحة PDF إلى صورة"""
    images = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            images.append(page.to_image(resolution=dpi).original.convert("RGB"))
    return images

def extract_text_from_image(client, image: Image.Image) -> str:
    """استخراج النص من صورة باستخدام Vision OCR"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    content = buf.getvalue()
    gimg = vision.Image(content=content)
    response = client.document_text_detection(image=gimg)
    if response.error.message:
        raise Exception(response.error.message)
    if response.full_text_annotation.text:
        return response.full_text_annotation.text
    elif response.text_annotations:
        return response.text_annotations[0].description
    return ""

def extract_text_any(client, uploaded_file, dpi: int = 200):
    """دعم PDF أو صورة"""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pages = pdf_bytes_to_images(data, dpi)
        texts = [extract_text_from_image(client, img) for img in pages]
        return "\n\n--- صفحة جديدة ---\n\n".join(texts)
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return extract_text_from_image(client, img)

# ======================================================
# إعداد Gemini
# ======================================================
def setup_gemini():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return api_key
    except Exception as e:
        st.error(f"❌ لم يتم العثور على مفتاح Gemini في secrets: {e}")
        return None

# ======================================================
# تحليل النص عبر Gemini
# ======================================================
def analyze_with_gemini(text: str, model_name="gemini-1.5-pro") -> dict:
    """
    يرسل النص إلى Gemini ليقوم بتلخيصه وتقسيمه إلى أقسام (تاجات).
    """
    prompt = f"""
أنت مساعد ذكي متخصص في تحليل النصوص المستخلصة من المستندات.

المطلوب:
1. لخص النص بشكل شامل ودقيق.
2. قسّم النص إلى أقسام منطقية (tags/sections) حسب المحتوى.
3. لكل قسم أعد:
   - اسم القسم (title)
   - وصف مختصر له (description)
   - النص الكامل للقسم (content)

أرجع النتيجة بصيغة JSON بالهيكل التالي فقط:

{{
  "summary": "ملخص النص الكامل",
  "sections": [
    {{
      "title": "اسم القسم",
      "description": "شرح مختصر للقسم",
      "content": "النص الأصلي الخاص بهذا القسم"
    }}
  ]
}}

النص لتحليله:
----------------
{text}
"""

    model = genai.GenerativeModel(model_name=model_name)
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.3,
    }

    response = model.generate_content(prompt, generation_config=generation_config)
    try:
        result = json.loads(response.text)
        return result
    except Exception:
        st.warning("⚠️ فشل تحويل الاستجابة إلى JSON منظم، سيتم عرض النص الخام.")
        return {"raw_text": response.text}

# ======================================================
# واجهة المستخدم
# ======================================================

st.subheader("📂 1) ارفع ملف PDF أو صورة")
uploaded = st.file_uploader("اختر الملف", type=["pdf", "png", "jpg", "jpeg"])

dpi = st.slider("دقة التحويل من PDF إلى صور (DPI)", 100, 300, 200, step=50)

if uploaded and st.button("🚀 تشغيل OCR واستخراج النص"):
    client = setup_google_vision_client()
    if not client:
        st.stop()
    with st.spinner("🧠 جاري استخراج النص من الملف..."):
        text = extract_text_any(client, uploaded, dpi)
        st.session_state["ocr_text"] = text
        st.success("✅ تم استخراج النص بنجاح!")

if "ocr_text" in st.session_state:
    st.text_area("📜 النص المستخرج:", st.session_state["ocr_text"], height=250)
else:
    st.info("👆 ارفع ملف ثم اضغط على زر التشغيل لاستخراج النص.")

# ======================================================
# تحليل AI
# ======================================================
st.subheader("🤖 2) تحليل النص باستخدام Gemini AI")
if st.button("تشغيل تحليل AI"):
    if "ocr_text" not in st.session_state:
        st.warning("⚠️ لم يتم استخراج أي نص بعد.")
        st.stop()

    api_key = setup_gemini()
    if not api_key:
        st.stop()

    with st.spinner("🔍 جاري تحليل النص عبر Gemini..."):
        result = analyze_with_gemini(st.session_state["ocr_text"])
        st.success("✅ تم التحليل بنجاح!")

        st.markdown("### 📋 النتيجة:")
        st.json(result, expanded=False)

        # تنزيل النتيجة
        st.download_button(
            "⬇️ تحميل النتيجة بصيغة JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="analysis_result.json",
            mime="application/json"
        )

# ======================================================
# ملاحظات ختامية
# ======================================================
st.markdown("---")
st.markdown("""
### 💡 ملاحظات:
- استخدم Google Vision لاستخراج النص بدقة من PDF أو الصور.
- Gemini يقوم بتلخيص وتقسيم النص إلى أقسام قابلة للتحليل لاحقًا.
- يمكنك تعديل النموذج أو مستوى التفصيل من الكود بسهولة.
""")
