import streamlit as st
import google.generativeai as genai
import base64, tempfile, os, io
from google.cloud import vision
from PIL import Image
import pdfplumber

# ===========================
# 1) المفاتيح والتهيئة
# ===========================
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", None)
VISION_KEY_B64 = st.secrets.get("GOOGLE_VISION_KEY_B64", None)

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

if VISION_KEY_B64:
    key_json = base64.b64decode(VISION_KEY_B64).decode("utf-8")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(key_json.encode("utf-8"))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

# ===========================
# 2) OCR للصور و PDF
# ===========================
def _vision_client():
    return vision.ImageAnnotatorClient()

def _ocr_image_bytes(client, img_bytes: bytes) -> str:
    image = vision.Image(content=img_bytes)
    resp = client.document_text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    if resp.text_annotations:
        return resp.text_annotations[0].description
    return ""

def extract_text_any(uploaded_file, dpi: int = 200) -> str:
    name = (uploaded_file.name or "").lower()
    uploaded_file.seek(0)
    data = uploaded_file.read()

    client = _vision_client()

    if name.endswith(".pdf"):
        pages_text = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                pil = page.to_image(resolution=dpi).original.convert("RGB")
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                pages_text.append(_ocr_image_bytes(client, buf.getvalue()))
        return ("\n\n--- صفحة جديدة ---\n\n".join(t.strip() for t in pages_text)).strip()
    else:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return _ocr_image_bytes(client, buf.getvalue())

# ===========================
# 3) واجهة Streamlit
# ===========================
st.set_page_config(page_title="تحليل حر باستخدام Gemini", layout="wide")
st.title("🤖 منصة تحليل النصوص باستخدام Google Vision + Gemini")

st.markdown("### 🧾 الخطوة 1: رفع الملف (صورة أو PDF)")
uploaded = st.file_uploader("📤 ارفع ملفك هنا", type=["png", "jpg", "jpeg", "pdf"])

if uploaded and st.button("📄 استخراج النص"):
    try:
        with st.spinner("جارٍ استخراج النص..."):
            text = extract_text_any(uploaded)
        st.session_state["ocr_text"] = text
        st.success("✅ تم استخراج النص بنجاح!")
    except Exception as e:
        st.error(f"❌ فشل استخراج النص: {e}")

# عرض النص المستخرج
if "ocr_text" in st.session_state:
    st.markdown("### 📜 النص المستخرج")
    st.text_area("", st.session_state["ocr_text"], height=300)

# ===========================
# 4) إعدادات Gemini
# ===========================
if GEMINI_KEY:
    st.markdown("### ⚙️ إعداد الموديل والتعليمات")

    models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-exp",
    ]
    selected_model = st.selectbox("اختر الموديل:", models, index=0)

    user_prompt = st.text_area(
        "✏️ اكتب تعليماتك هنا:",
        placeholder="مثلاً: استخرج المواد بصيغة جدول Markdown مع الأسعار والكمية...",
        height=150,
    )

    debug = st.toggle("🧠 عرض مخرجات الـ Raw (اختياري)")

    if st.button("🚀 أرسل التعليمات"):
        if not st.session_state.get("ocr_text"):
            st.error("⚠️ لم يتم استخراج أي نص بعد.")
        elif not user_prompt.strip():
            st.warning("⚠️ الرجاء إدخال تعليمات أولاً.")
        else:
            try:
                full_prompt = f"النص:\n\n{st.session_state['ocr_text']}\n\nالتعليمات:\n{user_prompt}"
                model = genai.GenerativeModel(model_name=selected_model)
                with st.spinner("🤖 جارٍ تحليل النص بالتعليمات المحددة..."):
                    resp = model.generate_content(
                        full_prompt,
                        generation_config={"temperature": 0.2, "max_output_tokens": 8192},
                        safety_settings=[
                                {"category": "dangerous", "threshold": "block_none"},
                                {"category": "harassment", "threshold": "block_none"},
                                {"category": "hate_speech", "threshold": "block_none"},
                                {"category": "sexual", "threshold": "block_none"},
                                {"category": "violence", "threshold": "block_none"},
                        ],
                    )

                # استخراج النص فعلياً من الأجزاء
                text_parts = []
                for cand in getattr(resp, "candidates", []) or []:
                    content = getattr(cand, "content", None)
                    if content and getattr(content, "parts", None):
                        for p in content.parts:
                            if getattr(p, "text", None):
                                text_parts.append(p.text)
                final_text = "\n".join(text_parts).strip()

                if debug:
                    st.code(final_text[:1500], language="markdown")

                if final_text:
                    st.markdown("### 🧩 النتيجة بتنسيق Markdown")
                    st.markdown(final_text)
                else:
                    st.warning("⚠️ النموذج لم يُرجع أي نص قابل للعرض.")

            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء التحليل: {e}")
else:
    st.error("❌ لم يتم العثور على مفتاح Gemini API في secrets")
