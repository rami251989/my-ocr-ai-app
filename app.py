# app.py
import os
import io
import json
import base64
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image
import pdfplumber

# Google Vision OCR
from google.cloud import vision

# Gemini
from google import generativeai as genai


# ======================================================
# إعداد واجهة Streamlit
# ======================================================
st.set_page_config(page_title="AI PDF Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 AI PDF Analyzer – Google Vision OCR + Gemini Smart Tagging")
st.caption("استخراج النص من PDF/الصور عبر Google Vision، ثم تلخيص وتقسيم النص إلى أقسام قابلة للتحليل عبر Gemini.")


# ======================================================
# إعداد Google Vision (من secrets Base64)
# ======================================================
@st.cache_resource
def setup_google_vision_client() -> Optional[vision.ImageAnnotatorClient]:
    try:
        key_b64 = st.secrets["GOOGLE_VISION_KEY_B64"]
    except KeyError:
        st.error("❌ لم يتم العثور على GOOGLE_VISION_KEY_B64 في Secrets.")
        return None
    try:
        key_bytes = base64.b64decode(key_b64)
        with open("google_vision.json", "wb") as f:
            f.write(key_bytes)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_vision.json"
        return vision.ImageAnnotatorClient()
    except Exception as e:
        st.error(f"❌ خطأ في تهيئة Google Vision: {e}")
        return None


# ======================================================
# وظائف OCR: PDF -> Images -> Vision
# ======================================================
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    """تحويل كل صفحة PDF إلى صورة PIL بالدقة المطلوبة."""
    images: List[Image.Image] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pil = page.to_image(resolution=dpi).original
            images.append(pil.convert("RGB"))
    return images


def extract_text_from_image(client: vision.ImageAnnotatorClient, image: Image.Image) -> str:
    """استخراج النص من صورة واحدة باستخدام document_text_detection."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    content = buf.getvalue()
    gimg = vision.Image(content=content)

    resp = client.document_text_detection(image=gimg)
    if resp.error.message:
        raise RuntimeError(resp.error.message)

    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    if resp.text_annotations:
        return resp.text_annotations[0].description
    return ""


def extract_text_any(client: vision.ImageAnnotatorClient, uploaded_file, dpi: int = 200) -> str:
    """دعم PDF أو صورة: يرجع النص الكامل."""
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pages = pdf_bytes_to_images(data, dpi=dpi)
        parts = []
        for img in pages:
            parts.append(extract_text_from_image(client, img).strip())
        return "\n\n--- صفحة جديدة ---\n\n".join(parts).strip()
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return extract_text_from_image(client, img).strip()


# ======================================================
# إعداد Gemini + اختبار الاتصال وجلب الموديلات
# ======================================================
@st.cache_resource
def setup_gemini_and_list_models() -> Tuple[Optional[str], List[str]]:
    """يضبط مفتاح Gemini ويعيد (api_key, قائمة الموديلات المتاحة)."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, []
    try:
        genai.configure(api_key=api_key)
        # جلب الموديلات التي تدعم generateContent
        models = []
        try:
            for m in genai.list_models():
                if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
                    models.append(m.name)
        except Exception:
            # بعض البيئات قد لا تسمح بسرد الموديلات؛ سنسمح بالمتابعة بأسماء شائعة
            models = []
        return api_key, models
    except Exception:
        return None, []


# ======================================================
# تحليل النص عبر Gemini: ملخص + أقسام (Tags)
# ======================================================
def analyze_with_gemini(text: str, model_name: str = "gemini-1.5-flash") -> dict:
    """
    يرسل النص إلى Gemini لإنتاج JSON:
    {
      "summary": "...",
      "sections": [
        {"title": "...", "description": "...", "content": "..."}
      ]
    }
    يعتمد على response_mime_type=application/json
    وينفّذ Fallback تلقائي على موديلات بديلة لو حصل خطأ.
    """
    prompt = f"""
أنت مساعد ذكي متخصص في تحليل النصوص المستخلصة من المستندات.

المطلوب:
1) لخص النص بشكل شامل ودقيق.
2) قسّم النص إلى أقسام منطقية (tags/sections) حسب المحتوى.
3) لكل قسم أعد:
   - title: اسم القسم
   - description: وصف قصير يشرح محتواه
   - content: النص الأصلي الخاص بهذا القسم كما هو

أرجع JSON فقط بالشكل التالي:

{{
  "summary": "ملخص النص الكامل",
  "sections": [
    {{
      "title": "اسم القسم",
      "description": "وصف قصير",
      "content": "النص الخاص بالقسم"
    }}
  ]
}}

النص لتحليله:
----------------
{text}
    """.strip()

    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
        "max_output_tokens": 8192
    }

    favorites = [model_name, "gemini-1.5-flash", "gemini-1.5-pro"]
    # إزالة التكرارات مع الحفاظ على الترتيب
    seen = set()
    models_to_try = [m for m in favorites if not (m in seen or seen.add(m))]

    last_err = None
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(model_name=m)
            resp = model.generate_content(prompt, generation_config=generation_config)
            raw = getattr(resp, "text", "") or ""
            raw = raw.strip()
            return json.loads(raw)
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"فشل التحليل عبر Gemini. جرّبنا: {models_to_try}. آخر خطأ: {last_err}")


# ======================================================
# واجهة المستخدم
# ======================================================
with st.sidebar:
    st.header("الإعدادات")
    dpi = st.slider("دقة تحويل PDF → صور (DPI)", 120, 300, 200, step=20)
    st.caption("رفع DPI يحسن دقة OCR (أبطأ قليلًا).")

st.subheader("📂 1) ارفع ملف PDF أو صورة")
uploaded = st.file_uploader("اختر الملف", type=["pdf", "png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
if uploaded and col1.button("🚀 تشغيل OCR"):
    client = setup_google_vision_client()
    if not client:
        st.stop()
    with st.spinner("🧠 جاري استخراج النص عبر Google Vision..."):
        try:
            uploaded.seek(0)
            text = extract_text_any(client, uploaded, dpi=dpi)
            text = (text or "").replace("\x0c", "\n").strip()
            st.session_state["ocr_text"] = text
            st.success("✅ تم استخراج النص.")
        except Exception as e:
            st.error(f"❌ فشل OCR: {e}")

if col2.button("🧹 تنظيف النص"):
    t = st.session_state.get("ocr_text", "")
    if t:
        st.session_state["ocr_text"] = t.strip()
        st.success("✅ تم التنظيف.")
    else:
        st.warning("لا يوجد نص بعد.")

ocr_text = st.session_state.get("ocr_text", "")
st.text_area("📜 النص المستخرج:", ocr_text, height=260)

if ocr_text:
    st.download_button("⬇️ تنزيل النص", data=ocr_text.encode("utf-8"),
                       file_name="ocr_text.txt", mime="text/plain")


# ======================================================
# اتصال Gemini + اختيار الموديل
# ======================================================
st.subheader("🔌 2) التحقق من اتصال Gemini")
api_key, available_models = setup_gemini_and_list_models()

if not api_key:
    st.error("❌ GEMINI_API_KEY غير موجود أو غير صالح في Secrets.")
    st.stop()

if available_models:
    st.success("✅ مفتاح Gemini صالح.")
    # أعرض أول 5 موديلات كمعلومة
    st.caption("بعض الموديلات المتاحة لحسابك:")
    st.code(", ".join(available_models[:5]) + (" ..." if len(available_models) > 5 else ""))
else:
    st.warning("⚠️ لم يتم جلب قائمة الموديلات (قد لا يكون ذلك متاحًا في بيئتك). سنستخدم أسماء شائعة.")

selected_model = st.selectbox(
    "اختر موديل للتحليل",
    options=(available_models or ["gemini-1.5-flash", "gemini-1.5-pro"]),
    index=0
)


# ======================================================
# تشغيل التحليل عبر Gemini
# ======================================================
st.subheader("🤖 3) تحليل النص: ملخص + أقسام")
if st.button("تشغيل تحليل AI"):
    if not ocr_text.strip():
        st.warning("⚠️ لم يتم استخراج أي نص بعد.")
        st.stop()
    with st.spinner("🔍 جاري تحليل النص عبر Gemini..."):
        try:
            result = analyze_with_gemini(ocr_text, model_name=selected_model)
            st.success("✅ تم التحليل بنجاح!")
            st.markdown("### 📋 النتيجة (JSON)")
            st.json(result, expanded=False)

            st.download_button(
                "⬇️ تنزيل JSON",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name="analysis_result.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"❌ فشل التحليل: {e}")


# ======================================================
# ملاحظات ختامية
# ======================================================
st.markdown("---")
st.markdown("""
### 💡 ملاحظات:
- تأكد من صحة المفاتيح داخل **Secrets** بالأسماء: `GEMINI_API_KEY` و `GOOGLE_VISION_KEY_B64`.
- لو ظهر خطأ NotFound من Gemini، غيّر الموديل إلى `gemini-1.5-flash` أو `gemini-1.5-pro` وتأكد من ترقية الحزمة `google-generativeai`.
- يمكنك لاحقًا إضافة خيارات (Prompt مخصص، حفظ Sections كملف، معالجة Batch).
""")
