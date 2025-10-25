import os
import io
import re
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from PIL import Image
import pdfplumber
from google.cloud import vision
from google import generativeai as genai

# =========================
# إعداد الصفحة والتصميم
# =========================
st.set_page_config(page_title="اتفاقيات المؤسسة العسكرية", page_icon="📄", layout="wide")
st.markdown("""
<style>
.card {
  background: #fff;
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.05);
  border: 1px solid rgba(0,0,0,0.05);
  margin-bottom: 10px;
}
.section-title {font-weight:700;font-size:20px;margin:8px 0 6px 0;}
.metric{display:flex;align-items:center;gap:12px;font-weight:600;}
.metric .label{color:#666;}
.metric .value{font-size:17px;}
</style>
""", unsafe_allow_html=True)

# =========================
# Safe JSON Loader
# =========================
def safe_json_loads(text: str):
    if not text:
        raise ValueError("نص فارغ من الموديل")
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            candidate = re.sub(r",\s*([}\]])", r"\\1", s[start:end+1])
            return json.loads(candidate)
    raise ValueError(f"تعذر تحويل الاستجابة إلى JSON: {s[:150]}")

# =========================
# Google Vision Setup
# =========================
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

def pdf_bytes_to_images(pdf_bytes, dpi=200):
    imgs = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            imgs.append(page.to_image(resolution=dpi).original.convert("RGB"))
    return imgs

def extract_text_any(client, uploaded, dpi=200):
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith(".pdf"):
        pages = pdf_bytes_to_images(data, dpi)
        return "\n\n--- صفحة جديدة ---\n\n".join([
            client.document_text_detection(image=vision.Image(content=io.BytesIO(
                (lambda buf: (img.save(buf, format='PNG'), buf.getvalue())[1])(io.BytesIO())
            ).getvalue())).full_text_annotation.text
            for img in pages
        ])
    else:
        buf = io.BytesIO()
        Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="PNG")
        content = buf.getvalue()
        return client.document_text_detection(image=vision.Image(content=content)).full_text_annotation.text

# =========================
# Gemini Setup + تحليل الاتفاقية
# =========================
@st.cache_resource
def setup_gemini():
    api = st.secrets.get("GEMINI_API_KEY", "")
    if not api:
        return None
    genai.configure(api_key=api)
    return api

AGREEMENT_PROMPT = """
أنت مساعد ذكي لتحليل الاتفاقيات للمؤسسة الاستهلاكية العسكرية.
استخرج القيم التالية بصيغة JSON:
{
 "الفريق_الأول": "...",
 "الفريق_الثاني": "...",
 "تاريخ_البدء": "YYYY-MM-DD",
 "تاريخ_الانتهاء": "YYYY-MM-DD",
 "ملخص_الاتفاقية": "...",
 "المواد": [
   {
     "اسم_المادة": "...",
     "سعر_الشراء_قبل_الضريبة": 0.0,
     "سعر_الشراء_مع_الضريبة": 0.0,
     "الكمية_المشتراة_بالحبة": 0,
     "القيمة_المشتراة_بالدينار": 0.0,
     "نسبة_ضريبة_المبيعات": 0.0
   }
 ],
 "فقرة_الكفالات": "...",
 "الشروط_الخاصة": "...",
 "الشروط_العامة": "..."
}
النص:
----------------
{doc_text}
"""

def analyze_agreement_with_gemini(text):
    schema = {
        "type": "object",
        "properties": {
            "الفريق_الأول": {"type": ["string", "null"]},
            "الفريق_الثاني": {"type": ["string", "null"]},
            "تاريخ_البدء": {"type": ["string", "null"]},
            "تاريخ_الانتهاء": {"type": ["string", "null"]},
            "ملخص_الاتفاقية": {"type": ["string", "null"]},
            "المواد": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "اسم_المادة": {"type": ["string", "null"]},
                        "سعر_الشراء_قبل_الضريبة": {"type": ["number", "null"]},
                        "سعر_الشراء_مع_الضريبة": {"type": ["number", "null"]},
                        "الكمية_المشتراة_بالحبة": {"type": ["number", "null"]},
                        "القيمة_المشتراة_بالدينار": {"type": ["number", "null"]},
                        "نسبة_ضريبة_المبيعات": {"type": ["number", "null"]}
                    },
                    "required": ["اسم_المادة"]
                }
            },
            "فقرة_الكفالات": {"type": ["string", "null"]},
            "الشروط_الخاصة": {"type": ["string", "null"]},
            "الشروط_العامة": {"type": ["string", "null"]}
        }
    }
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = AGREEMENT_PROMPT.format(doc_text=text)
    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0.2
        },
    )
    raw = getattr(resp, "text", "") or ""
    return safe_json_loads(raw)

# =========================
# واجهة Streamlit
# =========================
st.title("📑 محلل اتفاقيات المؤسسة العسكرية")
st.write("ارفع ملف PDF أو صورة، وسيتولى الذكاء الاصطناعي استخراج وتحليل الاتفاقية بدقة وهيكلة البيانات.")

with st.sidebar:
    dpi = st.slider("دقة OCR", 120, 300, 200, 20)

uploaded = st.file_uploader("📂 ارفع الاتفاقية", type=["pdf", "png", "jpg", "jpeg"])
if uploaded and st.button("🚀 تشغيل OCR"):
    client = setup_google_vision_client()
    if client:
        with st.spinner("🧠 استخراج النص..."):
            text = extract_text_any(client, uploaded, dpi)
            st.session_state["ocr_text"] = text
            st.success("✅ تم استخراج النص.")
st.text_area("📜 النص المستخرج:", st.session_state.get("ocr_text", ""), height=250)

if st.button("🤖 تحليل الاتفاقية"):
    api = setup_gemini()
    if not api:
        st.error("❌ تأكد من وضع مفتاح GEMINI_API_KEY في secrets.")
    elif not st.session_state.get("ocr_text"):
        st.warning("⚠️ لم يتم استخراج نص بعد.")
    else:
        with st.spinner("🔍 جاري التحليل..."):
            try:
                result = analyze_agreement_with_gemini(st.session_state["ocr_text"])
                st.success("✅ تم تحليل الاتفاقية.")
                st.json(result)
            except Exception as e:
                st.error(f"❌ فشل التحليل: {e}")

st.markdown("---")
st.caption("📘 يستخدم Google Vision + Gemini لتحليل الوثائق الرسمية بدقة.")
