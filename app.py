# -*- coding: utf-8 -*-
import os
import io
import re
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from PIL import Image
import pdfplumber

# Google Vision
from google.cloud import vision

# Gemini
from google import generativeai as genai


# =========================
# إعداد الصفحة + تنسيق بسيط
# =========================
st.set_page_config(page_title="اتفاقيات المؤسسة العسكرية - OCR + Gemini", page_icon="📄", layout="wide")
st.markdown("""
<style>
.card {
  background: #fff;
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 6px 22px rgba(0,0,0,0.06);
  border: 1px solid rgba(0,0,0,0.05);
  margin-bottom: 12px;
}
.section-title {font-weight:800;font-size:20px;margin:4px 0 10px;}
.metric{display:flex;align-items:center;gap:10px;font-weight:600;}
.metric .label{color:#6b7280;}
.metric .value{font-size:17px;}
</style>
""", unsafe_allow_html=True)


# =========================
# أدوات JSON آمنة
# =========================
def safe_json_loads(text: str):
    """
    JSON sanitizer قوي:
    - يزيل محارف الاتجاه: LRM/RLM/LRE/RLE/PDF/LRI/RLI/FSI/PDI
    - يزيل BOM و zero-width
    - يطبّع الاقتباسات الذكية إلى عادية
    - يلفّ مفاتيح تبدأ من أول سطر بدون { } بأقواس
    - يزيل الفواصل الزائدة قبل } أو ]
    - يحاول ast.literal_eval كحل أخير
    """
    import re, json, ast

    if not text:
        raise ValueError("نص فارغ من النموذج")

    s = text

    # 1) أزل code fences ```json ... ```
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE)

    # 2) أزل محارف الاتجاه + BOM + zero-width
    bidi_ctrl = r"[\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u200B\u200C\u200D]"
    s = re.sub(bidi_ctrl, "", s)

    # 3) طبّع الاقتباسات الذكية
    s = s.replace("“", '"').replace("”", '"').replace("„", '"').replace("«", '"').replace("»", '"')
    s = s.replace("’", "'").replace("‘", "'")

    s = s.strip()

    # 4) لو بدأ النص بمفتاح بين "" بدون { } لفّه
    if s and not s.startswith("{") and s.lstrip().startswith('"'):
        s = "{\n" + s
        if not s.rstrip().endswith("}"):
            s = s.rstrip().rstrip(",") + "\n}"

    # 5) إزالة الفواصل الزائدة قبل } أو ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 6) محاولة JSON مباشرة
    try:
        return json.loads(s)
    except Exception:
        pass

    # 7) اقتنص أول كائن { ... } متوازن
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 8) كحل أخير: dict بصيغة بايثون (single quotes)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
            return obj[0]
    except Exception:
        pass

    raise ValueError(f"تعذّر تحويل استجابة النموذج إلى JSON. جزء من النص:\n{s[:300]}")



# =========================
# Google Vision: التهيئة
# =========================
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


# =========================
# OCR: PDF -> صور -> Vision
# =========================
def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    images: List[Image.Image] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pil = page.to_image(resolution=dpi).original
            images.append(pil.convert("RGB"))
    return images

def extract_text_from_image(client: vision.ImageAnnotatorClient, pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    gimg = vision.Image(content=buf.getvalue())
    resp = client.document_text_detection(image=gimg)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    if resp.full_text_annotation and resp.full_text_annotation.text:
        return resp.full_text_annotation.text
    if resp.text_annotations:
        return resp.text_annotations[0].description
    return ""

def extract_text_any(client: vision.ImageAnnotatorClient, uploaded_file, dpi: int = 200) -> str:
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pages = pdf_bytes_to_images(data, dpi=dpi)
        parts = [extract_text_from_image(client, p).strip() for p in pages]
        return "\n\n--- صفحة جديدة ---\n\n".join(parts).strip()
    else:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return extract_text_from_image(client, pil).strip()


# =========================
# Gemini: تهيئة + سرد الموديلات
# =========================
@st.cache_resource
def setup_gemini_and_list_models() -> Tuple[Optional[str], List[str]]:
    api = st.secrets.get("GEMINI_API_KEY", "")
    if not api:
        return None, []
    try:
        genai.configure(api_key=api)
        models = []
        try:
            for m in genai.list_models():
                if getattr(m, "supported_generation_methods", None) and \
                   "generateContent" in m.supported_generation_methods:
                    models.append(m.name)
        except Exception:
            models = []
        return api, models
    except Exception:
        return None, []


# =========================
# Prompt + Schema للتحليل
# =========================
AGREEMENT_PROMPT_TEMPLATE = """
أنت مساعد خبير في تحليل الاتفاقيات والعروض الخاصة بـ "المؤسسة الاستهلاكية العسكرية".

المطلوب من النص التالي:
- تحديد "الفريق الأول" و"الفريق الثاني".
- استخراج "تاريخ بدء الاتفاقية" و"تاريخ انتهائها" بصيغة YYYY-MM-DD إن أمكن.
- استخراج "ملخص الاتفاقية" بشكل موجز وواضح.
- استخراج "قائمة المواد" في جدول منظّم، لكل مادة الحقول التالية (أرقام فقط بدون وحدات):
  * اسم_المادة (نص)
  * سعر_الشراء_قبل_الضريبة (قيمة عشرية موحدة بدينار: اجمع "الدينار" + "الفلس/1000" إن ظهرت منفصلة)
  * سعر_الشراء_مع_الضريبة (قيمة عشرية موحدة)
  * الكمية_المشتراة_بالحبة (عدد صحيح)
  * القيمة_المشتراة_بالدينار (قيمة عشرية)
  * نسبة_ضريبة_المبيعات (قيمة عشرية للنسبة مثل 0.16)
- استخراج فقرات نصية:
  * فقرة_الكفالات
  * الشروط_الخاصة
  * الشروط_العامة

أعد **JSON فقط** بالهيكل التالي:

{
  "الفريق_الأول": "...",
  "الفريق_الثاني": "...",
  "تاريخ_البدء": "YYYY-MM-DD or null",
  "تاريخ_الانتهاء": "YYYY-MM-DD or null",
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

AGREEMENT_JSON_SCHEMA: Dict[str, Any] = {
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
                    "الكمية_المشتراة_بالحبة": {"type": ["integer","number","null"]},
                    "القيمة_المشتراة_بالدينار": {"type": ["number", "null"]},
                    "نسبة_ضريبة_المبيعات": {"type": ["number", "null"]}
                },
                "required": ["اسم_المادة"],
                "additionalProperties": True
            }
        },
        "فقرة_الكفالات": {"type": ["string", "null"]},
        "الشروط_الخاصة": {"type": ["string", "null"]},
        "الشروط_العامة": {"type": ["string", "null"]}
    },
    "required": ["الفريق_الأول", "الفريق_الثاني", "المواد"],
    "additionalProperties": False
}


# =========================
# Fallback لأسماء الموديلات
# =========================
def build_model_fallbacks(selected: str) -> list:
    seen, out = set(), []
    def add(m):
        if m and m not in seen:
            seen.add(m); out.append(m)

    add(selected)
    if "2.5" in selected:
        add(selected.replace("2.5", "1.5"))

    add("models/gemini-1.5-flash")
    add("models/gemini-1.5-pro")
    return out


# =========================
# تحليل الاتفاقية عبر Gemini
# =========================
def analyze_agreement_with_gemini(text: str, selected_model: str, debug: bool = False) -> dict:
    prompt = AGREEMENT_PROMPT_TEMPLATE.format(doc_text=text)

    def run_once(model_name: str, use_schema: bool) -> str:
        model = genai.GenerativeModel(model_name=model_name)
        gen_cfg = {
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "max_output_tokens": 8192,
        }
        if use_schema:
            gen_cfg["response_schema"] = AGREEMENT_JSON_SCHEMA
        resp = model.generate_content(prompt, generation_config=gen_cfg)
        raw = getattr(resp, "text", "") or ""
        if not raw and getattr(resp, "candidates", None):
            parts = []
            for c in resp.candidates:
                for p in getattr(c.content, "parts", []):
                    if getattr(p, "text", ""):
                        parts.append(p.text)
            raw = "\n".join(parts)
        if debug:
            st.caption(f"📄 Raw ({model_name}, schema={use_schema}):")
            st.code(raw[:1000] + ("..." if len(raw) > 1000 else ""))
        return raw

    errors = []
    for m in build_model_fallbacks(selected_model):
        for use_schema in (True, False):  # جرّب مع schema ثم بدون
            try:
                raw = run_once(m, use_schema)
                return safe_json_loads(raw)
            except Exception as e:
                errors.append(f"{m} (schema={use_schema}): {e}")
                continue

    raise RuntimeError("فشل التحليل عبر Gemini:\n" + "\n".join(errors[:6]))


# =========================
# واجهة Streamlit
# =========================
st.title("📑 محلّل اتفاقيات المؤسسة الاستهلاكية العسكرية")
st.write("ارفع ملف PDF/صورة → OCR → تحليل بالذكاء الاصطناعي وإخراج منظّم وجميل.")

with st.sidebar:
    st.header("الإعدادات")
    dpi = st.slider("دقة OCR (DPI)", 120, 320, 200, step=20)
    debug = st.checkbox("إظهار مخرجات التشخيص (Raw)", value=False)

# 1) رفع وتشغيل OCR
st.subheader("1) رفع الملف وتشغيل OCR")
uploaded = st.file_uploader("📂 اختر الاتفاقية (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])
b1, b2 = st.columns(2)
if uploaded and b1.button("✨ تشغيل OCR"):
    client = setup_google_vision_client()
    if client:
        with st.spinner("🧠 استخراج النص عبر Google Vision..."):
            try:
                uploaded.seek(0)
                text = extract_text_any(client, uploaded, dpi=dpi)
                text = (text or "").replace("\x0c", "\n").strip()
                st.session_state["ocr_text"] = text
                st.success("✅ تم استخراج النص.")
            except Exception as e:
                st.error(f"❌ فشل OCR: {e}")

if b2.button("🧹 تنظيف النص"):
    t = st.session_state.get("ocr_text", "")
    if t:
        st.session_state["ocr_text"] = t.strip()
        st.success("✅ تم التنظيف.")
    else:
        st.warning("لا يوجد نص بعد.")

ocr_text = st.session_state.get("ocr_text", "")
st.text_area("🧾 النص المستخرج:", ocr_text, height=260)
if ocr_text:
    st.download_button("⬇️ تنزيل النص", data=ocr_text.encode("utf-8"),
                       file_name="ocr_text.txt", mime="text/plain")

# 2) اتصال Gemini + اختيار الموديل
st.subheader("2) واختيار الموديل للاتصال بـ Gemini")
api_key, available_models = setup_gemini_and_list_models()
if not api_key:
    st.error("❌ GEMINI_API_KEY غير موجود في Secrets.")
    st.stop()

if available_models:
    st.success("✅ مفتاح Gemini صالح.")
    st.caption("موديلات متاحة (أول 5):")
    st.code(", ".join(available_models[:5]) + (" ..." if len(available_models) > 5 else ""))
else:
    st.warning("⚠️ تعذر سرد الموديلات. سنستخدم الأسماء الشائعة (models/gemini-1.5-...).")

selected_model = st.selectbox(
    "اختر اسم الموديل",
    options=(available_models or [
        "models/gemini-2.5-pro-preview-03-25",
        "models/gemini-2.5-flash-preview-05-20",
        "models/gemini-2.5-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
    ]),
    index=0
)

# 3) تحليل الاتفاقية
st.subheader("3) تحليل الاتفاقية وإظهار النتائج")
if st.button("🤖 تحليل الاتفاقية"):
    if not ocr_text.strip():
        st.warning("⚠️ لم يتم استخراج نص بعد.")
        st.stop()
    with st.spinner("🔍 جاري التحليل عبر Gemini..."):
        try:
            result = analyze_agreement_with_gemini(ocr_text, selected_model, debug=debug)
            st.success("✅ تم التحليل وهيكلة البيانات.")

            # ===== عرض مرتب =====
            # الأطراف + المدة
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">👥 الأطراف</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">الفريق الأول:</span><span class="value">{result.get("الفريق_الأول","—")}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">الفريق الثاني:</span><span class="value">{result.get("الفريق_الثاني","—")}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🗓️ المدة</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">تاريخ البدء:</span><span class="value">{result.get("تاريخ_البدء","—")}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">تاريخ الانتهاء:</span><span class="value">{result.get("تاريخ_الانتهاء","—")}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ملخص
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🧾 ملخص الاتفاقية</div>', unsafe_allow_html=True)
            st.write(result.get("ملخص_الاتفاقية", "—"))
            st.markdown('</div>', unsafe_allow_html=True)

            # جدول المواد
            import pandas as pd
            items = result.get("المواد", []) or []
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📦 المواد ضمن الاتفاقية</div>', unsafe_allow_html=True)
            if items:
                df = pd.DataFrame(items, columns=[
                    "اسم_المادة",
                    "سعر_الشراء_قبل_الضريبة",
                    "سعر_الشراء_مع_الضريبة",
                    "الكمية_المشتراة_بالحبة",
                    "القيمة_المشتراة_بالدينار",
                    "نسبة_ضريبة_المبيعات"
                ])
                # أعمدة حسابية اختيارية
                try:
                    df["قيمة_قبل_الضريبة_(حساب)"] = df["سعر_الشراء_قبل_الضريبة"].astype(float) * df["الكمية_المشتراة_بالحبة"].fillna(0).astype(float)
                except Exception:
                    pass
                try:
                    df["قيمة_مع_الضريبة_(حساب)"] = df["سعر_الشراء_مع_الضريبة"].astype(float) * df["الكمية_المشتراة_بالحبة"].fillna(0).astype(float)
                except Exception:
                    pass

                st.dataframe(df, use_container_width=True, height=380)
                st.download_button("⬇️ تنزيل المواد (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="items.csv", mime="text/csv")
            else:
                st.info("لا توجد مواد مستخرجة.")
            st.markdown('</div>', unsafe_allow_html=True)

            # فقرات نصية
            c3, c4, c5 = st.columns(3)
            with c3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🛡️ فقرة الكفالات</div>', unsafe_allow_html=True)
                st.write(result.get("فقرة_الكفالات", "—"))
                st.markdown('</div>', unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">⚙️ الشروط الخاصة</div>', unsafe_allow_html=True)
                st.write(result.get("الشروط_الخاصة", "—"))
                st.markdown('</div>', unsafe_allow_html=True)
            with c5:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📜 الشروط العامة</div>', unsafe_allow_html=True)
                st.write(result.get("الشروط_العامة", "—"))
                st.markdown('</div>', unsafe_allow_html=True)

            # تنزيل JSON كامل
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⬇️ تنزيل النتيجة الكاملة</div>', unsafe_allow_html=True)
            st.download_button("تحميل JSON كامل",
                               data=json.dumps(result, ensure_ascii=False, indent=2),
                               file_name="agreement_analysis.json",
                               mime="application/json")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ فشل التحليل: {e}")

st.markdown("---")
st.caption("تحليل الوثائق الرسمية بدقة باستخدام Google Vision + Gemini.")
