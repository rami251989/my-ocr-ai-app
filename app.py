import os
import io
import re
import json
import base64
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from PIL import Image
import pdfplumber

# OCR
from google.cloud import vision

# Gemini
from google import generativeai as genai


# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="اتفاقيات المؤسسة - OCR + AI", page_icon="📄", layout="wide")
st.markdown("""
<style>
/* بطاقات جميلة */
.card {
  background: #ffffff;
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 6px 22px rgba(0,0,0,0.06);
  border: 1px solid rgba(0,0,0,0.04);
}
.metric {
  display:flex;align-items:center;gap:12px;font-weight:600;
}
.metric .label {color:#6b7280;}
.metric .value {font-size:18px;}
.section-title{
  font-weight:800;font-size:20px;margin:8px 0 6px 0;
}
hr{border:none;border-top:1px solid #eee;margin:8px 0 16px;}
/* جدول أنيق */
table.dataframe td, table.dataframe th { padding: 8px 10px !important; }
</style>
""", unsafe_allow_html=True)


# =========================
# إعداد Google Vision من secrets (Base64)
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
    imgs: List[Image.Image] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pil = page.to_image(resolution=dpi).original
            imgs.append(pil.convert("RGB"))
    return imgs

def extract_text_from_image(client: vision.ImageAnnotatorClient, image: Image.Image) -> str:
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
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"):
        pages = pdf_bytes_to_images(data, dpi=dpi)
        parts = [extract_text_from_image(client, img).strip() for img in pages]
        return "\n\n--- صفحة جديدة ---\n\n".join(parts).strip()
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return extract_text_from_image(client, img).strip()


# =========================
# Gemini: إعداد + قائمة موديلات (اختياري)
# =========================
@st.cache_resource
def setup_gemini_and_list_models() -> Tuple[Optional[str], List[str]]:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, []
    try:
        genai.configure(api_key=api_key)
        models = []
        try:
            for m in genai.list_models():
                if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
                    models.append(m.name)
        except Exception:
            models = []
        return api_key, models
    except Exception:
        return None, []


# =========================
# أدوات الأرقام: توحيد دينار/فلس + أرقام عربية
# =========================
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_digits(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.translate(ARABIC_DIGITS)
    s = s.replace(",", "").replace("،", "")
    return s.strip()

def coalesce_price(dinar: Any, fils: Any) -> Optional[float]:
    """دمج دينار + فلس إلى قيمة عشرية: dinar + fils/1000 (بعض الوثائق تستخدم 1000 فلس = دينار)."""
    try:
        dn = float(normalize_digits(dinar)) if str(dinar).strip() != "" else 0.0
    except:
        dn = 0.0
    try:
        fs = float(normalize_digits(fils)) if str(fils).strip() != "" else 0.0
    except:
        fs = 0.0
    # فلس إلى دينار: 1000 فلس = 1 دينار
    return round(dn + fs/1000.0, 3)

def to_float(val: Any) -> Optional[float]:
    try:
        s = normalize_digits(val)
        if s == "": return None
        return float(s)
    except:
        return None

def to_int(val: Any) -> Optional[int]:
    try:
        s = normalize_digits(val)
        if s == "": return None
        return int(float(s))
    except:
        return None


# =========================
# تحليل النص عبر Gemini وفق مخططك
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
  * نسبة_ضريبة_المبيعات (نسبة % كقيمة عشرية مثل 0.16)
- استخراج فقرات نصية:
  * فقرة_الكفالات
  * الشروط_الخاصة
  * الشروط_العامة

تعليمات مهمة:
- إذا ظهر السعر مقسّمًا إلى عمودين (دينار/فلس) قم بدمجه إلى رقم عشري موحد (دينار + فلس/1000).
- استخدم الأرقام العربية/الإنجليزية كلاهما مسموح، لكن أعد القيم الرقمية كأرقام فقط.
- إن تعذر العثور على قيمة، ضع null أو اترك الحقل النصي فارغًا.

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

def analyze_agreement_with_gemini(text: str, model_name: str = "gemini-1.5-flash") -> Dict[str, Any]:
    prompt = AGREEMENT_PROMPT_TEMPLATE.format(doc_text=text)
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
        "max_output_tokens": 8192
    }

    candidates = [model_name, "gemini-1.5-flash", "gemini-1.5-pro"]
    seen = set()
    models_to_try = [m for m in candidates if not (m in seen or seen.add(m))]

    last_err = None
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(model_name=m)
            resp = model.generate_content(prompt, generation_config=generation_config)
            raw = (getattr(resp, "text", "") or "").strip()
            result = json.loads(raw)
            return result
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"فشل التحليل عبر Gemini. جرّبنا {models_to_try}. آخر خطأ: {last_err}")


# =========================
# Post-processing للمواد
# =========================
def postprocess_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    - توحيد الأرقام (تحويل عربية -> إنجليزية).
    - التأكد من تحويل الأسعار إلى float ودمج الدينار/الفلس لو أعادها AI منفصلة بشكل نصي.
    - تنظيف التواريخ والنصوص.
    """
    if not isinstance(data, dict):
        return data

    # نصوص عليا
    for k in ["الفريق_الأول", "الفريق_الثاني", "ملخص_الاتفاقية", "فقرة_الكفالات", "الشروط_الخاصة", "الشروط_العامة"]:
        if k in data and isinstance(data[k], str):
            data[k] = data[k].strip()

    # تواريخ (نتركها كما هي أو None)
    for k in ["تاريخ_البدء", "تاريخ_الانتهاء"]:
        if k in data and isinstance(data[k], str):
            data[k] = data[k].strip() or None

    # المواد
    items = data.get("المواد", [])
    cleaned_items = []
    for it in items if isinstance(items, list) else []:
        name = (it.get("اسم_المادة") or "").strip()

        # محاولة التقاط حالات محتملة: price_dinar, price_fils (إن رجعها الذكاء بمفتاحين)
        # ثم تحويلها إلى الحقول الموحدة المطلوبة
        pbt = it.get("سعر_الشراء_قبل_الضريبة")
        pwt = it.get("سعر_الشراء_مع_الضريبة")

        # إن جاء كقواميس {"دينار":.., "فلس":..} أو نص يحتوي "دينار/فلس"
        def unify_price(val) -> Optional[float]:
            if isinstance(val, dict):
                return coalesce_price(val.get("دينار"), val.get("فلس"))
            if isinstance(val, str):
                s = normalize_digits(val)
                # محاولات: "12 دينار 500 فلس" أو "12+500" الخ
                m = re.search(r"(\d+)\D+(\d+)", s)
                if m:
                    return coalesce_price(m.group(1), m.group(2))
                # رقم جاهز
                f = to_float(s)
                return f
            if isinstance(val, (int, float)):
                return float(val)
            return None

        price_before = unify_price(pbt)
        price_with = unify_price(pwt)

        qty = to_int(it.get("الكمية_المشتراة_بالحبة"))
        total_value = to_float(it.get("القيمة_المشتراة_بالدينار"))
        tax_rate = to_float(it.get("نسبة_ضريبة_المبيعات"))

        cleaned_items.append({
            "اسم_المادة": name,
            "سعر_الشراء_قبل_الضريبة": price_before,
            "سعر_الشراء_مع_الضريبة": price_with,
            "الكمية_المشتراة_بالحبة": qty,
            "القيمة_المشتراة_بالدينار": total_value,
            "نسبة_ضريبة_المبيعات": tax_rate
        })

    data["المواد"] = cleaned_items
    return data


# =========================
# واجهة المستخدم
# =========================
with st.sidebar:
    st.header("الإعدادات")
    dpi = st.slider("دقة تحويل PDF → صور (DPI)", 120, 320, 200, step=20)
    st.caption("رفع الـ DPI يحسن دقة OCR (أبطأ قليلاً).")

st.title("📄 اتفاقيات/عروض المؤسسة الاستهلاكية العسكرية")
st.write("ارفع ملف الاتفاقية (PDF/صورة) لاستخراج النص، ثم حلّله عبر الذكاء الاصطناعي لهيكلة البيانات المطلوبة.")

# رفع وتشغيل OCR
st.subheader("1) رفع الملف وتشغيل OCR")
uploaded = st.file_uploader("اختر الملف", type=["pdf", "png", "jpg", "jpeg"])
c1, c2 = st.columns(2)
if uploaded and c1.button("🚀 تشغيل OCR"):
    client = setup_google_vision_client()
    if not client:
        st.stop()
    with st.spinner("جاري استخراج النص عبر Google Vision..."):
        try:
            uploaded.seek(0)
            text = extract_text_any(client, uploaded, dpi=dpi)
            text = (text or "").replace("\x0c", "\n").strip()
            st.session_state["ocr_text"] = text
            st.success("✅ تم استخراج النص.")
        except Exception as e:
            st.error(f"❌ فشل OCR: {e}")

if c2.button("🧹 تنظيف النص"):
    t = st.session_state.get("ocr_text", "")
    if t:
        st.session_state["ocr_text"] = t.strip()
        st.success("✅ تم التنظيف.")
    else:
        st.warning("لا يوجد نص بعد.")

ocr_text = st.session_state.get("ocr_text", "")
st.text_area("📜 النص المستخرج:", ocr_text, height=220)
if ocr_text:
    st.download_button("⬇️ تنزيل النص", data=ocr_text.encode("utf-8"), file_name="ocr_text.txt", mime="text/plain")

st.subheader("2) الاتصال بـ Gemini")
api_key, available_models = setup_gemini_and_list_models()
if not api_key:
    st.error("❌ GEMINI_API_KEY غير موجود أو غير صالح في Secrets.")
    st.stop()

if available_models:
    st.success("✅ مفتاح Gemini صالح.")
    st.caption("موديلات متاحة (أول 5):")
    st.code(", ".join(available_models[:5]) + (" ..." if len(available_models) > 5 else ""))
else:
    st.warning("⚠️ لم يتم جلب قائمة الموديلات. سنستخدم أسماء شائعة.")

selected_model = st.selectbox(
    "اختر موديل التحليل",
    options=(available_models or ["gemini-1.5-flash", "gemini-1.5-pro"]),
    index=0
)

st.subheader("3) تحليل الاتفاقية واستخراج الحقول المنظمة")
if st.button("🤖 تشغيل التحليل بالـ AI"):
    if not ocr_text.strip():
        st.warning("⚠️ لم يتم استخراج أي نص بعد.")
        st.stop()
    with st.spinner("🔍 جاري التحليل عبر Gemini..."):
        try:
            raw = analyze_agreement_with_gemini(ocr_text, model_name=selected_model)
            result = postprocess_result(raw)
            st.success("✅ تم التحليل وهيكلة البيانات.")

            # ====== عرض جميل ======
            # البطاقات العلوية: الفريقين + التواريخ
            colA, colB = st.columns(2)
            with colA:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">👥 الأطراف</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">الفريق الأول:</span><span class="value">{result.get("الفريق_الأول","—")}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">الفريق الثاني:</span><span class="value">{result.get("الفريق_الثاني","—")}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with colB:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🗓️ المدة</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">تاريخ البدء:</span><span class="value">{result.get("تاريخ_البدء","—")}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric"><span class="label">تاريخ الانتهاء:</span><span class="value">{result.get("تاريخ_الانتهاء","—")}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ملخص
            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🧾 ملخص الاتفاقية</div>', unsafe_allow_html=True)
            st.write(result.get("ملخص_الاتفاقية", "—"))
            st.markdown('</div>', unsafe_allow_html=True)

            # جدول المواد
            import pandas as pd
            items = result.get("المواد", [])
            df = pd.DataFrame(items, columns=[
                "اسم_المادة",
                "سعر_الشراء_قبل_الضريبة",
                "سعر_الشراء_مع_الضريبة",
                "الكمية_المشتراة_بالحبة",
                "القيمة_المشتراة_بالدينار",
                "نسبة_ضريبة_المبيعات"
            ])
            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📦 المواد ضمن الاتفاقية</div>', unsafe_allow_html=True)
            if not df.empty:
                # أعمدة محسوبة/تشكيلية اختيارية
                # مثال: قيمة قبل الضريبة = سعر_الشراء_قبل_الضريبة * الكمية (إن توفرت)
                try:
                    df["قيمة_قبل_الضريبة_(حساب)"] = (df["سعر_الشراء_قبل_الضريبة"].astype(float)) * (df["الكمية_المشتراة_بالحبة"].fillna(0).astype(float))
                except Exception:
                    pass
                try:
                    df["قيمة_مع_الضريبة_(حساب)"] = (df["سعر_الشراء_مع_الضريبة"].astype(float)) * (df["الكمية_المشتراة_بالحبة"].fillna(0).astype(float))
                except Exception:
                    pass

                st.dataframe(df, use_container_width=True, height=380)
                # تنزيل CSV + JSON للمواد فقط
                st.download_button("⬇️ تنزيل المواد (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="items.csv", mime="text/csv")
            else:
                st.info("لا توجد مواد مستخرجة.")
            st.markdown('</div>', unsafe_allow_html=True)

            # فقرات نصية
            colC, colD, colE = st.columns(3)
            with colC:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🛡️ فقرة الكفالات</div>', unsafe_allow_html=True)
                st.write(result.get("فقرة_الكفالات", "—"))
                st.markdown('</div>', unsafe_allow_html=True)
            with colD:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">⚙️ الشروط الخاصة</div>', unsafe_allow_html=True)
                st.write(result.get("الشروط_الخاصة", "—"))
                st.markdown('</div>', unsafe_allow_html=True)
            with colE:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📜 الشروط العامة</div>', unsafe_allow_html=True)
                st.write(result.get("الشروط_العامة", "—"))
                st.markdown('</div>', unsafe_allow_html=True)

            # تنزيل النتيجة كاملة
            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⬇️ تنزيل النتيجة الكاملة</div>', unsafe_allow_html=True)
            st.download_button("تحميل JSON كامل", data=json.dumps(result, ensure_ascii=False, indent=2),
                               file_name="agreement_analysis.json", mime="application/json")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ فشل التحليل: {e}")

st.markdown("---")
st.caption("تم البناء باستخدام Google Vision OCR + Gemini. يمكن توسيع التحليلات لاحقًا (مطابقة أصناف، حساب ضرائب، إلخ).")
