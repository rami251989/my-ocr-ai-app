import streamlit as st
import google.generativeai as genai
import json, re, os
from google.cloud import vision

# ===========================
# 1️⃣ إعداد المفاتيح
# ===========================
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", None)
VISION_KEY_B64 = st.secrets.get("GOOGLE_VISION_KEY_B64", None)

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

if VISION_KEY_B64:
    import base64, tempfile
    key_json = base64.b64decode(VISION_KEY_B64).decode("utf-8")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(key_json.encode("utf-8"))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name


# ===========================
# 2️⃣ OCR Google Vision
# ===========================
def extract_text_with_google_vision(image_file):
    client = vision.ImageAnnotatorClient()
    content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""


# ===========================
# 3️⃣ Prompt الوسوم
# ===========================
AGREEMENT_PROMPT_TEMPLATE = r"""
أنت مساعد لتحليل اتفاقيات "المؤسسة الاستهلاكية العسكرية".
أعد الرد بالضبط بالوسوم التالية (لا تضف أي شرح خارجها):

<<<TEAM_A>>>
[اكتب اسم الفريق الأول فقط]
<<<END_TEAM_A>>>

<<<TEAM_B>>>
[اكتب اسم الفريق الثاني فقط]
<<<END_TEAM_B>>>

<<<DATE_START>>>
[تاريخ البدء بصيغة YYYY-MM-DD أو فارغ]
<<<END_DATE_START>>>

<<<DATE_END>>>
[تاريخ الانتهاء بصيغة YYYY-MM-DD أو فارغ]
<<<END_DATE_END>>>

<<<SUMMARY>>>
[ملخص موجز وواضح للاتفاقية]
<<<END_SUMMARY>>>

<<<ITEMS_JSON_ARRAY>>>
[
  {
    "اسم_المادة": "...",
    "سعر_الشراء_قبل_الضريبة": 0.0,
    "سعر_الشراء_مع_الضريبة": 0.0,
    "الكمية_المشتراة_بالحبة": 0,
    "القيمة_المشتراة_بالدينار": 0.0,
    "نسبة_ضريبة_المبيعات": 0.0
  }
]
<<<END_ITEMS_JSON_ARRAY>>>

<<<WARRANTIES>>>
[نص فقرة الكفالات (إن وُجدت)]
<<<END_WARRANTIES>>>

<<<SPECIAL_TERMS>>>
[الشروط الخاصة (إن وُجدت)]
<<<END_SPECIAL_TERMS>>>

<<<GENERAL_TERMS>>>
[الشروط العامة (إن وُجدت)]
<<<END_GENERAL_TERMS>>>

تعليمات:
- اجمع الدينار + الفلس لتكون القيم بالدينار.
- أعد JSON المصفوفة بشكل صحيح فقط داخل <<<ITEMS_JSON_ARRAY>>>.
النص:
----------------
{doc_text}
"""


# ===========================
# 4️⃣ تحليل الوسوم
# ===========================
def _between(s: str, start_tag: str, end_tag: str) -> str:
    pat = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.S)
    m = pat.search(s)
    return (m.group(1).strip() if m else "")


def parse_tagged_response(raw: str) -> dict:
    raw = re.sub(r"[\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u200B\u200C\u200D]", "", raw).strip()

    def g(a, b): return _between(raw, a, b)

    items_json = g("<<<ITEMS_JSON_ARRAY>>>", "<<<END_ITEMS_JSON_ARRAY>>>").strip()
    try:
        items_json = re.sub(r",\s*\]", "]", items_json)
        items_json = items_json.replace("“", '"').replace("”", '"')
        items = json.loads(items_json) if items_json else []
    except Exception:
        items = []

    return {
        "الفريق_الأول": g("<<<TEAM_A>>>", "<<<END_TEAM_A>>>"),
        "الفريق_الثاني": g("<<<TEAM_B>>>", "<<<END_TEAM_B>>>"),
        "تاريخ_البدء": g("<<<DATE_START>>>", "<<<END_DATE_START>>>"),
        "تاريخ_الانتهاء": g("<<<DATE_END>>>", "<<<END_DATE_END>>>"),
        "ملخص_الاتفاقية": g("<<<SUMMARY>>>", "<<<END_SUMMARY>>>"),
        "المواد": items,
        "فقرة_الكفالات": g("<<<WARRANTIES>>>", "<<<END_WARRANTIES>>>"),
        "الشروط_الخاصة": g("<<<SPECIAL_TERMS>>>", "<<<END_SPECIAL_TERMS>>>"),
        "الشروط_العامة": g("<<<GENERAL_TERMS>>>", "<<<END_GENERAL_TERMS>>>")
    }


# ===========================
# 5️⃣ التحليل باستخدام Gemini
# ===========================
def analyze_agreement_with_gemini(text: str, selected_model: str, debug: bool = False) -> dict:
    prompt = AGREEMENT_PROMPT_TEMPLATE.format(doc_text=text)

    def run_once(model_name: str) -> str:
        model = genai.GenerativeModel(model_name=model_name)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 8192})
        raw = getattr(resp, "text", "") or ""
        if not raw and getattr(resp, "candidates", None):
            parts = [p.text for c in resp.candidates for p in getattr(c.content, "parts", []) if getattr(p, "text", "")]
            raw = "\n".join(parts)
        return raw

    tried = [selected_model]
    if "2.5" in selected_model:
        tried.append(selected_model.replace("2.5", "1.5"))
    tried += ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]

    errors = []
    for m in tried:
        try:
            raw = run_once(m)
            if debug:
                st.caption(f"📄 Raw from {m}:")
                st.code(raw[:1200] + ("..." if len(raw) > 1200 else ""))
            return parse_tagged_response(raw)
        except Exception as e:
            errors.append(f"{m}: {e}")

    raise RuntimeError("\n".join(errors))


# ===========================
# 6️⃣ Fallback بسيط لملء البيانات
# ===========================
def fallback_fill_from_text(result: dict, ocr_text: str) -> dict:
    if not result.get("الفريق_الأول"):
        m = re.search(r"الفريق\s*الأول\s*[:：]\s*(.+)", ocr_text)
        if m: result["الفريق_الأول"] = m.group(1).strip()
    if not result.get("الفريق_الثاني"):
        m = re.search(r"الفريق\s*الثاني\s*[:：]\s*(.+)", ocr_text)
        if m: result["الفريق_الثاني"] = m.group(1).strip()
    return result


# ===========================
# 7️⃣ واجهة Streamlit
# ===========================
st.set_page_config(page_title="تحليل اتفاقيات المؤسسة الاستهلاكية العسكرية", layout="wide")

st.title("📑 نظام تحليل اتفاقيات المؤسسة الاستهلاكية العسكرية")
st.markdown("باستخدام **Google Vision OCR + Gemini AI**")

# ✅ خطوة 1: رفع الصورة
uploaded = st.file_uploader("📤 ارفع صورة الاتفاقية", type=["png", "jpg", "jpeg", "pdf"])

if uploaded:
    if st.button("📄 استخراج النص"):
        with st.spinner("جارٍ تحليل الصورة..."):
            text = extract_text_with_google_vision(uploaded)
            st.session_state["ocr_text"] = text
        st.success("✅ تم استخراج النص بنجاح!")
        st.text_area("النص المستخرج:", text, height=250)

# ✅ خطوة 2: إعداد Gemini
if GEMINI_KEY:
    st.success("✅ مفتاح Gemini صالح.")
    models_list = genai.list_models()
    models = [m.name for m in models_list if "generateContent" in m.supported_generation_methods]
    selected_model = st.selectbox("اختر الموديل:", models, index=0)
else:
    st.error("❌ لم يتم العثور على مفتاح Gemini")

# ✅ خطوة 3: تحليل النص
debug = st.toggle("🧠 إظهار مخرجات التشخيص (Raw)")

if "ocr_text" in st.session_state and st.button("تحليل الاتفاقية"):
    try:
        result = analyze_agreement_with_gemini(st.session_state["ocr_text"], selected_model, debug)
        result = fallback_fill_from_text(result, st.session_state["ocr_text"])
        st.success("✅ تم التحليل بنجاح")
        st.json(result)
    except Exception as e:
        st.error(f"❌ فشل التحليل: {e}")
