import streamlit as st
import google.generativeai as genai
import json, re, os, io, base64, tempfile
from google.cloud import vision
from PIL import Image
import pdfplumber

# ===========================
# 1️⃣ إعداد المفاتيح
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
# 2️⃣ OCR شامل (صور + PDF)
# ===========================
def _vision_client():
    return vision.ImageAnnotatorClient()

def _ocr_image_bytes(client: vision.ImageAnnotatorClient, img_bytes: bytes) -> str:
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
    """
    يدعم PDF + صور (PNG/JPG). للـ PDF نُحوّل كل صفحة إلى صورة ثم نطبّق OCR.
    """
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
# 3️⃣ Prompt الوسوم
# ===========================
AGREEMENT_PROMPT_TEMPLATE = r"""
أنت مساعد لتحليل اتفاقيات "المؤسسة الاستهلاكية العسكرية".
أعد الرد **بالضبط** بهذه الوسوم، وبدون أي نص خارجها، وبدون تعليقات أو شروحات:

<<<TEAM_A>>>
[اكتب اسم الفريق الأول فقط]
<<<END_TEAM_A>>>

<<<TEAM_B>>>
[اكتب اسم الفريق الثاني فقط]
<<<END_TEAM_B>>>

<<<DATE_START>>>
[تاريخ البدء بصيغة YYYY-MM-DD أو اتركه فارغاً]
<<<END_DATE_START>>>

<<<DATE_END>>>
[تاريخ الانتهاء بصيغة YYYY-MM-DD أو اتركه فارغاً]
<<<END_DATE_END>>>

<<<SUMMARY>>>
[ملخص موجز جداً للاتفاقية]
<<<END_SUMMARY>>>

# المصفوفة التالية فقط بصيغة JSON صحيحة. لا تضف أي نص خارج الأقواس.
# الشروط:
# - "اسم_المادة": نص (دائماً String)
# - باقي الحقول أرقام عشرية بالدينار بعد دمج الدينار+الفلس (إن وجدتا منفصلتين بالنص).
# - "الكمية_المشتراة_بالحبة" عدد صحيح (اكتب رقماً فقط).
# - "نسبة_ضريبة_المبيعات" كقيمة عشرية (مثلاً 0.16 وليس 16%).
# - لا تعليقات، لا فواصل زائدة قبل ] أو }}.
# - إن لم توجد مواد، أعد مصفوفة فارغة [].
<<<ITEMS_JSON_ARRAY>>>
[
  {{
    "اسم_المادة": "مثال",
    "سعر_الشراء_قبل_الضريبة": 0.0,
    "سعر_الشراء_مع_الضريبة": 0.0,
    "الكمية_المشتراة_بالحبة": 0,
    "القيمة_المشتراة_بالدينار": 0.0,
    "نسبة_ضريبة_المبيعات": 0.0
  }}
]
<<<END_ITEMS_JSON_ARRAY>>>

<<<WARRANTIES>>>
[نص فقرة الكفالات إن وجدت]
<<<END_WARRANTIES>>>

<<<SPECIAL_TERMS>>>
[الشروط الخاصة إن وجدت]
<<<END_SPECIAL_TERMS>>>

<<<GENERAL_TERMS>>>
[الشروط العامة إن وجدت]
<<<END_GENERAL_TERMS>>>

تعليمات مهمة:
- وحِّد الأسعار بالدينار فقط (اجمع الدينار + الفلس/1000 إن ظهرت منفصلة).
- التزم بالبنية أعلاه حرفياً.
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
    import json, re
    # إزالة محارف الاتجاه/BOM/Zero-width
    raw = re.sub(r"[\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u200B\u200C\u200D]", "", raw).strip()

    def g(a, b):
        pat = re.compile(re.escape(a) + r"(.*?)" + re.escape(b), re.S)
        m = pat.search(raw)
        return (m.group(1).strip() if m else "")

    items_json = g("<<<ITEMS_JSON_ARRAY>>>", "<<<END_ITEMS_JSON_ARRAY>>>").strip()

    items = []
    if items_json:
        # إزالة code fences إن وُجدت
        items_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", items_json, flags=re.IGNORECASE | re.MULTILINE).strip()
        # تطبيع علامات الاقتباس والفواصل العربية
        items_json = (items_json
                      .replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
                      .replace("،", ",").replace("٫", "."))
        # إزالة الفواصل الزائدة قبل الأقواس
        items_json = re.sub(r",\s*([}\]])", r"\1", items_json)
        # اقتباس المفاتيح غير المقتبسة (عربية/إنجليزية/أرقام/شرطة سفلية)
        items_json = re.sub(r'([{,]\s*)([A-Za-z0-9_ء-ي]+)\s*:', r'\1"\2":', items_json)
        # في بعض الأحيان اسم_المادة يُعاد رقمًا → اقتبسه كسلسلة
        items_json = re.sub(r'("اسم_المادة"\s*:\s*)(-?\d+(?:\.\d+)?)', r'\1"\2"', items_json)

        try:
            parsed = json.loads(items_json)
        except Exception as e:
            # محاولة ثانية بعد تنظيف بسيط
            items_json2 = re.sub(r"\s+\n\s+", "\n", items_json)
            try:
                parsed = json.loads(items_json2)
            except Exception:
                parsed = []

        # تأكد أن النتيجة قائمة من كائنات
        if isinstance(parsed, dict):
            items = [parsed]
        elif isinstance(parsed, list):
            items = [x for x in parsed if isinstance(x, dict)]
        else:
            items = []
    else:
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
# 5️⃣ تحليل بالـ Gemini
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

    for m in tried:
        try:
            raw = run_once(m)
            if debug:
                st.caption(f"📄 Raw from {m}:")
                st.code(raw[:1200] + ("..." if len(raw) > 1200 else ""))
            return parse_tagged_response(raw)
        except Exception:
            continue

    raise RuntimeError("❌ فشل التحليل عبر جميع الموديلات")


# ===========================
# 6️⃣ واجهة Streamlit
# ===========================
st.set_page_config(page_title="تحليل اتفاقيات المؤسسة الاستهلاكية العسكرية", layout="wide")
st.title("📑 نظام تحليل اتفاقيات المؤسسة الاستهلاكية العسكرية")
st.markdown("باستخدام **Google Vision OCR + Gemini AI**")

uploaded = st.file_uploader("📤 ارفع صورة أو ملف PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded and st.button("📄 استخراج النص"):
    try:
        with st.spinner("جارٍ استخراج النص عبر Google Vision..."):
            text = extract_text_any(uploaded)
        st.session_state["ocr_text"] = text
        st.success("✅ تم استخراج النص!")
    except Exception as e:
        st.error(f"❌ فشل استخراج النص: {e}")

# عرض النص
st.text_area("📝 النص المستخرج:", st.session_state.get("ocr_text", ""), height=300)

# إعداد Gemini
if GEMINI_KEY:
    st.success("✅ مفتاح Gemini صالح.")
    models_list = genai.list_models()
    models = [m.name for m in models_list if "generateContent" in m.supported_generation_methods]
    selected_model = st.selectbox("اختر الموديل:", models, index=0)
else:
    st.error("❌ لم يتم العثور على مفتاح Gemini")

debug = st.toggle("🧠 إظهار مخرجات التشخيص (Raw)")

if "ocr_text" in st.session_state and st.button("تحليل الاتفاقية"):
    try:
        result = analyze_agreement_with_gemini(st.session_state["ocr_text"], selected_model, debug)
        st.success("✅ تم التحليل بنجاح")
        st.json(result)
    except Exception as e:
        st.error(f"❌ فشل التحليل: {e}")
