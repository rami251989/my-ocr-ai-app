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
   def chunk_text(text: str, max_chars: int = 12000) -> list:
    """
    يقص النص لقطع قصيرة حتى لا يرفضه الموديل بسبب الطول.
    يراعي الفصل على حدود أسطر إن أمكن.
    """
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # حاول القص عند أقرب سطر
        nl = text.rfind("\n", start, end)
        if nl == -1 or nl <= start + int(max_chars*0.5):
            nl = end
        chunks.append(text[start:nl].strip())
        start = nl
    return [c for c in chunks if c]


def merge_results(parts: list) -> dict:
    """
    يدمج قائمة نتائج parse_tagged_response.
    """
    merged = {
        "الفريق_الأول": None,
        "الفريق_الثاني": None,
        "تاريخ_البدء": None,
        "تاريخ_الانتهاء": None,
        "ملخص_الاتفاقية": "",
        "المواد": [],
        "فقرة_الكفالات": "",
        "الشروط_الخاصة": "",
        "الشروط_العامة": ""
    }
    def first_nonempty(cur, new):
        return cur if (cur and str(cur).strip()) else (new if (new and str(new).strip()) else cur)

    for p in parts:
        merged["الفريق_الأول"]    = first_nonempty(merged["الفريق_الأول"],    p.get("الفريق_الأول"))
        merged["الفريق_الثاني"]   = first_nonempty(merged["الفريق_الثاني"],   p.get("الفريق_الثاني"))
        merged["تاريخ_البدء"]     = first_nonempty(merged["تاريخ_البدء"],     p.get("تاريخ_البدء"))
        merged["تاريخ_الانتهاء"]  = first_nonempty(merged["تاريخ_الانتهاء"],  p.get("تاريخ_الانتهاء"))

        if p.get("ملخص_الاتفاقية"):
            if merged["ملخص_الاتفاقية"]:
                merged["ملخص_الاتفاقية"] += "\n• " + p["ملخص_الاتفاقية"].strip()
            else:
                merged["ملخص_الاتفاقية"] = "• " + p["ملخص_الاتفاقية"].strip()

        if p.get("المواد"):
            merged["المواد"].extend(p["المواد"])

        for k in ["فقرة_الكفالات","الشروط_الخاصة","الشروط_العامة"]:
            if p.get(k):
                if merged[k]:
                    merged[k] += "\n" + p[k].strip()
                else:
                    merged[k] = p[k].strip()

    return merged


def _model_fallbacks(selected: str) -> list:
    seen, out = set(), []
    def add(m):
        if m and m not in seen:
            seen.add(m); out.append(m)

    add(selected)
    # إن كان 2.5 جرّب 1.5 من نفس العائلة
    if "2.5" in selected:
        add(selected.replace("2.5", "1.5"))

    # مجموعة موسعة من الأسماء الشائعة
    add("models/gemini-1.5-pro")
    add("models/gemini-1.5-flash")
    add("models/gemini-1.5-pro-001")
    add("models/gemini-1.5-flash-001")
    return out


# ===========================
# أدوات التجزئة والدمج + قائمة الموديلات الاحتياطية
# ===========================
def chunk_text(text: str, max_chars: int = 10000) -> list:
    """
    يقص النص لقطع أقصر حتى لا يرفضه الموديل بسبب الطول.
    يراعي القص عند نهاية سطر إن أمكن.
    """
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # حاول القص عند أقرب سطر، بشرط ألا نرجع كثيراً
        nl = text.rfind("\n", start, end)
        if nl == -1 or nl <= start + int(max_chars * 0.5):
            nl = end
        chunk = text[start:nl].strip()
        if chunk:
            chunks.append(chunk)
        start = nl
    return chunks

def merge_results(parts: list) -> dict:
    """
    يدمج نتائج متعددة من parse_tagged_response في نتيجة واحدة.
    يأخذ أول قيمة غير فارغة للحقول الفردية ويجمع الجداول والنصوص.
    """
    merged = {
        "الفريق_الأول": None,
        "الفريق_الثاني": None,
        "تاريخ_البدء": None,
        "تاريخ_الانتهاء": None,
        "ملخص_الاتفاقية": "",
        "المواد": [],
        "فقرة_الكفالات": "",
        "الشروط_الخاصة": "",
        "الشروط_العامة": ""
    }

    def first_nonempty(cur, new):
        return cur if (cur and str(cur).strip()) else (new if (new and str(new).strip()) else cur)

    for p in parts or []:
        merged["الفريق_الأول"]    = first_nonempty(merged["الفريق_الأول"],    p.get("الفريق_الأول"))
        merged["الفريق_الثاني"]   = first_nonempty(merged["الفريق_الثاني"],   p.get("الفريق_الثاني"))
        merged["تاريخ_البدء"]     = first_nonempty(merged["تاريخ_البدء"],     p.get("تاريخ_البدء"))
        merged["تاريخ_الانتهاء"]  = first_nonempty(merged["تاريخ_الانتهاء"],  p.get("تاريخ_الانتهاء"))

        if p.get("ملخص_الاتفاقية"):
            if merged["ملخص_الاتفاقية"]:
                merged["ملخص_الاتفاقية"] += "\n• " + p["ملخص_الاتفاقية"].strip()
            else:
                merged["ملخص_الاتفاقية"] = "• " + p["ملخص_الاتفاقية"].strip()

        if p.get("المواد"):
            merged["المواد"].extend([x for x in p["المواد"] if isinstance(x, dict)])

        for k in ["فقرة_الكفالات","الشروط_الخاصة","الشروط_العامة"]:
            if p.get(k):
                if merged[k]:
                    merged[k] += "\n" + p[k].strip()
                else:
                    merged[k] = p[k].strip()

    return merged

def _model_fallbacks(selected: str) -> list:
    """
    يبني قائمة موديلات نجربها بالتسلسل.
    """
    seen, out = set(), []
    def add(m):
        if m and m not in seen:
            seen.add(m); out.append(m)

    add(selected)
    if "2.5" in selected:
        add(selected.replace("2.5", "1.5"))

    add("models/gemini-1.5-pro")
    add("models/gemini-1.5-flash")
    add("models/gemini-1.5-pro-001")
    add("models/gemini-1.5-flash-001")
    return out


# ===========================
# 5️⃣ تحليل بالـ Gemini
# ===========================
def analyze_agreement_with_gemini(text: str, selected_model: str, debug: bool = False) -> dict:
    """
    أولاً نحاول تحليل كامل النص. إذا فشل (رد فاضي/مرفوض/خطأ)،
    ننتقل للخطة (ب): تقسيم النص لقطع وتشغيل التحليل على كل قطعة، ثم دمج النتيجة.
    """
    prompt_full = AGREEMENT_PROMPT_TEMPLATE.format(doc_text=text)

    def run_once(model_name: str, prompt: str) -> str:
        model = genai.GenerativeModel(model_name=model_name)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.15, "max_output_tokens": 8192}
        )
        raw = getattr(resp, "text", "") or ""
        if not raw and getattr(resp, "candidates", None):
            parts = [p.text for c in resp.candidates for p in getattr(c.content, "parts", []) if getattr(p, "text", "")]
            raw = "\n".join(parts)
        return raw

    # 1) محاولة كاملة مع fallback على الموديلات
    for m in _model_fallbacks(selected_model):
        try:
            raw = run_once(m, prompt_full)
            if debug:
                st.caption(f"📄 Raw (full) from {m}:")
                st.code((raw or "")[:1200] + ("..." if raw and len(raw) > 1200 else ""))
            parsed = parse_tagged_response(raw)
            # لو كل الحقول فارغة تقريباً، اعتبره فشل منطقي
            if any([parsed.get("الفريق_الأول"), parsed.get("الفريق_الثاني"), parsed.get("المواد")]):
                return parsed
        except Exception as e:
            if debug:
                st.warning(f"⚠️ فشل محاولة كاملة على {m}: {type(e).__name__}: {e}")

    # 2) خطة (ب): تجزئة النص
    chunks = chunk_text(text, max_chars=10000)
    parts = []
    for idx, ch in enumerate(chunks, 1):
        prompt_chunk = AGREEMENT_PROMPT_TEMPLATE.format(doc_text=ch)
        ok = False
        for m in _model_fallbacks(selected_model):
            try:
                raw = run_once(m, prompt_chunk)
                if debug:
                    st.caption(f"📄 Raw (chunk {idx}/{len(chunks)}) from {m}:")
                    st.code((raw or "")[:800] + ("..." if raw and len(raw) > 800 else ""))
                parsed = parse_tagged_response(raw)
                parts.append(parsed)
                ok = True
                break
            except Exception as e:
                if debug:
                    st.warning(f"⚠️ فشل chunk {idx} على {m}: {type(e).__name__}: {e}")
                continue
        if not ok and debug:
            st.error(f"❌ لم ننجح في chunk {idx}")

    if not parts:
        raise RuntimeError("فشل التحليل عبر جميع الموديلات (كامل + مجزأ).")

    # دمج النتائج الجزئية
    merged = merge_results(parts)
    return merged

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
