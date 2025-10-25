import streamlit as st
import google.generativeai as genai
import json, re, os, io, base64, tempfile
from google.cloud import vision
from PIL import Image
import pdfplumber

# ===========================
# 1) مفاتيح وربط الخدمات
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
# 2) OCR شامل (صور + PDF)
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
# 3) Prompt الوسوم (مواد فقط)
# ===========================
AGREEMENT_PROMPT_TEMPLATE = r"""
أنت مساعد لاستخراج **قائمة المواد فقط** من نص اتفاقية أو عرض.
أعد الرد **بالضبط** ضمن الوسمين التاليين، ولا تضف أي نص خارجهما:

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

تعليمات مهمة:
- أعِد مصفوفة JSON صحيحة فقط داخل <<<ITEMS_JSON_ARRAY>>>…<<<END_ITEMS_JSON_ARRAY>>>.
- "اسم_المادة" نص إجباري.
- باقي الحقول قيم رقمية (عشرية) بالدينار بعد دمج الدينار + الفلس/1000 إن ظهرت منفصلة.
- "الكمية_المشتراة_بالحبة" رقم صحيح.
- "نسبة_ضريبة_المبيعات" كقيمة عشرية (مثلاً 0.16 وليس 16%).
- لا تعليقات، لا أسطر شرح، لا فواصل زائدة قبل ] أو }}.
- إن لم توجد مواد، أعد [].

النص:
----------------
{doc_text}
"""


# ===========================
# 4) تحليل الوسوم + تنظيف JSON المواد
# ===========================
def parse_tagged_response(raw: str) -> dict:
    import json, re
    raw = re.sub(r"[\u200E\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u200B\u200C\u200D]", "", raw or "").strip()

    m = re.search(r"<<<ITEMS_JSON_ARRAY>>>(.*?)<<<END_ITEMS_JSON_ARRAY>>>", raw, flags=re.S)
    items_json = (m.group(1).strip() if m else "")

    items = []
    if items_json:
        # إزالة code fences إن وُجدت
        items_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", items_json, flags=re.I | re.M).strip()
        # تطبيع علامات الاقتباس والفواصل العربية
        items_json = (items_json
                      .replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
                      .replace("،", ",").replace("٫", "."))
        # إزالة الفواصل الزائدة قبل الأقواس
        items_json = re.sub(r",\s*([}\]])", r"\1", items_json)
        # اقتباس المفاتيح غير المقتبسة
        items_json = re.sub(r'([{,]\s*)([A-Za-z0-9_ء-ي]+)\s*:', r'\1"\2":', items_json)
        # اسم_المادة يجب أن يكون نصًا
        items_json = re.sub(r'("اسم_المادة"\s*:\s*)(-?\d+(?:\.\d+)?)', r'\1"\2"', items_json)

        try:
            parsed = json.loads(items_json)
        except Exception:
            try:
                parsed = json.loads(re.sub(r"\s+\n\s+", "\n", items_json))
            except Exception:
                parsed = []

        if isinstance(parsed, dict):
            items = [parsed]
        elif isinstance(parsed, list):
            items = [x for x in parsed if isinstance(x, dict)]
        else:
            items = []
    return {"المواد": items}


# ===========================
# 4.1) أدوات التجزئة والدمج + الفالْباكس
# ===========================
def chunk_text(text: str, max_chars: int = 10000) -> list:
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        nl = text.rfind("\n", start, end)
        if nl == -1 or nl <= start + int(max_chars * 0.5):
            nl = end
        chunk = text[start:nl].strip()
        if chunk:
            chunks.append(chunk)
        start = nl
    return chunks

def merge_items_only(parts: list) -> dict:
    merged = {"المواد": []}
    for p in parts or []:
        its = p.get("المواد") or []
        merged["المواد"].extend([x for x in its if isinstance(x, dict)])
    return merged

def _sanitize_model_name(name: str) -> str:
    return (name or "").replace("models/", "").strip()

def _available_fallbacks(selected: str) -> list:
    """
    مقتصر على الموديلات المتاحة في حسابك (2.5/2.0).
    عدّل الترتيب إذا تحب الدقة (pro) أو السرعة (flash/lite).
    """
    wanted = []
    sel = _sanitize_model_name(selected)
    def add(m):
        m = _sanitize_model_name(m)
        if m and m not in wanted:
            wanted.append(m)

    add(sel)
    add("gemini-2.5-pro")
    add("gemini-2.5-flash")
    add("gemini-2.5-flash-lite")
    add("gemini-2.0-flash")
    add("gemini-2.0-flash-lite")
    add("gemini-2.0-flash-exp")
    return wanted


# ===========================
# 4.2) تنسيق جدول المواد (تنظيف أرقام + أعمدة مشتقة + عرض)
# ===========================
def _arabic_digits_to_western(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(trans)

def _to_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = _arabic_digits_to_western(s)
    s = s.replace("،", ",").replace("٫", ".")
    s = re.sub(r"(دينار|JD|د\.|فلس|ضريبة|%)", "", s, flags=re.I).strip()
    s = s.replace(",", "").replace(" ", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def normalize_items_table(items: list):
    import pandas as pd
    cols = [
        "اسم_المادة",
        "سعر_الشراء_قبل_الضريبة",
        "سعر_الشراء_مع_الضريبة",
        "الكمية_المشتراة_بالحبة",
        "القيمة_المشتراة_بالدينار",
        "نسبة_ضريبة_المبيعات",
    ]
    norm_rows = []
    for it in (items or []):
        row = {k: it.get(k) for k in cols}
        norm_rows.append(row)
    df = pd.DataFrame(norm_rows, columns=cols)

    for c in ["سعر_الشراء_قبل_الضريبة", "سعر_الشراء_مع_الضريبة", "الكمية_المشتراة_بالحبة",
              "القيمة_المشتراة_بالدينار", "نسبة_ضريبة_المبيعات"]:
        df[c] = df[c].apply(_to_float)

    df["نسبة_ضريبة_المبيعات"] = df["نسبة_ضريبة_المبيعات"].apply(
        lambda x: (x/100.0) if (x is not None and 1.0 <= x <= 100.0) else x
    )

    df["قيمة_قبل_الضريبة_(حساب)"] = (
        (df["سعر_الشراء_قبل_الضريبة"].fillna(0.0)) * (df["الكمية_المشتراة_بالحبة"].fillna(0.0))
    )
    df["قيمة_مع_الضريبة_(حساب)"] = (
        (df["سعر_الشراء_مع_الضريبة"].fillna(0.0)) * (df["الكمية_المشتراة_بالحبة"].fillna(0.0))
    )

    df["القيمة_المشتراة_بالدينار"] = df["القيمة_المشتراة_بالدينار"].fillna(
        df["قيمة_مع_الضريبة_(حساب)"].where(df["قيمة_مع_الضريبة_(حساب)"] > 0, df["قيمة_قبل_الضريبة_(حساب)"])
    )

    display_cols = [
        "اسم_المادة",
        "الكمية_المشتراة_بالحبة",
        "سعر_الشراء_قبل_الضريبة",
        "سعر_الشراء_مع_الضريبة",
        "نسبة_ضريبة_المبيعات",
        "قيمة_قبل_الضريبة_(حساب)",
        "قيمة_مع_الضريبة_(حساب)",
        "القيمة_المشتراة_بالدينار",
    ]
    totals = {
        "إجمالي_الكمية": float(df["الكمية_المشتراة_بالحبة"].fillna(0).sum()),
        "إجمالي_قيمة_قبل_الضريبة_(حساب)": float(df["قيمة_قبل_الضريبة_(حساب)"].fillna(0).sum()),
        "إجمالي_قيمة_مع_الضريبة_(حساب)": float(df["قيمة_مع_الضريبة_(حساب)"].fillna(0).sum()),
        "إجمالي_القيمة_المشتراة_بالدينار": float(df["القيمة_المشتراة_بالدينار"].fillna(0).sum()),
    }
    return df[display_cols], totals

def render_items_table(items: list, title: str = "📦 المواد ضمن الاتفاقية"):
    import pandas as pd
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    df, totals = normalize_items_table(items)

    if df.empty:
        st.info("لا توجد مواد مستخرجة.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.dataframe(df, use_container_width=True, height=420)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي الكمية", f"{totals['إجمالي_الكمية']:.0f}")
    c2.metric("إجمالي قبل الضريبة", f"{totals['إجمالي_قيمة_قبل_الضريبة_(حساب)']:.3f}")
    c3.metric("إجمالي مع الضريبة", f"{totals['إجمالي_قيمة_مع_الضريبة_(حساب)']:.3f}")
    c4.metric("إجمالي القيمة المدخلة", f"{totals['إجمالي_القيمة_المشتراة_بالدينار']:.3f}")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ تنزيل CSV", data=csv_bytes, file_name="items.csv", mime="text/csv")

    try:
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="items")
        st.download_button("⬇️ تنزيل Excel", data=buf.getvalue(),
                           file_name="items.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        pass

    st.markdown('</div>', unsafe_allow_html=True)


# ===========================
# 5) تحليل بالـ Gemini (محاولة كاملة ثم تجزئة)
# ===========================
def analyze_agreement_with_gemini(text: str, selected_model: str, debug: bool = False) -> dict:
    """
    نحاول تحليل كامل النص. إذا فشل أو جاء رد محجوب، نجزّئ النص ونحاول ثم ندمج "المواد" فقط.
    """
    prompt_full = AGREEMENT_PROMPT_TEMPLATE.format(doc_text=text)

    def run_once(model_name: str, prompt: str) -> str:
        model_name = _sanitize_model_name(model_name)
        model = genai.GenerativeModel(model_name=model_name)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.15, "max_output_tokens": 8192},
        )
        # لا تعتمد على resp.text؛ استخرج من candidates/parts وتجاوز الردود المحجوبة (finish_reason=2)
        texts = []
        for cand in getattr(resp, "candidates", []) or []:
            fr = getattr(cand, "finish_reason", None)
            if fr is not None and int(fr) == 2:
                continue
            content = getattr(cand, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    t = getattr(p, "text", None)
                    if t:
                        texts.append(t)
        return "\n".join(texts).strip()

    # 1) محاولة كاملة مع fallback على الموديلات المتاحة
    for m in _available_fallbacks(selected_model):
        try:
            raw = run_once(m, prompt_full)
            if debug:
                st.caption(f"📄 Raw (full) from {m}:")
                st.code((raw or "")[:1200] + ("..." if raw and len(raw) > 1200 else ""))
            parsed = parse_tagged_response(raw)
            # نجاح لو عندنا مواد (حتى لو فاضية بنجرّب التجزئة)
            if parsed.get("المواد"):
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
        for m in _available_fallbacks(selected_model):
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

    return merge_items_only(parts)


# ===========================
# 6) واجهة Streamlit
# ===========================
st.set_page_config(page_title="تحليل اتفاقيات المؤسسة الاستهلاكية العسكرية", layout="wide")

# لمسة CSS بسيطة
st.markdown("""
<style>
.section-title{font-weight:700;font-size:1.1rem;margin:8px 0 12px}
.card{background:#ffffff;border:1px solid #eee;border-radius:12px;padding:12px;margin-bottom:14px}
</style>
""", unsafe_allow_html=True)

st.title("📑 نظام تحليل اتفاقيات المؤسسة الاستهلاكية العسكرية")
st.markdown("باستخدام **Google Vision OCR + Gemini AI** — استخراج المواد فقط")

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
    try:
        # يمكنك تجاهل list_models والاكتفاء بالقائمة الثابتة بالأسفل
        _ = genai.list_models()
        models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-exp",
        ]
    except Exception:
        models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-exp",
        ]
    selected_model = st.selectbox("اختر الموديل:", models, index=0)
    selected_model = _sanitize_model_name(selected_model)
else:
    st.error("❌ لم يتم العثور على مفتاح Gemini")
    selected_model = None

debug = st.toggle("🧠 إظهار مخرجات التشخيص (Raw)")

if "ocr_text" in st.session_state and selected_model and st.button("تحليل (مواد فقط)"):
    try:
        result = analyze_agreement_with_gemini(st.session_state["ocr_text"], selected_model, debug)
        st.success("✅ تم استخراج المواد")
        # عرض المواد فقط
        render_items_table(result.get("المواد", []) or [], title="📦 المواد المستخرجة")
        # خيار: إظهار JSON للمواد فقط
        with st.expander("🔎 JSON المواد"):
            st.json(result.get("المواد", []))
    except Exception as e:
        st.error(f"❌ فشل التحليل: {e}")
