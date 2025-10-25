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
# 3) Prompt الوسوم (مع تهريب الأقواس)
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
[3–5 نقاط قصيرة جداً تلخّص الاتفاقية، نقطة لكل سطر تبدأ بـ "- "]
<<<END_SUMMARY>>>

# المصفوفة التالية فقط بصيغة JSON صحيحة. لا تضف أي نص خارج الأقواس.
# أولوية قصوى لاستخراج المواد بشكل صحيح.
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
[حوّل فقرة الكفالات إلى نقاط قصيرة جداً، نقطة لكل سطر تبدأ بـ "- "]
<<<END_WARRANTIES>>>

<<<SPECIAL_TERMS>>>
[حوّل الشروط الخاصة إلى نقاط قصيرة جداً، نقطة لكل سطر تبدأ بـ "- "]
<<<END_SPECIAL_TERMS>>>

<<<GENERAL_TERMS>>>
[حوّل الشروط العامة إلى نقاط قصيرة جداً، نقطة لكل سطر تبدأ بـ "- "]
<<<END_GENERAL_TERMS>>>

تعليمات مهمة:
- ركّز على استخراج (التواريخ + المواد) بدقة عالية.
- وحِّد الأسعار بالدينار فقط (اجمع الدينار + الفلس/1000 إن ظهرت منفصلة).
- التزم بالبنية أعلاه حرفياً.
النص:
----------------
{doc_text}
"""



# ===========================
# 4) تحليل الوسوم + تنظيف JSON المواد
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

    def to_points(text: str) -> list:
        """حوّل سطور تبدأ بـ '- ' إلى نقاط قصيرة نظيفة."""
        if not text:
            return []
        lines = [re.sub(r"^\s*-\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]
        # احتفظ فقط بالسطر الذي كان يبدأ بـ "- " أو قصير جداً
        out = []
        for ln in lines:
            if ln.startswith("- "):
                ln = ln[2:].strip()
            out.append(ln)
        # فلترة الفراغات وتحديد حد أقصى منطقي
        out = [x for x in out if x]
        return out[:20]  # سقف 20 نقطة

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
        # اقتباس المفاتيح غير المقتبسة
        items_json = re.sub(r'([{,]\s*)([A-Za-z0-9_ء-ي]+)\s*:', r'\1"\2":', items_json)
        # أحياناً اسم_المادة يُعاد رقمًا → اقتبسه كسلسلة
        items_json = re.sub(r'("اسم_المادة"\s*:\s*)(-?\d+(?:\.\d+)?)', r'\1"\2"', items_json)

        try:
            parsed = json.loads(items_json)
        except Exception:
            items_json2 = re.sub(r"\s+\n\s+", "\n", items_json)
            try:
                parsed = json.loads(items_json2)
            except Exception:
                parsed = []

        if isinstance(parsed, dict):
            items = [parsed]
        elif isinstance(parsed, list):
            items = [x for x in parsed if isinstance(x, dict)]
        else:
            items = []
    else:
        items = []

    summary_txt   = g("<<<SUMMARY>>>",        "<<<END_SUMMARY>>>")
    warranties    = g("<<<WARRANTIES>>>",     "<<<END_WARRANTIES>>>")
    special_terms = g("<<<SPECIAL_TERMS>>>",  "<<<END_SPECIAL_TERMS>>>")
    general_terms = g("<<<GENERAL_TERMS>>>",  "<<<END_GENERAL_TERMS>>>")

    return {
        "الفريق_الأول":   g("<<<TEAM_A>>>", "<<<END_TEAM_A>>>"),
        "الفريق_الثاني":  g("<<<TEAM_B>>>", "<<<END_TEAM_B>>>"),
        "تاريخ_البدء":    g("<<<DATE_START>>>", "<<<END_DATE_START>>>"),
        "تاريخ_الانتهاء": g("<<<DATE_END>>>",   "<<<END_DATE_END>>>"),
        "ملخص_الاتفاقية": summary_txt,             # النص الأصلي (احتياط)
        "ملخص_الاتفاقية_نقاط": to_points(summary_txt),
        "المواد": items,
        "فقرة_الكفالات": warranties,               # النص الأصلي (احتياط)
        "فقرة_الكفالات_نقاط": to_points(warranties),
        "الشروط_الخاصة": special_terms,            # النص الأصلي (احتياط)
        "الشروط_الخاصة_نقاط": to_points(special_terms),
        "الشروط_العامة": general_terms,            # النص الأصلي (احتياط)
        "الشروط_العامة_نقاط": to_points(general_terms),
    }



# ===========================
# 4.1) أدوات التجزئة والدمج + قائمة الموديلات الاحتياطية
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
# 4.2) تنسيق جدول المواد (تنظيف أرقام + أعمدة مشتقة + عرض)
# ===========================
def _arabic_digits_to_western(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(trans)

def _to_float(val):
    """يحاول تحويل أي تمثيل رقمي (عربي/إنجليزي) إلى float بأمان."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # طبّع الأرقام العربية + الفواصل العربية
    s = _arabic_digits_to_western(s)
    s = s.replace("،", ",").replace("٫", ".")
    # أزل الكلمات الشائعة
    s = re.sub(r"(دينار|JD|د\.|فلس|ضريبة|%)", "", s, flags=re.I).strip()
    # أزل فواصل الآلاف والمسافات
    s = s.replace(",", "").replace(" ", "")
    # التقط أول رقم عشري صالح (يدعم السالب)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def normalize_items_table(items: list):
    """يعيد DataFrame منظّم + أرقام موحدة + أعمدة مشتقة ومجاميع."""
    import pandas as pd
    cols = [
        "اسم_المادة",
        "سعر_الشراء_قبل_الضريبة",
        "سعر_الشراء_مع_الضريبة",
        "الكمية_المشتراة_بالحبة",
        "القيمة_المشتراة_بالدينار",
        "نسبة_ضريبة_المبيعات",
    ]
    # ضمان وجود الأعمدة
    norm_rows = []
    for it in (items or []):
        row = {k: it.get(k) for k in cols}
        norm_rows.append(row)
    df = pd.DataFrame(norm_rows, columns=cols)

    # تحويل الأرقام
    for c in ["سعر_الشراء_قبل_الضريبة", "سعر_الشراء_مع_الضريبة", "الكمية_المشتراة_بالحبة",
              "القيمة_المشتراة_بالدينار", "نسبة_ضريبة_المبيعات"]:
        df[c] = df[c].apply(_to_float)

    # نسبة الضريبة: إن كانت بين 1..100 اعتبرها % وقسمها على 100
    df["نسبة_ضريبة_المبيعات"] = df["نسبة_ضريبة_المبيعات"].apply(
        lambda x: (x/100.0) if (x is not None and 1.0 <= x <= 100.0) else x
    )

    # أعمدة مشتقة
    df["قيمة_قبل_الضريبة_(حساب)"] = (
        (df["سعر_الشراء_قبل_الضريبة"].fillna(0.0)) * (df["الكمية_المشتراة_بالحبة"].fillna(0.0))
    )
    df["قيمة_مع_الضريبة_(حساب)"] = (
        (df["سعر_الشراء_مع_الضريبة"].fillna(0.0)) * (df["الكمية_المشتراة_بالحبة"].fillna(0.0))
    )

    # إن كانت "القيمة_المشتراة_بالدينار" فارغة، نملأها بالحساب المتاح
    df["القيمة_المشتراة_بالدينار"] = df["القيمة_المشتراة_بالدينار"].fillna(
        df["قيمة_مع_الضريبة_(حساب)"].where(df["قيمة_مع_الضريبة_(حساب)"] > 0, df["قيمة_قبل_الضريبة_(حساب)"])
    )

    # ترتيب أعمدة العرض
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
    # مجاميع
    totals = {
        "إجمالي_الكمية": float(df["الكمية_المشتراة_بالحبة"].fillna(0).sum()),
        "إجمالي_قيمة_قبل_الضريبة_(حساب)": float(df["قيمة_قبل_الضريبة_(حساب)"].fillna(0).sum()),
        "إجمالي_قيمة_مع_الضريبة_(حساب)": float(df["قيمة_مع_الضريبة_(حساب)"].fillna(0).sum()),
        "إجمالي_القيمة_المشتراة_بالدينار": float(df["القيمة_المشتراة_بالدينار"].fillna(0).sum()),
    }
    return df[display_cols], totals

def render_items_table(items: list, title: str = "📦 المواد ضمن الاتفاقية"):
    """يعرض الجدول + ملخص + أزرار تنزيل."""
    import pandas as pd
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

    df, totals = normalize_items_table(items)

    if df.empty:
        st.info("لا توجد مواد مستخرجة.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # عرض الجدول
    st.dataframe(df, use_container_width=True, height=420)

    # ملخص سريع
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي الكمية", f"{totals['إجمالي_الكمية']:.0f}")
    c2.metric("إجمالي قبل الضريبة", f"{totals['إجمالي_قيمة_قبل_الضريبة_(حساب)']:.3f}")
    c3.metric("إجمالي مع الضريبة", f"{totals['إجمالي_قيمة_مع_الضريبة_(حساب)']:.3f}")
    c4.metric("إجمالي القيمة المدخلة", f"{totals['إجمالي_القيمة_المشتراة_بالدينار']:.3f}")

    # تنزيل CSV/Excel
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
            # لو بعض الحقول تملأت أو في مواد، اعتبرها ناجحة
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
    try:
        models_list = genai.list_models()
        models = [m.name for m in models_list if "generateContent" in m.supported_generation_methods]
    except Exception:
        # fallback للأسماء الشائعة لو فشل list_models
        models = [
            "models/gemini-1.5-pro",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro-001",
            "models/gemini-1.5-flash-001",
        ]
    selected_model = st.selectbox("اختر الموديل:", models, index=0)
else:
    st.error("❌ لم يتم العثور على مفتاح Gemini")
    selected_model = None

debug = st.toggle("🧠 إظهار مخرجات التشخيص (Raw)")

if "ocr_text" in st.session_state and selected_model and st.button("تحليل الاتفاقية"):
    try:
        result = analyze_agreement_with_gemini(st.session_state["ocr_text"], selected_model, debug)
        st.success("✅ تم التحليل بنجاح")

        # عرض الأقسام بشكل أنيق
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="card"><div class="section-title">👥 الفريق الأول</div>', unsafe_allow_html=True)
                st.write(result.get("الفريق_الأول") or "—")
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="card"><div class="section-title">👥 الفريق الثاني</div>', unsafe_allow_html=True)
                st.write(result.get("الفريق_الثاني") or "—")
                st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            c3, c4 = st.columns(2)
            with c3:
                st.markdown('<div class="card"><div class="section-title">📅 تاريخ البدء</div>', unsafe_allow_html=True)
                st.write(result.get("تاريخ_البدء") or "—")
                st.markdown('</div>', unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="card"><div class="section-title">📅 تاريخ الانتهاء</div>', unsafe_allow_html=True)
                st.write(result.get("تاريخ_الانتهاء") or "—")
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="section-title">📝 ملخص الاتفاقية</div>', unsafe_allow_html=True)
        st.write(result.get("ملخص_الاتفاقية") or "—")
        st.markdown('</div>', unsafe_allow_html=True)

        # جدول المواد
        render_items_table(result.get("المواد", []) or [])

        # باقي الأقسام النصية
        with st.container():
            st.markdown('<div class="card"><div class="section-title">🛡️ فقرة الكفالات</div>', unsafe_allow_html=True)
            st.write(result.get("فقرة_الكفالات") or "—")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card"><div class="section-title">⚙️ الشروط الخاصة</div>', unsafe_allow_html=True)
            st.write(result.get("الشروط_الخاصة") or "—")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card"><div class="section-title">📜 الشروط العامة</div>', unsafe_allow_html=True)
            st.write(result.get("الشروط_العامة") or "—")
            st.markdown('</div>', unsafe_allow_html=True)

        # عرض JSON الخام (للتنزيل/المراجعة)
        with st.expander("🔎 JSON الكامل للنتيجة"):
            st.json(result)

    except Exception as e:
        st.error(f"❌ فشل التحليل: {e}")
