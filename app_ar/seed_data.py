"""Seed Arabic FAQ data — run: python -m app_ar.seed_data"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from app_ar.database import apply_schema, create_ticket, get_conn, insert_faq
from app_ar.embeddings import embed_batch

FAQ_ARTICLES_AR = [
    {
        "intent": "password_reset",
        "title": "كيفية إعادة تعيين كلمة المرور",
        "content": (
            "لإعادة تعيين كلمة المرور:\n"
            "1. قم بزيارة https://example.com/forgot-password\n"
            "2. أدخل بريدك الإلكتروني المسجل\n"
            "3. تحقق من بريدك الوارد لرابط إعادة التعيين (صالحة لمدة 30 دقيقة)\n"
            "4. انقر على الرابط واختر كلمة مرور جديدة\n\n"
            "إذا لم يصلك البريد خلال 5 دقائق، تحقق من مجلد الرسائل غير المرغوب فيها."
        ),
    },
    {
        "intent": "password_reset",
        "title": "الحساب مقفل بعد محاولات تسجيل دخول فاشلة",
        "content": (
            "يتم قفل حسابك بعد 5 محاولات تسجيل دخول فاشلة لأمنك.\n\n"
            "لفتح الحساب:\n"
            "1. انتظر 15 دقيقة للفتح التلقائي، أو\n"
            "2. استخدم 'نسيت كلمة المرور' لإعادة التعيين فوراً عبر البريد الإلكتروني\n\n"
            "إذا كنت تعتقد أن محاولات الدخول لم تكن من قبلك، اتصل بـ security@example.com."
        ),
    },
    {
        "intent": "billing",
        "title": "فهم فاتورتك",
        "content": (
            "يتم إصدار فاتورتك في اليوم الأول من كل شهر وتغطي الشهر السابق.\n\n"
            "لعرض الفواتير:\n"
            "1. سجل الدخول إلى حسابك\n"
            "2. اذهب إلى الحساب ← الفواتير ← سجل الفواتير\n\n"
            "يتم إرسال الفواتير أيضاً إلى بريدك الإلكتروني المسجل. "
            "للنزاعات، اتصل بـ billing@support.example.com."
        ),
    },
    {
        "intent": "billing",
        "title": "كيفية الحصول على استرداد مالي",
        "content": (
            "يتم معالجة طلبات استرداد المبالغ خلال 5-7 أيام عمل.\n\n"
            "الأهلية:\n"
            "• الطلبات خلال 30 يوماً من تاريخ الدفع\n"
            "• الخدمة لم تستخدم بالكامل (يتم تطبيق الاسترداد النسبي)\n\n"
            "لتقديم طلب استرداد، اذهب إلى الحساب ← الفواتير ← طلب استرداد، "
            "أو اتصل بـ billing@support.example.com مع رقم فاتورتك."
        ),
    },
    {
        "intent": "technical_support",
        "title": "التطبيق لا يعمل أو يظهر أخطاء",
        "content": (
            "خطوات استكشاف الأخطاء وإصلاحها:\n"
            "1. تحديث الصفحة: Ctrl+Shift+R (Windows) / Cmd+Shift+R (Mac)\n"
            "2. مسح ذاكرة التخزين المؤقت وملفات تعريف الارتباط في المتصفح\n"
            "3. جرب نافذة تصفح متخفي\n"
            "4. جرب متصفح آخر\n"
            "5. تحقق من status.example.com للحوادث الحالية\n\n"
            "إذا استمرت المشكلة، سيقوم فريقنا الهندسي بالتحقيق."
        ),
    },
    {
        "intent": "technical_support",
        "title": "أخطاء API وحدود الطلبات",
        "content": (
            "أخطاء API الشائعة:\n"
            "• 429 طلبات كثيرة جداً: تجاوزت حد الطلبات. انتظر 60 ثانية وحاول مرة أخرى.\n"
            "• 401 غير مصرح: تحقق من صحة مفتاح API الخاص بك.\n"
            "• 500 خطأ خادم داخلي: مشكلة مؤقتة — حاول مرة أخرى.\n\n"
            "حدود الطلبات: 100 طلب/دقيقة للنسخة المجانية، 1000/دقيقة للنسخة المدفوعة."
        ),
    },
]

SEED_TICKETS_AR = [
    {"message": "تم خصم مبلغ مني مرتين هذا الشهر", "intent": "billing"},
    {"message": "لا أستطيع تسجيل الدخول إلى حسابي أبداً", "intent": "password_reset"},
    {"message": "لوحة التحكم تظهر خطأ 500 على كروم", "intent": "technical_support"},
    {"message": "أحتاج استرداد مالي لاشتراك الشهر الماضي", "intent": "billing"},
    {"message": "بريد إعادة تعيين كلمة المرور لم يصلني", "intent": "password_reset"},
    {"message": "API دايم يرد علي بـ 429", "intent": "technical_support"},
]


def already_seeded() -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM faq_articles")
            return cur.fetchone()[0] >= len(FAQ_ARTICLES_AR)


def seed() -> None:
    print("جاري تطبيق هيكل قاعدة البيانات...")
    apply_schema()

    if already_seeded():
        print(f"البيانات موجودة بالفعل ({len(FAQ_ARTICLES_AR)} مقالة). تم التخطي.")
        return

    print(f"جارٍ تضمين {len(FAQ_ARTICLES_AR)} مقالة...")
    faq_texts = [f"{a['title']} {a['content']}" for a in FAQ_ARTICLES_AR]
    faq_embeddings = embed_batch(faq_texts)
    for article, emb in zip(FAQ_ARTICLES_AR, faq_embeddings):
        insert_faq(article["intent"], article["title"], article["content"], emb)
    print(f"  ✓ {len(FAQ_ARTICLES_AR)} مقالة تم إدراجها")

    print(f"جارٍ تضمين {len(SEED_TICKETS_AR)} تذكرة...")
    ticket_texts = [t["message"] for t in SEED_TICKETS_AR]
    ticket_embeddings = embed_batch(ticket_texts)
    for ticket, emb in zip(SEED_TICKETS_AR, ticket_embeddings):
        create_ticket(ticket["message"], ticket["intent"], emb)
    print(f"  ✓ {len(SEED_TICKETS_AR)} تذكرة تم إدراجها")

    print("تم الانتهاء من البذر.")


if __name__ == "__main__":
    seed()
