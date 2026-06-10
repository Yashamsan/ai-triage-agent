"""Seed 6 FAQ articles + 6 support tickets with embeddings.

Run once:  python -m app.seed_data
Safe to re-run — checks row count before inserting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from app.database import apply_schema, insert_faq, create_ticket, get_conn
from app.embeddings import embed_batch

FAQ_ARTICLES = [
    {
        "intent": "password_reset",
        "title": "How to reset your password",
        "content": (
            "To reset your password:\n"
            "1. Visit https://example.com/forgot-password\n"
            "2. Enter your registered email address\n"
            "3. Check your inbox for a reset link (expires in 30 minutes)\n"
            "4. Click the link and choose a new password\n\n"
            "If the email doesn't arrive within 5 minutes, check your spam folder."
        ),
    },
    {
        "intent": "password_reset",
        "title": "Account locked after failed login attempts",
        "content": (
            "Your account is locked after 5 failed login attempts for security.\n\n"
            "To unlock it:\n"
            "1. Wait 15 minutes for automatic unlock, OR\n"
            "2. Use 'Forgot Password' to reset via email immediately\n\n"
            "If you believe the login attempts were not made by you, contact security@example.com."
        ),
    },
    {
        "intent": "billing",
        "title": "Understanding your invoice",
        "content": (
            "Your invoice is generated on the 1st of each month and covers the previous month.\n\n"
            "To view invoices:\n"
            "1. Log in to your account\n"
            "2. Go to Account → Billing → Invoice History\n\n"
            "Invoices are also emailed to your registered address. "
            "For billing disputes, contact billing@support.example.com."
        ),
    },
    {
        "intent": "billing",
        "title": "How to get a refund",
        "content": (
            "Refunds are processed within 5–7 business days to the original payment method.\n\n"
            "Eligibility:\n"
            "• Requests made within 30 days of charge\n"
            "• Service was not fully used (pro-rata refund applies)\n\n"
            "To request a refund, visit Account → Billing → Request Refund, "
            "or contact billing@support.example.com with your invoice number."
        ),
    },
    {
        "intent": "technical_support",
        "title": "App not loading or showing errors",
        "content": (
            "Common troubleshooting steps:\n"
            "1. Hard refresh: Ctrl+Shift+R (Windows) / Cmd+Shift+R (Mac)\n"
            "2. Clear browser cache and cookies\n"
            "3. Try an incognito/private window\n"
            "4. Switch to a different browser\n"
            "5. Check status.example.com for ongoing incidents\n\n"
            "If the issue persists after these steps, our engineering team will investigate."
        ),
    },
    {
        "intent": "technical_support",
        "title": "API errors and rate limits",
        "content": (
            "Common API errors:\n"
            "• 429 Too Many Requests: you've hit the rate limit. Wait 60 seconds and retry.\n"
            "• 401 Unauthorized: check your API key is valid and not expired.\n"
            "• 500 Internal Server Error: temporary issue — retry with exponential backoff.\n\n"
            "Rate limits: 100 requests/minute on Free tier, 1000/minute on Premium. "
            "View your usage at Account → API → Usage Dashboard."
        ),
    },
]

SEED_TICKETS = [
    {"message": "I've been charged twice this month", "intent": "billing"},
    {"message": "I can't log into my account at all", "intent": "password_reset"},
    {"message": "The dashboard shows a 500 error on Chrome", "intent": "technical_support"},
    {"message": "I need a refund for last month's subscription", "intent": "billing"},
    {"message": "My password reset email never arrived", "intent": "password_reset"},
    {"message": "The API keeps returning 429 errors", "intent": "technical_support"},
]


def already_seeded() -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM faq_articles")
            return cur.fetchone()[0] >= len(FAQ_ARTICLES)


def seed() -> None:
    print("Applying schema...")
    apply_schema()

    if already_seeded():
        print(f"Already seeded ({len(FAQ_ARTICLES)} FAQ articles found). Skipping.")
        return

    print(f"Embedding {len(FAQ_ARTICLES)} FAQ articles...")
    faq_texts = [f"{a['title']} {a['content']}" for a in FAQ_ARTICLES]
    faq_embeddings = embed_batch(faq_texts)
    for article, emb in zip(FAQ_ARTICLES, faq_embeddings):
        insert_faq(article["intent"], article["title"], article["content"], emb)
    print(f"  ✓ {len(FAQ_ARTICLES)} FAQ articles inserted")

    print(f"Embedding {len(SEED_TICKETS)} seed tickets...")
    ticket_texts = [t["message"] for t in SEED_TICKETS]
    ticket_embeddings = embed_batch(ticket_texts)
    for ticket, emb in zip(SEED_TICKETS, ticket_embeddings):
        create_ticket(ticket["message"], ticket["intent"], emb)
    print(f"  ✓ {len(SEED_TICKETS)} tickets inserted")

    print("Seeding complete.")


if __name__ == "__main__":
    seed()
