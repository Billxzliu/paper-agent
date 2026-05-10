import os
import json
import arxiv
import smtplib
import html
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import OpenAI


KEYWORDS = [
    "novel view synthesis",
    "single image novel view synthesis",
    "3D Gaussian Splatting",
    "Gaussian splatting",
    "3DGS",
    "feed-forward",
    "feedforward",
    "3D scene generation",
    "3D scene reconstruction",
    "3D reconstructions",
    "Gaussian Splatting Completion",
    "Video Generation",
    "3D Inpainting",
]

DAILY_LOOKBACK_HOURS = 24
MAX_AI_PAPERS = 15
DEEPSEEK_MODEL = "deepseek-v4-flash"
TIMEZONE = ZoneInfo("Asia/Shanghai")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_arxiv_query(keywords):
    """
    Build arXiv query automatically from KEYWORDS.
    """
    keyword_query = " OR ".join([f'"{kw}"' for kw in keywords])
    return f"cat:cs.CV AND ({keyword_query})"


def is_recent_paper(published_time, lookback_hours: int = DAILY_LOOKBACK_HOURS) -> bool:
    """
    Keep only papers published within the recent daily window.
    """
    now_utc = datetime.now(timezone.utc)

    if published_time.tzinfo is None:
        published_time = published_time.replace(tzinfo=timezone.utc)
    else:
        published_time = published_time.astimezone(timezone.utc)

    return published_time >= now_utc - timedelta(hours=lookback_hours)


def simple_relevance_score(title: str, abstract: str) -> int:
    """
    First-stage keyword-based filtering.
    This step only reduces obvious noise before sending papers to DeepSeek.
    """
    text = f"{title} {abstract}".lower()
    score = 0

    for keyword in KEYWORDS:
        if keyword.lower() in text:
            score += 1

    if "novel view synthesis" in text and (
        "gaussian" in text or "3dgs" in text
    ):
        score += 3

    if "single image" in text and "novel view synthesis" in text:
        score += 2

    if ("feed-forward" in text or "feedforward" in text) and (
        "novel view synthesis" in text or "gaussian" in text
    ):
        score += 2

    if (
        "3d scene reconstruction" in text
        or "3d scene generation" in text
        or "3d reconstructions" in text
    ) and (
        "gaussian" in text
        or "splatting" in text
        or "3dgs" in text
    ):
        score += 3

    if "gaussian splatting completion" in text:
        score += 4

    if "3d inpainting" in text and (
        "gaussian" in text
        or "splatting" in text
        or "novel view" in text
        or "3d reconstruction" in text
    ):
        score += 3

    if "video generation" in text and (
        "3d" in text
        or "scene" in text
        or "novel view" in text
        or "camera" in text
        or "world model" in text
    ):
        score += 2

    return score


def search_arxiv(max_results: int = 150):
    """
    Search recent arXiv papers from cs.CV using KEYWORDS.
    Only papers published within DAILY_LOOKBACK_HOURS are kept.
    """
    query = build_arxiv_query(KEYWORDS)

    client_arxiv = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    total_seen = 0
    old_skipped = 0
    low_score_skipped = 0

    print("=" * 100)
    print("arXiv Query")
    print("=" * 100)
    print(query)
    print("=" * 100)

    for result in client_arxiv.results(search):
        total_seen += 1

        if not is_recent_paper(result.published):
            old_skipped += 1
            continue

        title = result.title.replace("\n", " ").strip()
        abstract = result.summary.replace("\n", " ").strip()
        score = simple_relevance_score(title, abstract)

        if score <= 0:
            low_score_skipped += 1
            continue

        published_utc = result.published.astimezone(timezone.utc)
        published_bj = result.published.astimezone(TIMEZONE)

        papers.append({
            "title": title,
            "authors": ", ".join(author.name for author in result.authors[:6]),
            "published_utc": published_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "published_bj": published_bj.strftime("%Y-%m-%d %H:%M:%S Beijing Time"),
            "abstract": abstract,
            "url": result.entry_id,
            "pdf": result.pdf_url,
            "keyword_score": score,
        })

    papers.sort(key=lambda x: x["keyword_score"], reverse=True)

    print("=" * 100)
    print("arXiv Search Summary")
    print("=" * 100)
    print(f"Total arXiv results checked: {total_seen}")
    print(f"Skipped old papers: {old_skipped}")
    print(f"Skipped low-score recent papers: {low_score_skipped}")
    print(f"Kept recent keyword-matched papers: {len(papers)}")
    print(f"Lookback window: last {DAILY_LOOKBACK_HOURS} hours")
    print("=" * 100)

    return papers


def ai_analyze_paper(title: str, abstract: str) -> dict:
    """
    Use DeepSeek to judge relevance and generate Chinese interpretation.
    """
    system_prompt = """
You are an expert research assistant in computer vision, 3D vision, and generative AI.

The user's research interests include:
novel view synthesis (NVS), single-image novel view synthesis, feed-forward 3D Gaussian Splatting (3DGS),
3D scene reconstruction, 3D scene generation, Gaussian Splatting completion, 3D inpainting,
video generation, world models, and efficient 3D representation learning.

Your task:
Given a paper title and abstract, decide whether this paper is relevant to the user's research interests.
Return your answer in Chinese.

Use this relevance scale:
0 = unrelated, ignore
1 = weakly related, maybe scan
2 = related, worth reading
3 = highly related, should read carefully

Be strict. Do not give high scores to general segmentation, detection, classification, language model,
medical imaging, or unrelated generation papers unless they clearly connect to 3D vision, NVS,
3DGS, 3D reconstruction, 3D generation, 3D inpainting, or video generation.

Return valid JSON only.
"""

    user_prompt = f"""
Paper title:
{title}

Abstract:
{abstract}

Please return JSON with the following fields:
{{
  "relevance_score": 0,
  "reading_priority": "忽略/略读/精读",
  "category": "NVS / single-image NVS / feed-forward 3DGS / 3DGS / 3D scene generation / 3D scene reconstruction / Gaussian completion / 3D inpainting / video generation / world model / weakly related / unrelated",
  "summary_zh": "",
  "why_relevant_zh": "",
  "technical_takeaway_zh": "",
  "research_value_zh": "",
  "limitations_zh": "",
  "final_recommendation_zh": ""
}}
"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "relevance_score": 0,
                "reading_priority": "解析失败",
                "category": "unknown",
                "summary_zh": content,
                "why_relevant_zh": "",
                "technical_takeaway_zh": "",
                "research_value_zh": "",
                "limitations_zh": "",
                "final_recommendation_zh": "",
            }

    except Exception as e:
        return {
            "relevance_score": 0,
            "reading_priority": "API调用失败",
            "category": "unknown",
            "summary_zh": f"DeepSeek API error: {str(e)}",
            "why_relevant_zh": "",
            "technical_takeaway_zh": "",
            "research_value_zh": "",
            "limitations_zh": "",
            "final_recommendation_zh": "",
        }


def analyze_papers_with_ai(papers, max_ai_papers: int = MAX_AI_PAPERS):
    """
    Analyze only top keyword-matched recent papers to control API cost.
    """
    analyzed = []
    total_to_analyze = min(len(papers), max_ai_papers)

    for idx, paper in enumerate(papers[:max_ai_papers], start=1):
        print(f"\nAnalyzing paper {idx}/{total_to_analyze}: {paper['title']}")

        ai_result = ai_analyze_paper(
            title=paper["title"],
            abstract=paper["abstract"],
        )

        paper["ai"] = ai_result
        analyzed.append(paper)

    analyzed.sort(
        key=lambda x: (
            safe_int(x["ai"].get("relevance_score", 0)),
            x["keyword_score"],
        ),
        reverse=True,
    )

    return analyzed


def get_useful_papers(papers):
    return [
        p for p in papers
        if safe_int(p.get("ai", {}).get("relevance_score", 0)) >= 1
    ]


def print_paper_report(papers):
    """
    Print final report to GitHub Actions logs.
    """
    useful_papers = get_useful_papers(papers)

    print("\n" + "=" * 100)
    print("Daily 3D Vision / NVS / 3DGS Paper Agent Report")
    print("=" * 100)
    print(f"Total AI-analyzed papers: {len(papers)}")
    print(f"Useful papers with relevance_score >= 1: {len(useful_papers)}")
    print(f"Lookback window: last {DAILY_LOOKBACK_HOURS} hours")
    print("=" * 100)

    if not useful_papers:
        print("今天没有发现与你方向明显相关的新论文。")
        return

    for idx, paper in enumerate(useful_papers, start=1):
        ai = paper["ai"]

        print("\n" + "=" * 100)
        print(f"[{idx}] {paper['title']}")
        print("=" * 100)
        print(f"Authors: {paper['authors']}")
        print(f"Published: {paper['published_bj']}")
        print(f"arXiv: {paper['url']}")
        print(f"PDF: {paper['pdf']}")
        print(f"Keyword score: {paper['keyword_score']}")
        print(f"AI relevance score: {ai.get('relevance_score', 0)}")
        print(f"Reading priority: {ai.get('reading_priority', '')}")
        print(f"Category: {ai.get('category', '')}")

        print("\n中文解读:")
        print(ai.get("summary_zh", ""))

        print("\n为什么相关:")
        print(ai.get("why_relevant_zh", ""))

        print("\n技术启发:")
        print(ai.get("technical_takeaway_zh", ""))

        print("\n研究价值:")
        print(ai.get("research_value_zh", ""))

        print("\n可能局限:")
        print(ai.get("limitations_zh", ""))

        print("\n最终建议:")
        print(ai.get("final_recommendation_zh", ""))


def priority_badge(score: int, priority: str) -> str:
    if score >= 3:
        return "🔥 强相关，建议精读"
    if score == 2:
        return "✅ 相关，建议阅读"
    if score == 1:
        return "👀 弱相关，可快速浏览"
    return priority or "忽略"


def build_email_html(papers):
    """
    Build HTML email report.
    """
    useful_papers = get_useful_papers(papers)
    now_bj = datetime.now(TIMEZONE)
    date_str = now_bj.strftime("%Y-%m-%d")

    html_parts = [
        "<html>",
        "<body style='font-family: Arial, sans-serif; line-height: 1.6; color: #222;'>",
        f"<h2>Daily 3D Vision / NVS / 3DGS Paper Agent Report - {date_str}</h2>",
        "<p>",
        f"<b>Total AI-analyzed papers:</b> {len(papers)}<br>",
        f"<b>Useful papers with relevance_score >= 1:</b> {len(useful_papers)}<br>",
        f"<b>Lookback window:</b> last {DAILY_LOOKBACK_HOURS} hours<br>",
        f"<b>Keywords:</b> {html.escape(', '.join(KEYWORDS))}",
        "</p>",
    ]

    if not useful_papers:
        html_parts.extend([
            "<hr>",
            "<p>今天没有发现与你方向明显相关的新论文。</p>",
            "</body>",
            "</html>",
        ])
        return "\n".join(html_parts)

    for idx, paper in enumerate(useful_papers, start=1):
        ai = paper["ai"]
        score = safe_int(ai.get("relevance_score", 0))
        badge = priority_badge(score, ai.get("reading_priority", ""))

        title = html.escape(paper.get("title", ""))
        authors = html.escape(paper.get("authors", ""))
        published = html.escape(paper.get("published_bj", ""))
        category = html.escape(str(ai.get("category", "")))
        reading_priority = html.escape(str(ai.get("reading_priority", "")))

        summary_zh = html.escape(str(ai.get("summary_zh", ""))).replace("\n", "<br>")
        why_relevant_zh = html.escape(str(ai.get("why_relevant_zh", ""))).replace("\n", "<br>")
        technical_takeaway_zh = html.escape(str(ai.get("technical_takeaway_zh", ""))).replace("\n", "<br>")
        research_value_zh = html.escape(str(ai.get("research_value_zh", ""))).replace("\n", "<br>")
        limitations_zh = html.escape(str(ai.get("limitations_zh", ""))).replace("\n", "<br>")
        final_recommendation_zh = html.escape(str(ai.get("final_recommendation_zh", ""))).replace("\n", "<br>")

        arxiv_url = html.escape(paper.get("url", ""))
        pdf_url = html.escape(paper.get("pdf", ""))

        html_parts.append(f"""
        <div style="border-top: 1px solid #ddd; padding: 18px 0;">
            <h3>[{idx}] {html.escape(badge)}</h3>
            <h2 style="font-size: 18px;">{title}</h2>

            <p>
                <b>Authors:</b> {authors}<br>
                <b>Published:</b> {published}<br>
                <b>Keyword score:</b> {paper.get("keyword_score", 0)}<br>
                <b>AI relevance score:</b> {score}<br>
                <b>Reading priority:</b> {reading_priority}<br>
                <b>Category:</b> {category}
            </p>

            <p><b>中文解读：</b><br>{summary_zh}</p>
            <p><b>为什么相关：</b><br>{why_relevant_zh}</p>
            <p><b>技术启发：</b><br>{technical_takeaway_zh}</p>
            <p><b>研究价值：</b><br>{research_value_zh}</p>
            <p><b>可能局限：</b><br>{limitations_zh}</p>
            <p><b>最终建议：</b><br>{final_recommendation_zh}</p>

            <p>
                <a href="{arxiv_url}">arXiv 页面</a> |
                <a href="{pdf_url}">PDF</a>
            </p>
        </div>
        """)

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    return "\n".join(html_parts)


def send_email(subject: str, html_content: str):
    """
    Send HTML email via Gmail SMTP.

    Required GitHub Secrets:
    EMAIL_USER, EMAIL_PASSWORD, TO_EMAIL
    """
    email_user = os.getenv("EMAIL_USER")
    email_password = os.getenv("EMAIL_PASSWORD")
    to_email = os.getenv("TO_EMAIL")

    if not email_user or not email_password or not to_email:
        raise RuntimeError(
            "Missing email settings. Please set EMAIL_USER, EMAIL_PASSWORD, and TO_EMAIL in GitHub Secrets."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = to_email

    msg.attach(MIMEText(html_content, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(email_user, email_password)
        server.sendmail(email_user, to_email, msg.as_string())


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please add it to GitHub Actions Secrets."
        )

    papers = search_arxiv(max_results=150)

    print(
        f"\nFound {len(papers)} keyword-matched papers from arXiv "
        f"within the last {DAILY_LOOKBACK_HOURS} hours."
    )

    if not papers:
        analyzed_papers = []
    else:
        analyzed_papers = analyze_papers_with_ai(
            papers=papers,
            max_ai_papers=MAX_AI_PAPERS,
        )

    print_paper_report(analyzed_papers)

    today_bj = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    useful_count = len(get_useful_papers(analyzed_papers))

    subject = f"Daily 3D Vision Paper Agent - {today_bj} - {useful_count} useful papers"
    html_content = build_email_html(analyzed_papers)
    send_email(subject, html_content)

    print("Email sent successfully.")


if __name__ == "__main__":
    main()