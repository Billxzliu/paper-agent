import os
import json
import arxiv
import smtplib
import html
import re
import urllib.request
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
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
MIN_FINAL_RELEVANCE_SCORE = 2
MAX_DAILY_PUSH_PAPERS = 5
FIGURE_CACHE_DIR = "paper_figures"
MAX_FIGURE_PAGES = 4
MIN_FIGURE_AREA = 120000
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
1 = weakly related or only shares broad keywords, do not recommend in the daily email
2 = directly related, worth reading
3 = highly related to NVS / 3DGS / 3D generation / reconstruction, should read carefully

Be strict. Do not give high scores to general segmentation, detection, classification, language model,
medical imaging, or unrelated generation papers unless they clearly connect to 3D vision, NVS,
3DGS, 3D reconstruction, 3D generation, 3D inpainting, or video generation.

Only assign score 2 or 3 when the title or abstract contains concrete technical evidence that the paper
solves, improves, evaluates, or substantially uses one of the user's core research topics. If the connection
is just a possible application or a shared buzzword, assign score 1 or 0.

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
    useful_papers = [
        p for p in papers
        if safe_int(p.get("ai", {}).get("relevance_score", 0)) >= MIN_FINAL_RELEVANCE_SCORE
    ]

    useful_papers.sort(
        key=lambda x: (
            safe_int(x.get("ai", {}).get("relevance_score", 0)),
            x.get("keyword_score", 0),
        ),
        reverse=True,
    )

    return useful_papers[:MAX_DAILY_PUSH_PAPERS]


def safe_filename(value: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return filename[:100] or "paper"


def paper_id_from_url(url: str) -> str:
    paper_id = url.rstrip("/").split("/")[-1] if url else "paper"
    return safe_filename(paper_id)


def download_pdf(pdf_url: str, pdf_path: str) -> bool:
    if not pdf_url:
        return False

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    try:
        request = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "paper-agent/1.0"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            with open(pdf_path, "wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download PDF: {pdf_url} ({e})")
        return False


def render_clip_to_png(page, clip, output_path: str):
    try:
        import fitz

        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            clip=clip,
            alpha=False,
        )

        if pixmap.width * pixmap.height < MIN_FIGURE_AREA:
            return None

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pixmap.save(output_path)
        return output_path
    except Exception:
        return None


def horizontal_overlap(rect_a, rect_b) -> float:
    return max(0, min(rect_a.x1, rect_b.x1) - max(rect_a.x0, rect_b.x0))


def rect_area(rect) -> float:
    return max(0, rect.width) * max(0, rect.height)


def rect_flag(rect, name: str) -> bool:
    value = getattr(rect, name, False)
    return bool(value() if callable(value) else value)


def find_visual_rect_above_caption(page, caption_rect, page_rect, text_blocks):
    import fitz

    candidates = []
    if caption_rect.width < page_rect.width * 0.75:
        figure_band = fitz.Rect(
            max(page_rect.x0, caption_rect.x0 - 36),
            page_rect.y0,
            min(page_rect.x1, caption_rect.x1 + 36),
            caption_rect.y0,
        )
    else:
        figure_band = fitz.Rect(
            page_rect.x0,
            page_rect.y0,
            page_rect.x1,
            caption_rect.y0,
        )

    for block in text_blocks:
        if block.get("type") != 1:
            continue

        rect = fitz.Rect(block["bbox"])
        if rect_area(rect) >= 4000:
            candidates.append(rect)

    for image in page.get_images(full=True):
        xref = image[0]
        try:
            candidates.extend(page.get_image_rects(xref))
        except Exception:
            continue

    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if not rect:
            continue

        rect = fitz.Rect(rect)
        if rect_area(rect) >= 1200 and rect.width >= 20 and rect.height >= 8:
            candidates.append(rect)

    filtered = []
    for rect in candidates:
        if rect_flag(rect, "is_empty") or rect_flag(rect, "is_infinite"):
            continue
        if rect.y1 > caption_rect.y0 + 3:
            continue
        if rect.y1 < page_rect.y0 + 24:
            continue
        if horizontal_overlap(rect, figure_band) < min(40, rect.width * 0.25):
            continue

        gap_to_caption = caption_rect.y0 - rect.y1
        if gap_to_caption > page_rect.height * 0.35:
            continue

        filtered.append(rect)

    if not filtered:
        return None

    filtered.sort(key=lambda rect: rect.y1, reverse=True)
    visual_rect = filtered[0]
    max_join_gap = 28

    for rect in filtered[1:]:
        if rect.y1 < visual_rect.y0 - max_join_gap:
            continue

        expanded_visual = fitz.Rect(
            visual_rect.x0 - 12,
            visual_rect.y0 - max_join_gap,
            visual_rect.x1 + 12,
            visual_rect.y1,
        )
        if horizontal_overlap(rect, expanded_visual) <= 0:
            continue

        visual_rect = fitz.Rect(
            min(visual_rect.x0, rect.x0),
            min(visual_rect.y0, rect.y0),
            max(visual_rect.x1, rect.x1),
            max(visual_rect.y1, rect.y1),
        )

    return visual_rect


def extract_figure_by_caption(document, output_path: str):
    import fitz

    caption_pattern = re.compile(r"\b(fig\.?|figure)\s*1\b", re.IGNORECASE)

    for page_index in range(min(len(document), MAX_FIGURE_PAGES)):
        page = document[page_index]
        page_rect = page.rect
        text_blocks = page.get_text("dict").get("blocks", [])

        for block in text_blocks:
            if block.get("type") != 0:
                continue

            block_text_parts = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text_parts.append(span.get("text", ""))

            block_text = " ".join(block_text_parts)
            if not caption_pattern.search(block_text):
                continue

            caption_rect = fitz.Rect(block["bbox"])
            if caption_rect.y0 < page_rect.height * 0.15:
                continue

            if caption_rect.width < page_rect.width * 0.75:
                x0 = max(page_rect.x0, caption_rect.x0 - 16)
                x1 = min(page_rect.x1, caption_rect.x1 + 16)
            else:
                x0 = page_rect.x0
                x1 = page_rect.x1

            visual_rect = find_visual_rect_above_caption(
                page=page,
                caption_rect=caption_rect,
                page_rect=page_rect,
                text_blocks=text_blocks,
            )

            if visual_rect:
                x0 = max(page_rect.x0, min(x0, visual_rect.x0) - 10)
                x1 = min(page_rect.x1, max(x1, visual_rect.x1) + 10)
                y0 = max(page_rect.y0, visual_rect.y0 - 10)
            else:
                y0 = max(page_rect.y0, caption_rect.y0 - min(page_rect.height * 0.32, 260))

            y1 = min(page_rect.y1, caption_rect.y1 + 12)

            if y1 - y0 < 80 or x1 - x0 < 120:
                continue

            extracted = render_clip_to_png(
                page=page,
                clip=fitz.Rect(x0, y0, x1, y1),
                output_path=output_path,
            )
            if extracted:
                return extracted

    return None


def extract_largest_pdf_image(document, output_path: str):
    import fitz

    best = None
    seen_xrefs = set()

    for page_index in range(min(len(document), MAX_FIGURE_PAGES)):
        page = document[page_index]

        for image in page.get_images(full=True):
            xref = image[0]
            if xref in seen_xrefs:
                continue

            seen_xrefs.add(xref)

            try:
                pixmap = fitz.Pixmap(document, xref)
                width = pixmap.width
                height = pixmap.height
                area = width * height
            except Exception:
                continue

            if width < 250 or height < 120 or area < MIN_FIGURE_AREA:
                continue

            score = area - page_index * 50000
            if best is None or score > best["score"]:
                best = {"xref": xref, "score": score}

    if not best:
        return None

    try:
        pixmap = fitz.Pixmap(document, best["xref"])
        if pixmap.n >= 5:
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pixmap.save(output_path)
        return output_path
    except Exception:
        return None


def extract_overview_figure(paper):
    paper_id = paper_id_from_url(paper.get("url", ""))
    pdf_path = os.path.join(FIGURE_CACHE_DIR, "pdfs", f"{paper_id}.pdf")
    image_path = os.path.join(FIGURE_CACHE_DIR, "images", f"{paper_id}.png")

    if os.path.exists(image_path):
        return image_path

    if not os.path.exists(pdf_path) and not download_pdf(paper.get("pdf", ""), pdf_path):
        return None

    try:
        import fitz
    except ImportError:
        print("PyMuPDF is not installed; skip overview figure extraction.")
        return None

    try:
        with fitz.open(pdf_path) as document:
            return (
                extract_figure_by_caption(document, image_path)
                or extract_largest_pdf_image(document, image_path)
            )
    except Exception as e:
        print(f"Failed to extract overview figure for {paper.get('title', '')}: {e}")
        return None


def add_overview_figures(papers):
    for idx, paper in enumerate(get_useful_papers(papers), start=1):
        print(f"Extracting overview figure {idx}: {paper.get('title', '')}")

        figure_path = extract_overview_figure(paper)
        if not figure_path:
            print("No overview figure extracted; continue without image.")
            continue

        paper_id = paper_id_from_url(paper.get("url", ""))
        paper["figure_path"] = figure_path
        paper["figure_cid"] = f"overview-{paper_id}@paper-agent"


def collect_inline_images(papers):
    inline_images = []

    for paper in get_useful_papers(papers):
        figure_path = paper.get("figure_path")
        figure_cid = paper.get("figure_cid")

        if figure_path and figure_cid and os.path.exists(figure_path):
            inline_images.append({
                "path": figure_path,
                "cid": figure_cid,
            })

    return inline_images


def print_paper_report(papers):
    """
    Print final report to GitHub Actions logs.
    """
    useful_papers = get_useful_papers(papers)

    print("\n" + "=" * 100)
    print("Daily 3D Vision / NVS / 3DGS Paper Agent Report")
    print("=" * 100)
    print(f"Total AI-analyzed papers: {len(papers)}")
    print(
        f"Recommended papers with relevance_score >= {MIN_FINAL_RELEVANCE_SCORE}: "
        f"{len(useful_papers)} / max {MAX_DAILY_PUSH_PAPERS}"
    )
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
        (
            f"<b>Recommended papers with relevance_score >= {MIN_FINAL_RELEVANCE_SCORE}:</b> "
            f"{len(useful_papers)} / max {MAX_DAILY_PUSH_PAPERS}<br>"
        ),
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
        figure_html = ""

        if paper.get("figure_cid"):
            figure_cid = html.escape(str(paper["figure_cid"]))
            figure_html = f"""
            <p>
                <b>Overview figure:</b><br>
                <img src="cid:{figure_cid}" alt="Overview figure for {title}"
                     style="max-width: 100%; height: auto; border: 1px solid #ddd;">
            </p>
            """

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

            {figure_html}

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


def send_email(subject: str, html_content: str, inline_images=None):
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

    inline_images = inline_images or []

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = email_user
    msg["To"] = to_email

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(html_content, "html", "utf-8"))
    msg.attach(alternative)

    for image in inline_images:
        try:
            with open(image["path"], "rb") as file:
                mime_image = MIMEImage(file.read())

            mime_image.add_header("Content-ID", f"<{image['cid']}>")
            mime_image.add_header(
                "Content-Disposition",
                "inline",
                filename=os.path.basename(image["path"]),
            )
            msg.attach(mime_image)
        except Exception as e:
            print(f"Failed to attach inline image {image.get('path', '')}: {e}")

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
    add_overview_figures(analyzed_papers)

    today_bj = datetime.now(TIMEZONE).strftime("%Y-%m-%d")
    useful_count = len(get_useful_papers(analyzed_papers))

    subject = f"Daily 3D Vision Paper Agent - {today_bj} - {useful_count} useful papers"
    html_content = build_email_html(analyzed_papers)
    inline_images = collect_inline_images(analyzed_papers)
    send_email(subject, html_content, inline_images=inline_images)

    print("Email sent successfully.")


if __name__ == "__main__":
    main()
