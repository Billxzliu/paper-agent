import os
import json
import arxiv
from openai import OpenAI


KEYWORDS = [
    "novel view synthesis",
    "single image novel view synthesis",
    "3D Gaussian Splatting",
    "Gaussian splatting",
    "3DGS",
    "feed-forward",
    "feedforward",
]

DEEPSEEK_MODEL = "deepseek-v4-flash"

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


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

    return score


def search_arxiv(max_results: int = 80):
    """
    Search recent arXiv papers from cs.CV using focused NVS / 3DGS keywords.
    """
    query = (
        'cat:cs.CV AND ('
        '"novel view synthesis" OR '
        '"single image novel view synthesis" OR '
        '"3D Gaussian Splatting" OR '
        '"Gaussian splatting" OR '
        '"3DGS" OR '
        '"feed-forward" OR '
        '"feedforward"'
        ')'
    )

    client_arxiv = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []

    for result in client_arxiv.results(search):
        title = result.title.replace("\n", " ").strip()
        abstract = result.summary.replace("\n", " ").strip()
        score = simple_relevance_score(title, abstract)

        if score > 0:
            papers.append({
                "title": title,
                "authors": ", ".join(author.name for author in result.authors[:6]),
                "published": result.published.strftime("%Y-%m-%d"),
                "abstract": abstract,
                "url": result.entry_id,
                "pdf": result.pdf_url,
                "keyword_score": score,
            })

    papers.sort(key=lambda x: x["keyword_score"], reverse=True)
    return papers


def ai_analyze_paper(title: str, abstract: str) -> dict:
    """
    Use DeepSeek V4 to judge relevance and generate Chinese interpretation.
    """
    system_prompt = """
You are an expert research assistant in computer vision and 3D reconstruction.

The user's research direction is:
Single-image novel view synthesis (NVS), feed-forward 3D Gaussian Splatting (3DGS),
large-view-deviation NVS, efficient feed-forward 3D reconstruction, and 3D Gaussian scene rendering.

The user is preparing a NeurIPS-style paper named Spackle.
The main idea is to mitigate capacity competition in feed-forward 3DGS by freezing
a baseline fixed-budget 3DGS model and learning a residual 3DGS branch only for poorly reconstructed
or disoccluded regions.

Your task:
Given a paper title and abstract, decide whether this paper is relevant to the user's research.
Return your answer in Chinese.

Use this relevance scale:
0 = unrelated, ignore
1 = weakly related, maybe scan
2 = related, worth reading
3 = highly related, should read carefully

Be strict. Do not give high scores to general 3D, segmentation, detection, language model, or unrelated generation papers.

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
  "category": "single-image NVS / feed-forward 3DGS / 3DGS / NVS / weakly related / unrelated",
  "summary_zh": "",
  "why_relevant_zh": "",
  "technical_takeaway_zh": "",
  "possible_use_for_spackle_zh": "",
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
                "possible_use_for_spackle_zh": "",
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
            "possible_use_for_spackle_zh": "",
            "limitations_zh": "",
            "final_recommendation_zh": "",
        }


def analyze_papers_with_ai(papers, max_ai_papers: int = 15):
    """
    Analyze only top keyword-matched papers to control API cost.
    """
    analyzed = []

    for idx, paper in enumerate(papers[:max_ai_papers], start=1):
        print(f"\nAnalyzing paper {idx}/{min(len(papers), max_ai_papers)}: {paper['title']}")
        ai_result = ai_analyze_paper(
            title=paper["title"],
            abstract=paper["abstract"],
        )
        paper["ai"] = ai_result
        analyzed.append(paper)

    analyzed.sort(
        key=lambda x: (
            int(x["ai"].get("relevance_score", 0)),
            x["keyword_score"],
        ),
        reverse=True,
    )

    return analyzed


def print_paper_report(papers):
    """
    Print final report to GitHub Actions logs.
    """
    useful_papers = [
        p for p in papers
        if int(p.get("ai", {}).get("relevance_score", 0)) >= 1
    ]

    print("\n" + "=" * 100)
    print("Daily NVS / 3DGS Paper Agent Report")
    print("=" * 100)
    print(f"Total AI-analyzed papers: {len(papers)}")
    print(f"Useful papers with relevance_score >= 1: {len(useful_papers)}")
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
        print(f"Published: {paper['published']}")
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

        print("\n对 Spackle 的潜在价值:")
        print(ai.get("possible_use_for_spackle_zh", ""))

        print("\n可能局限:")
        print(ai.get("limitations_zh", ""))

        print("\n最终建议:")
        print(ai.get("final_recommendation_zh", ""))


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please add it to GitHub Actions Secrets."
        )

    papers = search_arxiv(max_results=80)

    print(f"Found {len(papers)} keyword-matched papers from arXiv.")

    if not papers:
        print("No candidate papers found.")
        return

    analyzed_papers = analyze_papers_with_ai(
        papers=papers,
        max_ai_papers=15,
    )

    print_paper_report(analyzed_papers)


if __name__ == "__main__":
    main()