import arxiv


KEYWORDS = [
    "novel view synthesis",
    "single image novel view synthesis",
    "3D Gaussian Splatting",
    "Gaussian splatting",
    "3DGS",
    "feed-forward",
    "feedforward",
]


def simple_relevance_score(title: str, abstract: str) -> int:
    """
    Simple keyword-based relevance scoring.

    This is only the first-stage filter.
    Later we will add AI-based relevance judgment and Chinese interpretation.
    """
    text = f"{title} {abstract}".lower()
    score = 0

    for keyword in KEYWORDS:
        if keyword.lower() in text:
            score += 1

    # Stronger match: NVS + Gaussian representation
    if "novel view synthesis" in text and (
        "gaussian" in text or "3dgs" in text
    ):
        score += 3

    # Stronger match: single-image NVS
    if "single image" in text and "novel view synthesis" in text:
        score += 2

    # Stronger match: feed-forward NVS / Gaussian reconstruction
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

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []

    for result in client.results(search):
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
                "score": score,
            })

    papers.sort(key=lambda x: x["score"], reverse=True)
    return papers


def main():
    papers = search_arxiv(max_results=80)

    print(f"Found {len(papers)} potentially relevant papers.\n")

    for idx, paper in enumerate(papers[:10], start=1):
        print("=" * 100)
        print(f"[{idx}] Score: {paper['score']}")
        print(f"Title: {paper['title']}")
        print(f"Authors: {paper['authors']}")
        print(f"Published: {paper['published']}")
        print(f"URL: {paper['url']}")
        print(f"PDF: {paper['pdf']}")
        print()
        print("Abstract:")
        print(paper["abstract"][:1000])
        print()


if __name__ == "__main__":
    main()