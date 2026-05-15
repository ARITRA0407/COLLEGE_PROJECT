import argparse
import csv
import datetime as dt
import html
import os
import re
import time
import random
import urllib.parse
import urllib.request
import urllib.error
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "csv", "reviews.csv"))
PROJECT_ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
DOTENV_PATH = os.path.join(PROJECT_ROOT_DIR, ".env")
DUCKDUCKGO_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
DUCKDUCKGO_LITE_SEARCH_URL = "https://lite.duckduckgo.com/lite/?q={query}"
DEFAULT_MAX_SNIPPETS = 8
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_PAUSE_SECONDS = 1.2
MAX_FETCH_RETRIES = 2
PLACEHOLDER_PREFIXES = (
    "No reliable public review snippets were found online for ",
    "Could not collect online reviews for ",
)
JINA_AI_PROXY_PREFIX = "https://r.jina.ai/"
DEFAULT_PROVIDER = "ddg"
SERPER_ENDPOINT = "https://google.serper.dev/search"
SEARCHAPI_ENDPOINT = "https://www.searchapi.io/api/v1/search"


def clean_text(raw_text):
    text = html.unescape(raw_text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dotenv_if_present(dotenv_path=DOTENV_PATH):
    try:
        if not os.path.exists(dotenv_path):
            return
        with open(dotenv_path, "r", encoding="utf-8") as file:
            for line in file:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return


def extract_snippets(search_html):
    snippet_patterns = [
        r'<a[^>]*class="[^"]*snippet[^"]*"[^>]*>(.*?)</a>',
        r'<span[^>]*class="[^"]*snippet[^"]*"[^>]*>(.*?)</span>',
        r'<div[^>]*class="[^"]*snippet[^"]*"[^>]*>(.*?)</div>',
        r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
    ]
    snippets = []
    for pattern in snippet_patterns:
        matches = re.findall(pattern, search_html, flags=re.IGNORECASE | re.DOTALL)
        for match in matches:
            cleaned = clean_text(match)
            if cleaned and cleaned not in snippets and len(cleaned) >= 15:
                snippets.append(cleaned)
        if snippets:
            break
    return snippets


def normalize_college_name_for_query(college_name):
    name = (college_name or "").strip()
    if "," in name:
        name = name.split(",", 1)[0].strip()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-zA-Z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def http_get(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def _do_get(target_url):
        request = urllib.request.Request(target_url, headers=headers, method="GET")
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return response.read().decode("utf-8", errors="ignore")

    try:
        response_html = _do_get(url)
    except urllib.error.HTTPError as error:
        if int(getattr(error, "code", 0)) in (403, 429):
            proxied_url = f"{JINA_AI_PROXY_PREFIX}{url}"
            return _do_get(proxied_url)
        raise
    lower = response_html.lower()
    blocked_markers = [
        "captcha",
        "unusual traffic",
        "verify you are a human",
        "robot",
        "blocked",
    ]
    if any(marker in lower for marker in blocked_markers):
        proxied_url = f"{JINA_AI_PROXY_PREFIX}{url}"
        response_html = _do_get(proxied_url)
    return response_html


def fetch_online_review_snippets(college_name, max_snippets):
    simplified_name = normalize_college_name_for_query(college_name)
    queries = [
        f"{college_name} reviews",
        f"{college_name} student reviews",
        f"{simplified_name} reviews",
        f"{simplified_name} college reviews",
        f"{simplified_name} google reviews",
    ]
    for query in queries:
        encoded_query = urllib.parse.quote_plus(query)
        urls = [
            DUCKDUCKGO_HTML_SEARCH_URL.format(query=encoded_query),
            DUCKDUCKGO_LITE_SEARCH_URL.format(query=encoded_query),
        ]
        for url in urls:
            last_error = None
            for attempt in range(MAX_FETCH_RETRIES + 1):
                try:
                    response_html = http_get(url)
                    snippets = extract_snippets(response_html)
                    if snippets:
                        return snippets[:max_snippets]
                    break
                except Exception as error:
                    last_error = error
                    time.sleep(0.6 * (attempt + 1))
            if last_error is not None:
                continue
    return []


def fetch_serper_snippets(college_name, max_snippets, api_key, existing_text=""):
    base = normalize_college_name_for_query(college_name)
    query_options = [
        f"{base} student reviews",
        f"{base} campus life reviews",
        f"{base} placement reviews",
        f"{base} hostel mess reviews",
        f"{base} faculty reviews",
    ]
    random.shuffle(query_options)
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "User-Agent": "reviewCollection.py (college project)",
    }
    existing_norm = str(existing_text or "").strip().lower()
    snippets = []
    for query in query_options:
        for page_num in random.sample([1, 2, 3], 3):
            payload = json.dumps(
                {
                    "q": query,
                    "gl": "in",
                    "hl": "en",
                    "page": page_num,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                SERPER_ENDPOINT, data=payload, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=REQUEST_TIMEOUT_SECONDS
                ) as response:
                    data = json.loads(
                        response.read().decode("utf-8", errors="ignore") or "{}"
                    )
            except urllib.error.HTTPError as error:
                body = ""
                try:
                    body = error.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                detail = body.strip()
                if len(detail) > 500:
                    detail = detail[:500] + "..."
                if detail:
                    raise RuntimeError(f"Serper HTTP {error.code}: {detail}") from error
                raise RuntimeError(
                    f"Serper HTTP {error.code}: {error.reason}"
                ) from error
            for item in (data.get("organic") or [])[: max_snippets * 4]:
                snippet = (item.get("snippet") or "").strip()
                if not snippet:
                    continue
                snippet_norm = snippet.lower()
                if existing_norm and snippet_norm in existing_norm:
                    continue
                if snippet not in snippets:
                    snippets.append(snippet)
                if len(snippets) >= max_snippets:
                    return snippets[:max_snippets]
    return snippets[:max_snippets]


def fetch_searchapi_snippets(college_name, max_snippets, api_key):
    query = normalize_college_name_for_query(college_name)
    q = urllib.parse.quote_plus(f"{query} reviews")
    url = f"{SEARCHAPI_ENDPOINT}?engine=google&q={q}&api_key={urllib.parse.quote_plus(api_key)}"
    response_text = http_get(url)
    data = json.loads(response_text or "{}")
    snippets = []
    for item in (data.get("organic_results") or [])[: max_snippets * 2]:
        snippet = (item.get("snippet") or "").strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break
    return snippets[:max_snippets]


def fetch_debug_html(college_name):
    simplified_name = normalize_college_name_for_query(college_name)
    query = urllib.parse.quote_plus(f"{simplified_name} reviews")
    urls = [
        DUCKDUCKGO_HTML_SEARCH_URL.format(query=query),
        DUCKDUCKGO_LITE_SEARCH_URL.format(query=query),
    ]
    pages = []
    for url in urls:
        try:
            pages.append((url, http_get(url)))
        except Exception as error:
            pages.append((url, f"ERROR: {error}"))
    return pages


def build_review_text(college_name, snippets):
    if not snippets:
        return (
            f"No reliable public review snippets were found online for {college_name} "
            "during this run."
        )
    lines = [f"Collected online review snippets for {college_name}:"]
    for snippet in snippets:
        lines.append(f"- {snippet}")
    return "\n".join(lines)


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def write_rows(csv_path, fieldnames, rows):
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return csv_path
    except PermissionError:
        base, ext = os.path.splitext(csv_path)
        fallback_path = f"{base}_collected{ext or '.csv'}"
        with open(fallback_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"\nWarning: Could not write to '{csv_path}' (file is locked). "
            f"Wrote output to '{fallback_path}' instead."
        )
        return fallback_path


def ensure_required_columns(fieldnames):
    required_columns = ["college_name", "review_text", "source", "date"]
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(missing)}. "
            "Please fix the CSV header first."
        )


def is_placeholder_review_text(text):
    value = (text or "").strip()
    return any(value.startswith(prefix) for prefix in PLACEHOLDER_PREFIXES)


def collect_reviews(
    csv_path,
    max_snippets=DEFAULT_MAX_SNIPPETS,
    overwrite=True,
    overwrite_placeholders=False,
    limit=0,
    debug_save_html=False,
    provider=DEFAULT_PROVIDER,
    serper_api_key=None,
    write_empty=False,
    mode="replace",
    college_name_filter=None,
):
    fieldnames, rows = read_rows(csv_path)
    ensure_required_columns(fieldnames)
    total_rows = len(rows)
    updated_rows = 0
    today = dt.date.today().isoformat()
    if limit and limit > 0:
        rows_iter = rows[:limit]
        total_rows = len(rows_iter)
    else:
        rows_iter = rows
    for index, row in enumerate(rows_iter, start=1):
        college_name = (row.get("college_name") or "").strip()
        existing_text = (row.get("review_text") or "").strip()
        if college_name_filter:
            if college_name.lower() != str(college_name_filter).strip().lower():
                continue
        if not college_name:
            print(f"[{index}/{total_rows}] Skipped row without college_name.")
            continue
        if existing_text and not overwrite:
            if overwrite_placeholders and is_placeholder_review_text(existing_text):
                pass
            else:
                print(
                    f"[{index}/{total_rows}] Skipped (already has review_text): {college_name}"
                )
                continue
        elif existing_text and overwrite_placeholders and not overwrite:
            pass
        print(
            f"[{index}/{total_rows}] Collecting reviews for: {college_name}", flush=True
        )
        try:
            if debug_save_html and index == 1:
                debug_pages = fetch_debug_html(college_name)
                debug_path = os.path.join(BASE_DIR, "_debug_review_collection.html")
                with open(debug_path, "w", encoding="utf-8") as file:
                    for page_url, page_html in debug_pages:
                        file.write(f"===== URL: {page_url} =====\n")
                        file.write(page_html)
                        file.write("\n\n")
                print(f"  -> Debug HTML saved to: {debug_path}", flush=True)
            if provider == "serper":
                if not serper_api_key:
                    raise ValueError(
                        "Serper provider selected but no API key provided. "
                        "Set SERPER_API_KEY env var or pass --serper-api-key."
                    )
                snippets = fetch_serper_snippets(
                    college_name,
                    max_snippets=max_snippets,
                    api_key=serper_api_key,
                    existing_text=existing_text,
                )
                source_value = "Web Reviews (Serper)"
            elif provider == "searchapi":
                if not serper_api_key:
                    raise ValueError(
                        "SearchAPI provider selected but no API key provided. "
                        "Set SEARCHAPI_API_KEY env var or pass --serper-api-key."
                    )
                snippets = fetch_searchapi_snippets(
                    college_name, max_snippets=max_snippets, api_key=serper_api_key
                )
                source_value = "Web Reviews (SearchAPI.io)"
            else:
                snippets = fetch_online_review_snippets(
                    college_name, max_snippets=max_snippets
                )
                source_value = "Web Reviews (DuckDuckGo)"
            if snippets or write_empty:
                new_text = build_review_text(college_name, snippets)
                if mode == "append" and existing_text:
                    separator = "\n\n" + ("-" * 40) + "\n"
                    row["review_text"] = f"{existing_text}{separator}{new_text}"
                else:
                    row["review_text"] = new_text
                row["source"] = source_value
                row["date"] = today
                updated_rows += 1
                print(f"  -> Updated with {len(snippets)} snippet(s).", flush=True)
            else:
                print(
                    "  -> Got 0 snippets; leaving existing review_text unchanged.",
                    flush=True,
                )
        except Exception as error:
            print(f"  -> Error while collecting reviews: {error}", flush=True)
            if write_empty:
                row["review_text"] = (
                    f"Could not collect online reviews for {college_name}. "
                    f"Error: {error}"
                )
                row["source"] = (
                    "Web Reviews (Serper)"
                    if provider == "serper"
                    else "Web Reviews (DuckDuckGo)"
                )
                row["date"] = today
                updated_rows += 1
        time.sleep(REQUEST_PAUSE_SECONDS)
    if updated_rows > 0:
        saved_path = write_rows(csv_path, fieldnames, rows)
        print(f"\nDone. Updated {updated_rows} row(s) out of {total_rows}.")
        print(f"Saved updated file: {saved_path}")
    else:
        print(f"\nDone. No rows were updated; nothing was saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect online review snippets for each college_name in csv/reviews.csv "
            "and save into review_text."
        )
    )
    parser.add_argument(
        "--csv-path",
        default=CSV_FILE_PATH,
        help=f"Path to the reviews CSV (default: {CSV_FILE_PATH})",
    )
    parser.add_argument(
        "--max-snippets",
        type=int,
        default=DEFAULT_MAX_SNIPPETS,
        help=f"Maximum snippets per college (default: {DEFAULT_MAX_SNIPPETS})",
    )
    parser.add_argument(
        "--mode",
        choices=["replace", "append"],
        default="replace",
        help="How to write into review_text: replace existing text, or append to it.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite rows that already have review_text.",
    )
    parser.add_argument(
        "--overwrite-placeholders",
        action="store_true",
        help=(
            "Overwrite only placeholder/error review_text values (even if --no-overwrite "
            "is set). Useful to retry colleges that previously got 0 snippets."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N rows (0 means all).",
    )
    parser.add_argument(
        "--debug-save-html",
        action="store_true",
        help="Save fetched HTML for the first processed college into backend/_debug_review_collection.html",
    )
    parser.add_argument(
        "--provider",
        choices=["ddg", "serper", "searchapi"],
        default=DEFAULT_PROVIDER,
        help=(
            "Review source provider. 'ddg' is free but often blocked; "
            "'serper' needs a Serper API key; 'searchapi' uses SearchAPI.io key."
        ),
    )
    parser.add_argument(
        "--serper-api-key",
        default=(
            os.environ.get("SERPER_API_KEY", "")
            or os.environ.get("SEARCHAPI_API_KEY", "")
        ),
        help=(
            "API key for paid/free-tier providers. "
            "For 'serper' set SERPER_API_KEY. For 'searchapi' set SEARCHAPI_API_KEY. "
            "You can also pass it directly here."
        ),
    )
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="If enabled, writes placeholder review_text even when 0 snippets are found.",
    )
    parser.add_argument(
        "--college-name",
        default="",
        help="If set, updates only this exact college_name row.",
    )
    return parser.parse_args()


def main():
    load_dotenv_if_present()
    args = parse_args()
    csv_path = os.path.normpath(args.csv_path)
    overwrite = not args.no_overwrite
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if (
        args.provider in ("serper", "searchapi")
        and not (args.serper_api_key or "").strip()
    ):
        raise SystemExit(
            "Error: this provider requires an API key.\n"
            "PowerShell:\n"
            '  $env:SERPER_API_KEY="YOUR_KEY"   (for serper)\n'
            '  $env:SEARCHAPI_API_KEY="YOUR_KEY" (for searchapi)\n'
            "Or:\n"
            "  python backend/reviewCollection.py --provider <serper|searchapi> --serper-api-key YOUR_KEY\n"
        )
    print("--- Starting Online Review Collection ---")
    print(f"CSV path: {csv_path}")
    print(f"Overwrite existing review_text: {overwrite}")
    print(f"Max snippets per college: {args.max_snippets}\n")
    collect_reviews(
        csv_path=csv_path,
        max_snippets=max(1, args.max_snippets),
        overwrite=overwrite,
        overwrite_placeholders=args.overwrite_placeholders,
        limit=max(0, args.limit),
        debug_save_html=args.debug_save_html,
        provider=args.provider,
        serper_api_key=(args.serper_api_key or "").strip() or None,
        write_empty=args.write_empty,
        mode=args.mode,
        college_name_filter=(args.college_name or "").strip() or None,
    )


if __name__ == "__main__":
    main()
