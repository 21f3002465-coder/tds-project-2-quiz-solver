# solver.py
# solver.py  — FINAL UPDATED VERSION
import os
import io
import re
import json
import asyncio
import base64
from typing import Optional, Any, Dict, List
from urllib.parse import urljoin

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# ENV CONFIG
# ---------------------------

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = os.getenv("AIPIPE_URL")  # full inference endpoint
ANALYSIS_SUBMIT_URL = os.getenv(
    "ANALYSIS_SUBMIT_URL",
    "https://tds-llm-analysis.s-anand.net/submit"
)

DEFAULT_HTTP_TIMEOUT = 30.0
MAX_CHAIN_STEPS = int(os.getenv("MAX_CHAIN_STEPS", "10"))


# ---------------------------
# NORMALIZATION HELPERS
# ---------------------------

def _attempt_json_parse(value: str):
    try:
        return json.loads(value)
    except Exception:
        return None

def _attempt_number(value: str):
    try:
        if re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
        if re.fullmatch(r"-?\d+\.\d+", value.strip()):
            return float(value.strip())
    except Exception:
        return None
    return None

def _attempt_boolean(value: str):
    v = value.strip().lower()
    if v in ["true", "yes", "y"]:
        return True
    if v in ["false", "no", "n"]:
        return False
    return None

def _flatten_answer_object(obj: dict):
    # If solver/LLM returned {"answer": "...", "explanation": "..."}
    if "answer" in obj:
        v = obj["answer"]
        if isinstance(v, (str, int, float, bool)):
            return v
        try:
            return json.dumps(v)
        except Exception:
            return str(v)

    # General flattening
    cleaned = {}
    for k, v in obj.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned

def _normalize_answer_for_submission(answer: Any) -> Any:
    """
    Produce ONLY one of the types allowed by the grader:
    - boolean
    - number
    - string
    - dict (simple keys only)
    """
    if isinstance(answer, (int, float, bool)):
        return answer

    if isinstance(answer, dict):
        return _flatten_answer_object(answer)

    if isinstance(answer, str):
        s = answer.strip()

        # 1) boolean
        b = _attempt_boolean(s)
        if b is not None:
            return b

        # 2) number
        n = _attempt_number(s)
        if n is not None:
            return n

        # 3) JSON
        j = _attempt_json_parse(s)
        if j is not None:
            if isinstance(j, dict):
                return _flatten_answer_object(j)
            return j

        # 4) base64 URI
        if s.startswith("data:") and ";base64," in s:
            return s

        # 5) fallback simple string
        return s

    # fallback convert to string
    try:
        return str(answer)
    except Exception:
        return f"{answer}"


# ---------------------------
# HTTP & SCRAPING HELPERS
# ---------------------------

async def _fetch_text(url: str, client: httpx.AsyncClient, timeout: float = DEFAULT_HTTP_TIMEOUT) -> str:
    resp = await client.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n")
    text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
    return text

async def _download_and_process_file(file_url: str, client: httpx.AsyncClient, context_text: str) -> Any:
    resp = await client.get(file_url, timeout=DEFAULT_HTTP_TIMEOUT)
    resp.raise_for_status()
    content = resp.content

    if file_url.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(content.decode()))
    elif file_url.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        raise Exception("Unsupported file type for heuristic")

    # If page hints "sum", apply heuristic
    if "sum" in (context_text or "").lower():
        num_cols = df.select_dtypes("number").columns
        if len(num_cols) > 0:
            return int(df[num_cols[0]].sum())

    return df.to_dict("records")

def _extract_secret_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    # tds{...}
    m = re.search(r"(tds\{[^}]+\})", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # 6+ alphanumeric
    m = re.search(r"\b([A-Za-z0-9_\-]{6,})\b", text)
    if m:
        return m.group(1)

    # "secret: XXXX"
    m = re.search(r"secret[:\s]*([A-Za-z0-9_\-{}]{4,})", text, re.IGNORECASE)
    if m:
        return m.group(1)

    return None


# ---------------------------
# AIPIPE CALL
# ---------------------------

async def _call_aipipe(messages: List[Dict[str, str]]) -> str:
    if not AIPIPE_TOKEN or not AIPIPE_URL:
        raise Exception("AIPIPE_TOKEN or AIPIPE_URL not set")

    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4.1-nano",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1000
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(AIPIPE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0]["message"]["content"]


async def _solve_page_with_llm(page_url: str, page_text: str) -> Any:
    system_prompt = "You are a helpful assistant that outputs a JSON with 'answer' and 'explanation'."
    user_prompt = f"""
URL: {page_url}

Page content:
{page_text[:15000]}

Return ONLY JSON:
{{
  "answer": "<value>",
  "explanation": "<short reasoning>"
}}
"""

    assistant_output = await _call_aipipe([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    assistant_output = assistant_output.strip()

    # try parse JSON
    try:
        return json.loads(assistant_output)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", assistant_output)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return assistant_output


# ---------------------------
# SUBMISSION TO TDS GRADER
# ---------------------------

async def _submit_to_analysis(email: str, secret: str, url: str, answer: Any, client: httpx.AsyncClient) -> Dict[str, Any]:
    final_ans = _normalize_answer_for_submission(answer)

    payload = {
        "email": email,
        "secret": secret,
        "url": url,
        "answer": final_ans
    }

    resp = await client.post(ANALYSIS_SUBMIT_URL, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)

    try:
        parsed = resp.json()
    except Exception:
        parsed = {"raw": resp.text}

    return {
        "status_code": resp.status_code,
        "parsed": parsed,
        "raw_text": resp.text,
        "submitted_answer": final_ans
    }


# ---------------------------
# MAIN MULTI-STEP CHAIN SOLVER
# ---------------------------

async def solve_quiz(email: str, secret: str, url: str, required_min_steps: int = 3, max_steps: int = MAX_CHAIN_STEPS) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    current_url = url
    step_no = 0

    async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
        while current_url and step_no < max_steps:
            step_no += 1
            record = {"step": step_no, "url": current_url}

            try:
                # 1) Fetch page text
                page_text = await _fetch_text(current_url, client)

                # 2) Detect downloadable file links
                resp2 = await client.get(current_url)
                soup = BeautifulSoup(resp2.text, "html.parser")

                file_link = None
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if href.lower().endswith((".csv", ".xls", ".xlsx")):
                        file_link = href if href.startswith("http") else urljoin(current_url, href)
                        break

                # 3) Candidate answer
                answer = None

                # 3A) If file exists → heuristic
                if file_link:
                    try:
                        record["method"] = "file_heuristic"
                        answer = await _download_and_process_file(file_link, client, page_text)
                    except Exception as e:
                        record["file_error"] = str(e)
                        answer = None

                # 3B) If no answer → try secret extraction
                if answer is None:
                    extracted = _extract_secret_from_text(page_text)
                    if extracted:
                        answer = extracted
                        record["method"] = "extracted_secret"

                # 3C) Otherwise → LLM
                if answer is None:
                    record["method"] = "aipipe"
                    answer = await _solve_page_with_llm(current_url, page_text)

                record["raw_answer"] = answer

                # 4) Submit answer
                submit_res = await _submit_to_analysis(
                    email=email,
                    secret=secret,
                    url=current_url,
                    answer=answer,
                    client=client
                )

                record["submit_status"] = submit_res["status_code"]
                record["submitted_answer"] = submit_res["submitted_answer"]
                record["submit_parsed"] = submit_res["parsed"]

                # 5) Next URL?
                parsed = submit_res["parsed"]
                next_url = None

                if isinstance(parsed, dict):
                    next_url = parsed.get("url")

                    # If nested JSON in a string
                    if isinstance(next_url, str) and next_url.startswith("{"):
                        try:
                            nested = json.loads(next_url)
                            next_url = nested.get("url")
                        except Exception:
                            pass

                record["next_url"] = next_url
                steps.append(record)

                if not next_url:
                    break

                current_url = next_url
                await asyncio.sleep(0.2)

            except Exception as e:
                record["error"] = str(e)
                steps.append(record)
                break

        total = len(steps)
        success = all(s.get("submit_status", 500) < 400 for s in steps)

        return {
            "result": "chain_completed" if success else "chain_with_errors",
            "steps_done": total,
            "meets_required_steps": total >= required_min_steps,
            "steps": steps
        }






# # solver.py
# import os
# import io
# import re
# import json
# import asyncio
# from typing import Optional, Any, Dict, List
# from urllib.parse import urljoin

# import httpx
# import pandas as pd
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv

# load_dotenv()

# # Aipipe config
# AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
# AIPIPE_URL = os.getenv("AIPIPE_URL")  # full model endpoint to call
# ANALYSIS_SUBMIT_URL = os.getenv(
#     "ANALYSIS_SUBMIT_URL",
#     "https://tds-llm-analysis.s-anand.net/submit"
# )

# # timeouts & limits
# DEFAULT_HTTP_TIMEOUT = 30.0
# MAX_CHAIN_STEPS = int(os.getenv("MAX_CHAIN_STEPS", "10"))


# async def _fetch_text(url: str, client: httpx.AsyncClient, timeout: float = DEFAULT_HTTP_TIMEOUT) -> str:
#     """Fetch URL and return visible text extracted via BeautifulSoup."""
#     resp = await client.get(url, timeout=timeout)
#     resp.raise_for_status()
#     html = resp.text
#     soup = BeautifulSoup(html, "html.parser")
#     text = soup.get_text("\n")
#     text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
#     return text


# async def _download_and_process_file(file_url: str, client: httpx.AsyncClient, context_text: str) -> Any:
#     """
#     Download CSV/XLS/XLSX and apply simple heuristics (e.g., sum numeric column).
#     Returns either a numeric answer or list-of-records.
#     """
#     resp = await client.get(file_url, timeout=DEFAULT_HTTP_TIMEOUT)
#     resp.raise_for_status()
#     content = resp.content

#     if file_url.lower().endswith(".csv"):
#         df = pd.read_csv(io.StringIO(content.decode()))
#     elif file_url.lower().endswith((".xls", ".xlsx")):
#         df = pd.read_excel(io.BytesIO(content))
#     else:
#         raise Exception("Unsupported file type")

#     # Heuristic: if page asks for sum
#     if "sum" in (context_text or "").lower():
#         num_cols = df.select_dtypes(include="number").columns
#         if len(num_cols) > 0:
#             return int(df[num_cols[0]].sum())

#     return df.to_dict("records")


# def _extract_secret_from_text(text: str) -> Optional[str]:
#     """
#     Heuristic extraction of a secret code from free text.
#     - Prefer patterns like tds{...} or sequences of >=6 alphanum/{}_-
#     - Also look in <code> or <pre> blocks (handled elsewhere if needed)
#     """
#     if not text:
#         return None

#     # 1) tds{...} style
#     m = re.search(r"(tds\{[^}]+\})", text, re.IGNORECASE)
#     if m:
#         return m.group(1)

#     # 2) long alphanumeric tokens (6+)
#     m = re.search(r"\b([A-Za-z0-9_\-]{6,})\b", text)
#     if m:
#         return m.group(1)

#     # 3) any line that contains the word 'secret' and then a token
#     m = re.search(r"secret[:\s]*([A-Za-z0-9_\-{}]{4,})", text, re.IGNORECASE)
#     if m:
#         return m.group(1)

#     return None


# async def _call_aipipe(messages: List[Dict[str, str]]) -> str:
#     """Call AIPIPE endpoint and return assistant content as string."""
#     if not AIPIPE_TOKEN or not AIPIPE_URL:
#         raise Exception("AIPIPE_TOKEN or AIPIPE_URL not configured in environment")

#     headers = {
#         "Authorization": f"Bearer {AIPIPE_TOKEN}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "openai/gpt-4.1-nano",
#         "messages": messages,
#         "temperature": 0,
#         "max_tokens": 1000
#     }

#     async with httpx.AsyncClient(timeout=60.0) as client:
#         resp = await client.post(AIPIPE_URL, json=payload, headers=headers)
#         resp.raise_for_status()
#         data = resp.json()
#         # safe navigation
#         choices = data.get("choices", [])
#         if not choices:
#             return ""
#         return choices[0].get("message", {}).get("content", "")


# def _normalize_answer_for_submission(answer: Any) -> str:
#     """
#     Ensure the 'answer' field sent to the grading server is a plain string,
#     since their API rejects objects.
#     """
#     if isinstance(answer, str):
#         return answer
#     if isinstance(answer, (int, float)):
#         return str(answer)
#     if isinstance(answer, dict):
#         # common case: {"answer": "X", "explanation": "..."}
#         return str(answer.get("answer") or json.dumps(answer))
#     # fallback: convert to JSON string
#     try:
#         return json.dumps(answer)
#     except Exception:
#         return str(answer)


# async def _submit_to_analysis(email: str, secret: str, url: str, answer: Any, client: httpx.AsyncClient) -> Dict[str, Any]:
#     """
#     Submit to analysis submit endpoint and return parsed JSON (or raw text in 'raw').
#     """
#     payload = {
#         "email": email,
#         "secret": secret,
#         "url": url,
#         "answer": _normalize_answer_for_submission(answer)
#     }
#     resp = await client.post(ANALYSIS_SUBMIT_URL, json=payload, timeout=DEFAULT_HTTP_TIMEOUT)
#     # Try to parse JSON; if not JSON return text
#     text = resp.text
#     parsed = None
#     try:
#         parsed = resp.json()
#     except Exception:
#         parsed = {"raw": text}
#     return {"status_code": resp.status_code, "parsed": parsed, "raw_text": text}


# async def _solve_page_with_llm(page_url: str, page_text: str) -> Any:
#     """
#     Prepare a prompt for the LLM that asks for an answer in JSON form.
#     The returned value may be JSON or free text.
#     """
#     system_prompt = "You are a helpful assistant that reads a webpage and returns a concise answer in JSON."
#     user_prompt = f"""URL: {page_url}

# Page content (trimmed):
# {page_text[:15000]}

# Return ONLY JSON like:
# {{ "answer": "...", "explanation": "..." }}
# If you cannot find an explicit answer, provide a short textual judgment in "answer".
# """
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
#     assistant_text = await _call_aipipe(messages)
#     assistant_text = (assistant_text or "").strip()

#     # try parse JSON
#     try:
#         return json.loads(assistant_text)
#     except Exception:
#         # extract JSON substring if possible
#         m = re.search(r"(\{[\s\S]*\})", assistant_text)
#         if m:
#             try:
#                 return json.loads(m.group(1))
#             except Exception:
#                 pass
#         return assistant_text


# async def solve_quiz(email: str, secret: str, url: str, required_min_steps: int = 3, max_steps: int = MAX_CHAIN_STEPS) -> Dict[str, Any]:
#     """
#     Full orchestration:
#       - fetch initial url, attempt to solve (files heuristic or LLM)
#       - submit to analysis endpoint
#       - if analysis response includes a next 'url', follow it and repeat
#       - if next page is a 'scrape' page which contains a secret code, extract that secret and submit as answer
#       - repeat until no further url or until max_steps
#     Returns a summary object describing each step.
#     """
#     if max_steps <= 0:
#         raise Exception("max_steps must be >= 1")

#     steps: List[Dict[str, Any]] = []
#     current_url = url
#     step_no = 0

#     async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
#         while current_url and step_no < max_steps:
#             step_no += 1
#             step_record: Dict[str, Any] = {"step": step_no, "url": current_url}
#             try:
#                 # 1) Fetch page text
#                 page_text = await _fetch_text(current_url, client)

#                 # 2) Detect data file links
#                 resp2 = await client.get(current_url)
#                 soup = BeautifulSoup(resp2.text, "html.parser")


#                 # soup = BeautifulSoup(await client.get(current_url).text, "html.parser")
#                 file_link = None
#                 for a in soup.find_all("a", href=True):
#                     href = a["href"].strip()
#                     if href.lower().endswith((".csv", ".xls", ".xlsx")):
#                         file_link = href if href.startswith("http") else urljoin(current_url, href)
#                         break

#                 # 3) If data file exists -> try heuristic processing
#                 answer: Any = None
#                 if file_link:
#                     try:
#                         answer = await _download_and_process_file(file_link, client, page_text)
#                         step_record["method"] = "file_heuristic"
#                     except Exception as ex_file:
#                         # On failure fallback to LLM
#                         step_record["file_error"] = str(ex_file)
#                         answer = None

#                 # 4) If no answer yet -> try to extract secret from page text directly
#                 if answer is None:
#                     secret_candidate = _extract_secret_from_text(page_text)
#                     if secret_candidate:
#                         answer = secret_candidate
#                         step_record["method"] = "extracted_secret"
#                     else:
#                         # 5) Else ask AIPipe to solve this page
#                         step_record["method"] = "aipipe"
#                         answer = await _solve_page_with_llm(current_url, page_text)

#                 # 6) Normalize answer for record but keep original for submission conversion later
#                 step_record["raw_answer"] = answer

#                 # 7) Submit to analysis endpoint
#                 submit_result = await _submit_to_analysis(email=email, secret=secret, url=current_url, answer=answer, client=client)
#                 step_record["submit_status"] = submit_result["status_code"]
#                 step_record["submit_parsed"] = submit_result["parsed"]

#                 # 8) Parse next url if provided in parsed JSON
#                 parsed = submit_result["parsed"]
#                 next_url = None
#                 if isinstance(parsed, dict):
#                     # Common key names: 'url'
#                     next_url = parsed.get("url")
#                     # In some cases the server returns nested JSON as string; handle that
#                     if isinstance(next_url, str) and next_url.startswith("{"):
#                         try:
#                             nested = json.loads(next_url)
#                             next_url = nested.get("url")
#                         except Exception:
#                             pass

#                 step_record["next_url"] = next_url
#                 steps.append(step_record)

#                 # Decide whether to continue
#                 if not next_url:
#                     # No further quizzes; stop
#                     break

#                 # Prepare to follow next_url:
#                 # The next URL may instruct you to fetch another "scrape" page.
#                 # For the next loop we will set current_url = next_url and let the loop handle extraction/submission.
#                 current_url = next_url

#                 # small delay to be polite
#                 await asyncio.sleep(0.2)

#             except Exception as e:
#                 # record error and break or continue depending
#                 step_record["error"] = str(e)
#                 steps.append(step_record)
#                 # stop on unexpected exception
#                 break

#         # After loop, check if required_min_steps satisfied
#         total_steps_done = len(steps)
#         success = all(s.get("submit_status", 500) < 400 for s in steps) and total_steps_done >= 1
#         meets_required = total_steps_done >= required_min_steps

#         result = {
#             "result": "chain_completed" if not any(s.get("error") for s in steps) else "chain_with_errors",
#             "steps_done": total_steps_done,
#             "meets_required_steps": meets_required,
#             "steps": steps
#         }

#         return result







# import os
# import io
# import json
# import re
# import requests
# import pandas as pd
# import httpx
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv

# load_dotenv()

# AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
# AIPIPE_URL = os.getenv("AIPIPE_URL")   # Must be a full model endpoint URL


# def _download_and_process_file(file_url, context_text):
#     """Downloads CSV / XLS / XLSX and computes basic answers if possible."""
#     r = requests.get(file_url, timeout=30)
#     if r.status_code != 200:
#         raise Exception(f"Failed to download: {file_url}")

#     content = r.content

#     # handle CSV
#     if file_url.endswith(".csv"):
#         df = pd.read_csv(io.StringIO(content.decode()))

#     # handle Excel
#     elif file_url.endswith((".xls", ".xlsx")):
#         df = pd.read_excel(io.BytesIO(content))

#     else:
#         raise Exception("Unsupported file type")

#     # Heuristic: if quiz asks to sum values
#     if "sum" in context_text.lower():
#         num_cols = df.select_dtypes(include="number").columns
#         if len(num_cols) > 0:
#             return int(df[num_cols[0]].sum())

#     return df.to_dict("records")


# async def _call_aipipe(messages):
#     """Call AIPIPE LLM to solve the quiz."""
#     if not AIPIPE_TOKEN or not AIPIPE_URL:
#         raise Exception("AIPIPE_TOKEN or AIPIPE_URL missing")

#     headers = {
#         "Authorization": f"Bearer {AIPIPE_TOKEN}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "openai/gpt-4.1-nano",
#         "messages": messages,
#         "temperature": 0,
#         "max_tokens": 1000
#     }

#     async with httpx.AsyncClient(timeout=60) as client:
#         r = await client.post(AIPIPE_URL, json=payload, headers=headers)
#         data = r.json()
#         return data["choices"][0]["message"]["content"]


# async def solve_quiz(email, secret, url):
#     """Loads page with requests + BeautifulSoup, solves either by CSV/XLS or AIPIPE LLM."""
    
#     # 1. Load webpage
#     r = requests.get(url, timeout=30)
#     if r.status_code != 200:
#         raise Exception(f"Failed to load quiz URL: {r.status_code}")

#     html = r.text
#     soup = BeautifulSoup(html, "html.parser")

#     # Extract visible text
#     text = soup.get_text("\n")
#     text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()

#     # 2. Find downloadable data files
#     file_link = None
#     for a in soup.find_all("a", href=True):
#         href = a["href"]
#         if href.lower().endswith((".csv", ".xls", ".xlsx")):
#             if href.startswith("http"):
#                 file_link = href
#             else:
#                 from urllib.parse import urljoin
#                 file_link = urljoin(url, href)
#             break

#     # 3. If a data file exists → process heuristically
#     if file_link:
#         try:
#             return _download_and_process_file(file_link, text)
#         except Exception:
#             pass  # fallback → ask AIPIPE

#     # 4. Ask AIPIPE to solve
#     user_prompt = f"""
# Solve the quiz contained in this webpage.

# URL: {url}

# Extracted Page Text:
# {text[:15000]}

# Return ONLY JSON:
# {{
#   "answer": "...",
#   "explanation": "..."
# }}
# """

#     messages = [
#         {"role": "system", "content": "You solve quizzes accurately."},
#         {"role": "user", "content": user_prompt}
#     ]

#     response_text = await _call_aipipe(messages)

#     # Try extracting JSON
#     try:
#         return json.loads(response_text)
#     except:
#         # Try to extract JSON substring
#         match = re.search(r"\{[\s\S]*\}", response_text)
#         if match:
#             try:
#                 return json.loads(match.group(0))
#             except:
#                 pass
#         return response_text  # fallback
