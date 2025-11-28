# post.py
import httpx

payload = {
    "email": "21f3002465@ds.study.iitm.ac.in",
    "secret": "myquizsecret",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

resp = httpx.post("http://localhost:8000/quiz", json=payload, timeout=60.0)
print("status_code:", resp.status_code)
print("response:", resp.json())
