# check_codeword_test.py
#!/usr/bin/env python3
# check_codeword_test.py
#!/usr/bin/env python3
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = os.getenv("AIPIPE_URL")

if not AIPIPE_TOKEN or not AIPIPE_URL:
    raise Exception("Set AIPIPE_TOKEN and AIPIPE_URL in environment")

system_prompt = "You are a helpful assistant."
user_prompt = "Please answer succinctly."

headers = {
    "Authorization": f"Bearer {AIPIPE_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": "openai/gpt-4.1-nano",
    "max_tokens": 500,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
}

with httpx.Client() as client:
    r = client.post(AIPIPE_URL, json=payload, headers=headers)

print("status:", r.status_code)
print("response:", r.json())







# import os
# import httpx
# from dotenv import load_dotenv

# load_dotenv()

# AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
# AIPIPE_URL = os.getenv("AIPIPE_URL")

# if not AIPIPE_TOKEN or not AIPIPE_URL:
#     raise Exception("Set AIPIPE_TOKEN and AIPIPE_URL in environment")

# system_prompt = "You are a helpful assistant."
# user_prompt = "Please answer succinctly."

# headers = {
#     "Authorization": f"Bearer {AIPIPE_TOKEN}",
#     "Content-Type": "application/json"
# }

# payload = {
#     "model": "openai/gpt-4.1-nano",
#     "max_tokens": 500,
#     "messages": [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]
# }

# with httpx.Client() as client:
#     r = client.post(AIPIPE_URL, json=payload, headers=headers)

# print("status:", r.status_code)
# print("response:", r.json())
