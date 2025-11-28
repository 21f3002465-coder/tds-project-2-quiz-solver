# app.py
# app.py
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from solver import solve_quiz

load_dotenv()

QUIZ_SECRET = os.getenv("QUIZ_SECRET")

app = FastAPI()


@app.post("/quiz")
async def quiz_endpoint(request: Request):
    # Ensure environment secret is set
    if not QUIZ_SECRET:
        return JSONResponse(
            status_code=500,
            content={"error": "Server misconfigured: QUIZ_SECRET not set"}
        )

    # Parse incoming JSON
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})

    email = data.get("email")
    secret = data.get("secret")
    url = data.get("url")

    # Validate request fields
    if not email:
        return JSONResponse(status_code=400, content={"error": "Email missing"})
    if not secret:
        return JSONResponse(status_code=400, content={"error": "Secret missing"})
    if not url:
        return JSONResponse(status_code=400, content={"error": "URL missing"})

    # Validate provided secret against QUIZ_SECRET from env
    if secret != QUIZ_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid secret"})

    # Solve quiz chain (solver will handle submissions and chaining)
    try:
        result = await solve_quiz(email=email, secret=secret, url=url, required_min_steps=3)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Solve failed: {e}"})

    return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))






# import os
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from dotenv import load_dotenv
# import httpx
# from solver import solve_quiz

# load_dotenv()

# QUIZ_SECRET = os.getenv("QUIZ_SECRET")
# ANALYSIS_SUBMIT_URL = "https://tds-llm-analysis.s-anand.net/submit"

# app = FastAPI()


# @app.post("/quiz")
# async def quiz_endpoint(request: Request):
#     # Ensure environment secret is set
#     if not QUIZ_SECRET:
#         return JSONResponse(
#             status_code=500,
#             content={"error": "Server misconfigured: QUIZ_SECRET not set"}
#         )

#     # Parse incoming JSON
#     try:
#         data = await request.json()
#     except Exception:
#         return JSONResponse(status_code=400, content={"error": "Invalid JSON payload"})

#     email = data.get("email")
#     secret = data.get("secret")
#     url = data.get("url")

#     # Validate request fields
#     if not email:
#         return JSONResponse(status_code=400, content={"error": "Email missing"})
#     if not secret:
#         return JSONResponse(status_code=400, content={"error": "Secret missing"})
#     if not url:
#         return JSONResponse(status_code=400, content={"error": "URL missing"})

#     # Validate secret from env
#     if secret != QUIZ_SECRET:
#         return JSONResponse(status_code=403, content={"error": "Invalid secret"})

#     # Solve quiz
#     try:
#         answer = await solve_quiz(email=email, secret=secret, url=url)
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": f"Solve failed: {e}"})
#     if isinstance(answer, dict):
#         answer = answer.get("answer") or str(answer)

#     # Submit solution to analysis endpoint
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             resp = await client.post(
#                 ANALYSIS_SUBMIT_URL,
#                 json={"email": email, "secret": secret, "url": url, "answer": answer}
#             )
#         analysis_status = resp.status_code
#         analysis_text = resp.text
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Submission failed: {e}"}
#         )

#     return JSONResponse(
#         status_code=200,
#         content={
#             "result": "quiz solved",
#             "answer": answer,
#             "analysis_status": analysis_status,
#             "analysis_response": analysis_text,
#         }
#     )


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
