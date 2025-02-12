from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

import os
from src.models import img_caption, teach


IMG_DIR = os.path.abspath("imgs")
app = FastAPI()
app.mount("/imgs", StaticFiles(directory=IMG_DIR), name="imgs")
templates = Jinja2Templates(directory="templates")

images = os.listdir(IMG_DIR)

captions = {}


@app.get("/", response_class=HTMLResponse)
async def display_image(request: Request, image_index: int = 0):
    """
    첫 화면에서는 해당 인덱스의 이미지를 보여주며, 백엔드에서 캡션을 생성합니다.
    """
    if image_index >= len(images):
        return HTMLResponse("<h2>All images completed. Thank you!</h2>")

    image_url = f"/imgs/{images[image_index]}"

    # 해당 이미지에 대해 캡션 생성 (한번만 생성하도록)
    if image_index not in captions:
        captions[image_index] = img_caption(
            os.path.join(IMG_DIR, images[image_index]), max_new_tokens=50, temperature=1
        )

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image_url": image_url, "image_index": image_index},
    )


@app.post("/", response_class=HTMLResponse)
async def process_expression(
    request: Request, expression: str = Form(...), image_index: int = Form(...)
):
    """
    사용자가 영어로 이미지를 표현한 후 제출하면, Teacher LLM이 캡션과 사용자의 표현을 비교하여 피드백을 제공합니다.
    """
    if image_index >= len(images):
        return HTMLResponse("<h2>All images completed. Thank you!</h2>")

    caption = captions.get(image_index, "Caption not found")
    teacher_feedback = teach(expression, caption, max_new_tokens=100)

    next_index = image_index + 1
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "teacher_feedback": teacher_feedback,
            "caption": caption,
            "user_expression": expression,
            "next_index": next_index,
            "total_images": len(images),
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
