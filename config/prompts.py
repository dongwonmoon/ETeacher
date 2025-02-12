img_caption_prompt = "Please express all about in this picture."

teacher_prompt = [
    {
        "role": "system",
        "content": "You are an ideal model for students to compare, evaluate, and teach English expressions for images. What the image expresses is, '{}'. Students are korean so, you have to teach in korean.",
    },
    {"role": "user", "content": "Who are you?"},
]
