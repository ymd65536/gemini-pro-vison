import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.preview.language_models import TextGenerationModel

project_id = "project_id"
bucket_name = "bucket_name"
file_name = f"gs://{bucket_name}/my_email.png".format(bucket_name=bucket_name)


def generate(image: Part):
    vertexai.init(project=project_id, location="asia-northeast1")
    responses = GenerativeModel("gemini-1.0-pro-vision-001").generate_content(
        [image, """画像に含まれているテキストを抜き出してください。"""],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32
        }
    )

    return responses


if __name__ == "__main__":
    image_part = Part.from_uri(
        file_name, mime_type="image/png")

    responses = generate(image_part)
    if responses.candidates:
        response_text = responses.candidates[0].text

        check_prompt = """
質問:以下の情報に含まれてはいけないテキストはありますか？回答は回答方法に従って答えてください。
{}
---
含まれてはいけないテキスト
- メールアドレス
- 会社名
---
回答方法
「含まれている」または「含まれていない」のいずれかでお願いします。
含まれている場合は含まれているテキストを抜き出してください。
        """.format(response_text)
        generation_model = TextGenerationModel.from_pretrained(
            'text-bison@002')
        answer = generation_model.predict(
            check_prompt,
            temperature=0.2, max_output_tokens=1024,
            top_k=40, top_p=0.8).text

        print("Answer: ", answer)

    else:
        print("No response")
