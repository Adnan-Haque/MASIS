import base64
import json
from openai import OpenAI

client = OpenAI()

VISION_MODEL = "gpt-4o-mini"


def process_image_with_vision(image_bytes: bytes):

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4o-mini",  # cheaper + vision capable
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Analyze this image and describe any financial charts, tables, or dashboards present."
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ]
    )

    summary = response.output_text

    structured = {
        "type": "image_analysis",
        "summary": summary
    }

    return summary, structured


def build_summary_text(data):
    if data.get("type") == "dashboard":
        parts = ["Dashboard Summary:"]
        for m in data.get("metrics", []):
            parts.append(
                f"{m.get('name')}: {m.get('value')} {m.get('unit')} "
                f"(trend: {m.get('trend')})"
            )

        if data.get("time_period"):
            parts.append(f"Period: {data['time_period']}")

        if data.get("insights"):
            parts.append(f"Insights: {data['insights']}")

        return " | ".join(parts)

    return data.get("text", "")