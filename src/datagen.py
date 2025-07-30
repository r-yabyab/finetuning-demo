import json
from typing import List 
from pydantic import BaseModel
from litellm import completion
from generated_prompt import prompt_template

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

import os
os.environ["LITELLM_OLLAMA_API_BASE"] = "http://localhost:11434"

def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama/deepseek-r1",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(delta, end="") 
            data += delta 
    return json.loads(data)


from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

converter = DocumentConverter()
# doc = "data/docling-stuff/sigewinne.md"
doc = converter.convert("data/docling-stuff/sigewinne.md").document
chunker = HybridChunker()
chunks = chunker.chunk(dl_doc=doc)
# result = converter.convert(doc)
# print(result.document.export_to_markdown())


# chunk_iter = chunker.chunk(result.document)

dataset = {}
for i, chunk in enumerate(chunks):
    print(f"=== {i} ===")
    print(f"chunk.text:\n{f'{chunk.text[:300]}…'!r}")

    enriched_text = chunker.contextualize(chunk=chunk)
    print(f"chunker.contextualize(chunk):\n{f'{enriched_text[:300]}…'!r}")

    data = llm_call(
        enriched_text
    )
    dataset[i] = {"generated": data["generated"], "context": enriched_text}

with open("tm1data.jsonl", "w") as f:
    json.dump(dataset, f)