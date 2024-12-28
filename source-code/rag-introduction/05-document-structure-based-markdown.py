from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_text = """
# Title
## Section 1
Content of section 1
## Section 2
Content of section 2
### Subsection 2.1
Content of subsection 2.1
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

for i, chunk in enumerate(chunks):
    print(f'Chunk {i+1}:\n{chunk}')