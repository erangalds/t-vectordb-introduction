from langchain.text_splitter import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
  <html lang='en'>
  <head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Fancy Example HTML Page</title>
  </head>
  <body>
    <h1>Main Title</h1>
    <p>This is an introductory paragraph with some basic content.</p>
    
    <h2>Section 1: Introduction</h2>
    <p>This section introduces the topic. Below is a list:</p>
    <ul>
      <li>First item</li>
      <li>Second item</li>
      <li>Third item with <strong>bold text</strong> and <a href='#'>a link</a></li>
    </ul>
    
    <h3>Subsection 1.1: Details</h3>
    <p>This subsection provides additional details. Here's a table:</p>
    <table border='1'>
      <thead>
        <tr>
          <th>Header 1</th>
          <th>Header 2</th>
          <th>Header 3</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Row 1, Cell 1</td>
          <td>Row 1, Cell 2</td>
          <td>Row 1, Cell 3</td>
        </tr>
        <tr>
          <td>Row 2, Cell 1</td>
          <td>Row 2, Cell 2</td>
          <td>Row 2, Cell 3</td>
        </tr>
      </tbody>
    </table>
    
    <h2>Section 2: Media Content</h2>
    <p>This section contains an image and a video:</p>
      <img src='example_image_link.mp4' alt='Example Image'>
      <video controls width='250' src='example_video_link.mp4' type='video/mp4'>
      Your browser does not support the video tag.
    </video>

    <h2>Section 3: Code Example</h2>
    <p>This section contains a code block:</p>
    <pre><code data-lang="html">
    &lt;div&gt;
      &lt;p&gt;This is a paragraph inside a div.&lt;/p&gt;
    &lt;/div&gt;
    </code></pre>

    <h2>Conclusion</h2>
    <p>This is the conclusion of the document.</p>
  </body>
  </html>
"""

print(f'\n\n###################################################################')
print(f'\nUsingn HTMLHeaderTextSplitter\n')
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits


print(f'\n\n###################################################################')
print(f'\nUsingn HTMLSplitter\n')
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits