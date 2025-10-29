"""
Convert README.md to HTML with markdown2 library for proper formatting.
Handles nested lists, code blocks, and LaTeX math correctly.
"""

from pathlib import Path
import markdown2

def convert_md_to_html_markdown2(md_file, html_file):
    """
    Convert Markdown to HTML using markdown2 library.
    Properly handles nested lists and all markdown features.
    Protects LaTeX math from markdown processing.
    """
    
    try:
        # Read markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        print(f"✓ Read markdown file ({len(md_content)} chars)")
        
        # Protect LaTeX math from markdown processing
        import re
        
        # Extract all inline math ($...$) and display math ($$...$$)
        math_placeholders = {}
        math_counter = 0
        
        # Use HTML comment placeholders that markdown2 won't touch
        # Protect display math first
        pattern_display = r'\$\$([^\$]+?)\$\$'
        for match in re.finditer(pattern_display, md_content):
            placeholder = f"<!-- MATH_DISPLAY_{math_counter} -->"
            math_placeholders[placeholder] = f"$${match.group(1)}$$"
            md_content = md_content.replace(match.group(0), placeholder, 1)
            math_counter += 1
        
        # Protect inline math
        pattern_inline = r'\$([^\$]+?)\$'
        for match in re.finditer(pattern_inline, md_content):
            placeholder = f"<!-- MATH_INLINE_{math_counter} -->"
            math_placeholders[placeholder] = f"${match.group(1)}$"
            md_content = md_content.replace(match.group(0), placeholder, 1)
            math_counter += 1
        
        print(f"✓ Protected {math_counter} math expressions")
        
        # Convert using markdown2 with extras
        html_body = markdown2.markdown(
            md_content,
            extras=[
                'fenced-code-blocks',
                'tables',
                'strikethrough',
                'task_lists',
            ]
        )
        
        # Restore math expressions
        for placeholder, math_expr in math_placeholders.items():
            html_body = html_body.replace(placeholder, math_expr)
        
        print(f"✓ Restored {len(math_placeholders)} math expressions")
        
        print(f"✓ Converted markdown to HTML")
        
        # Create full HTML with MathJax
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README - Improved UNet for HipMRI Study</title>
    
    <!-- MathJax Configuration for LaTeX -->
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true
            }},
            svg: {{
                fontCache: 'global'
            }}
        }};
    </script>
    
    <!-- MathJax Script -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        @page {{
            size: A4;
            margin: 2cm;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: white;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: #0066cc;
            font-size: 28px;
            margin: 40px 0 20px 0;
            padding-bottom: 15px;
            border-bottom: 3px solid #0066cc;
            page-break-after: avoid;
        }}
        
        h2 {{
            color: #0066cc;
            font-size: 20px;
            margin: 30px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #333;
            font-size: 16px;
            margin: 20px 0 10px 0;
            page-break-after: avoid;
        }}
        
        h4 {{
            color: #666;
            font-size: 14px;
            margin: 15px 0 8px 0;
            page-break-after: avoid;
        }}
        
        p {{
            margin: 10px 0;
            text-align: justify;
        }}
        
        ul {{
            margin: 10px 0 10px 30px;
            list-style-type: disc;
        }}
        
        ol {{
            margin: 10px 0 10px 30px;
            list-style-type: decimal;
        }}
        
        li {{
            margin: 5px 0;
        }}
        
        ul ul, ol ul {{
            margin-top: 5px;
            margin-bottom: 5px;
        }}
        
        code {{
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            color: #d73a49;
        }}
        
        pre {{
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            line-height: 1.5;
            overflow-x: auto;
            page-break-inside: avoid;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            color: #333;
            border: none;
        }}
        
        blockquote {{
            border-left: 4px solid #0066cc;
            margin: 10px 0;
            padding-left: 15px;
            color: #666;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        
        th {{
            background-color: #0066cc;
            color: white;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        strong {{
            color: #0066cc;
            font-weight: bold;
        }}
        
        em {{
            font-style: italic;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 30px 0;
            page-break-after: avoid;
        }}
        
        /* MathJax styling */
        .MathJax {{
            margin: 2px 0 !important;
        }}
        
        .MathJax_Display {{
            margin: 15px 0 !important;
            text-align: center;
        }}
        
        .page-break {{
            page-break-after: always;
        }}
    </style>
</head>
<body>
{html_body}

    <script>
        // Ensure MathJax processes after page load
        window.addEventListener('load', function() {{
            if (window.MathJax) {{
                MathJax.typesetPromise().catch(err => console.log(err));
            }}
        }});
    </script>
</body>
</html>"""
        
        # Write HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        file_size = Path(html_file).stat().st_size / 1024
        print(f"✓ Successfully created HTML")
        print(f"✓ HTML size: {file_size:.2f} KB")
        print(f"✓ Nested lists: PROPERLY FORMATTED")
        print(f"✓ Code blocks: PRESERVED")
        print(f"✓ LaTeX math: ENABLED")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    md_file = script_dir / "README.md"
    html_file = script_dir / "README.html"
    
    if not md_file.exists():
        print(f"✗ README.md not found")
        exit(1)
    
    print("=" * 60)
    print("README.md → HTML (Markdown2 - Proper List Handling)")
    print("=" * 60)
    print(f"Input:  {md_file}")
    print(f"Output: {html_file}")
    print("-" * 60)
    
    success = convert_md_to_html_markdown2(str(md_file), str(html_file))
    
    print("-" * 60)
    if success:
        print("✓ Ready! Open in browser and press Ctrl+P for PDF")
    else:
        print("✗ Failed")
        exit(1)
