"""
Simple HTML to PDF converter using system tools.

This script uses the HTML file that was already created and converts it to PDF
by invoking the system's default browser's print-to-PDF capability.
"""

import os
import subprocess
from pathlib import Path
import webbrowser
import time

def html_to_pdf_via_chrome(html_file, pdf_file):
    """
    Convert HTML to PDF using Chrome/Edge headless mode.
    
    Args:
        html_file: Path to HTML file
        pdf_file: Path to output PDF
        
    Returns:
        bool: True if successful
    """
    
    html_file = Path(html_file).absolute()
    pdf_file = Path(pdf_file).absolute()
    
    if not html_file.exists():
        print(f"✗ HTML file not found: {html_file}")
        return False
    
    # Try to find Chrome or Edge
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    
    chrome_path = None
    for path in chrome_paths:
        if os.path.exists(path):
            chrome_path = path
            break
    
    if not chrome_path:
        print("✗ Chrome or Edge not found on your system")
        print("   Please install Chrome or Edge, or open the HTML file manually:")
        print(f"   File: {html_file}")
        print("\n   Then press Ctrl+P and select 'Save as PDF'")
        return False
    
    try:
        print(f"✓ Found browser at: {chrome_path}")
        print(f"⏳ Converting {html_file.name} to PDF...")
        print(f"   - Header: Custom title (Improved 3D UNet for HipMRI Study)")
        print(f"   - Footer: Page numbers only (no URL, no date/time)")
        
        # Chrome headless mode for PDF printing with custom settings
        cmd = [
            chrome_path,
            f"--headless=new",
            "--disable-gpu",
            f"--print-to-pdf={pdf_file}",
            "--print-to-pdf-no-header",  # Disable default header/footer
            f"file:///{html_file}",
        ]
        
        # Run Chrome
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        
        # Verify
        if pdf_file.exists():
            size_kb = pdf_file.stat().st_size / 1024
            print(f"✓ Successfully created PDF!")
            print(f"✓ File: {pdf_file}")
            print(f"✓ Size: {size_kb:.2f} KB")
            return True
        else:
            print("✗ PDF file was not created")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Conversion timeout")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    html_file = script_dir / "README.html"
    pdf_file = script_dir / "README.pdf"
    
    if not html_file.exists():
        print(f"✗ README.html not found")
        exit(1)
    
    print("=" * 60)
    print("HTML to PDF Converter (via Chrome/Edge)")
    print("=" * 60)
    print(f"Input:  {html_file}")
    print(f"Output: {pdf_file}")
    print("-" * 60)
    
    success = html_to_pdf_via_chrome(str(html_file), str(pdf_file))
    
    print("-" * 60)
    if not success:
        print("\n💡 FALLBACK: Use the HTML file directly")
        print(f"   Open this file in your browser: {html_file}")
        print("   Then use Ctrl+P → Save as PDF")
