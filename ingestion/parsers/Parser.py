"""
Document parser for RAG pipeline.
Extracts text from PDF, DOCX, PPTX, TXT, HTML, and CSV files into standardized dictionaries.
"""

from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
import pdfplumber
from docx import Document
from pptx import Presentation
import pandas as pd
from bs4 import BeautifulSoup


def parse(file_path: str) -> List[Dict]:
    """
    Parse a document into a list of page/section dictionaries.
    
    Args:
        file_path: Absolute or relative path to the document
        
    Returns:
        List of dicts with keys: text, page, section_heading, doc_type, source
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file extension is not supported
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = path.suffix.lower()
    
    # Dispatch to appropriate parser
    if suffix == ".pdf":
        return _parse_pdf(str(path))
    elif suffix == ".docx":
        return _parse_docx(str(path))
    elif suffix in [".pptx", ".ppt"]:
        return _parse_pptx(str(path))
    elif suffix == ".txt":
        return _parse_txt(str(path))
    elif suffix == ".md":
        return _parse_md(str(path))
    elif suffix in [".html", ".htm"]:
        return _parse_html(str(path))
    elif suffix == ".csv":
        return _parse_csv(str(path))
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")


def _parse_pdf(file_path: str) -> List[Dict]:
    """
    Parse PDF using PyMuPDF with pdfplumber fallback.
    Detects section headings from bold/larger font text.
    """
    results = []
    source = str(file_path)
    
    # Try PyMuPDF first
    doc = fitz.open(file_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text with detailed structure for heading detection
        text_dict = page.get_text("dict")
        page_text = page.get_text()
        
        # If PyMuPDF yields empty text, fall back to pdfplumber
        if not page_text.strip():
            with pdfplumber.open(file_path) as pdf:
                plumber_page = pdf.pages[page_num]
                page_text = plumber_page.extract_text() or ""
            
            results.append({
                "text": page_text,
                "page": page_num + 1,
                "section_heading": "",
                "doc_type": "pdf",
                "source": source
            })
            continue
        
        # Detect section heading from PyMuPDF text blocks
        # Look for bold or larger-than-average font text at the start of the page
        section_heading = ""
        blocks = text_dict.get("blocks", [])
        
        for block in blocks[:3]:  # Check first 3 blocks for headings
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    font_size = span.get("size", 0)
                    font_flags = span.get("flags", 0)
                    
                    # flags & 16 indicates bold; larger font size suggests heading
                    is_bold = (font_flags & 16) != 0
                    is_large = font_size > 12
                    
                    if text and (is_bold or is_large) and len(text) < 200:
                        section_heading = text
                        break
                
                if section_heading:
                    break
            
            if section_heading:
                break
        
        results.append({
            "text": page_text,
            "page": page_num + 1,
            "section_heading": section_heading,
            "doc_type": "pdf",
            "source": source
        })
    
    doc.close()
    return results


def _parse_docx(file_path: str) -> List[Dict]:
    """
    Parse DOCX files, detecting code blocks and preserving tables.
    """
    results = []
    source = str(file_path)
    doc = Document(file_path)
    
    current_heading = ""
    
    for element in doc.element.body:
        # Handle paragraphs
        if element.tag.endswith("p"):
            para = None
            for p in doc.paragraphs:
                if p._element == element:
                    para = p
                    break
            
            if para:
                text = para.text.strip()
                if not text:
                    continue
                
                # Detect headings (styles starting with "Heading")
                if para.style.name.startswith("Heading"):
                    current_heading = text
                    continue
                
                # Detect code blocks (fenced with ```)
                if text.startswith("```") and text.endswith("```"):
                    text = f"[CODE_BLOCK]{text}[/CODE_BLOCK]"
                elif "```" in text:
                    text = f"[CODE_BLOCK]{text}[/CODE_BLOCK]"
                
                results.append({
                    "text": text,
                    "page": 0,  # DOCX doesn't have page numbers in API
                    "section_heading": current_heading,
                    "doc_type": "docx",
                    "source": source
                })
        
        # Handle tables
        elif element.tag.endswith("tbl"):
            table = None
            for t in doc.tables:
                if t._element == element:
                    table = t
                    break
            
            if table:
                # Serialize table as markdown-style rows
                table_text = []
                for row in table.rows:
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    table_text.append("| " + " | ".join(cells) + " |")
                
                if table_text:
                    results.append({
                        "text": "\n".join(table_text),
                        "page": 0,
                        "section_heading": current_heading,
                        "doc_type": "docx",
                        "source": source
                    })
    
    return results


def _parse_pptx(file_path: str) -> List[Dict]:
    """
    Parse PowerPoint files, one dict entry per slide.
    """
    results = []
    source = str(file_path)
    prs = Presentation(file_path)
    
    for slide_num, slide in enumerate(prs.slides, start=1):
        # Extract title as section heading
        section_heading = ""
        if slide.shapes.title:
            section_heading = slide.shapes.title.text.strip()
        
        # Extract all text from slide
        text_parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = shape.text.strip()
                if text and text != section_heading:  # Avoid duplicating title
                    text_parts.append(text)
        
        slide_text = "\n".join(text_parts)
        
        results.append({
            "text": slide_text,
            "page": slide_num,
            "section_heading": section_heading,
            "doc_type": "pptx",
            "source": source
        })
    
    return results


def _parse_txt(file_path: str) -> List[Dict]:
    """
    Parse plain text files.
    """
    source = str(file_path)
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(file_path, "r", encoding="latin-1", errors="replace") as f:
            text = f.read()
    
    return [{
        "text": text,
        "page": 0,
        "section_heading": "",
        "doc_type": "txt",
        "source": source
    }]

def _parse_md(file_path: str) -> List[Dict]:
    """
    Parse Markdown files.
    """
    source = str(file_path)
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1", errors="replace") as f:
            text = f.read()
    
    return [{
        "text": text,
        "page": 0,
        "section_heading": "",
        "doc_type": "md",
        "source": source
    }]


def _parse_html(file_path: str) -> List[Dict]:
    """
    Parse HTML files, stripping nav/footer/script/style tags.
    Detects code blocks in <code> and <pre> tags.
    """
    results = []
    source = str(file_path)
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove unwanted tags
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()
    
    # Process remaining elements
    for element in soup.find_all(["p", "div", "section", "article", "pre", "code", "table"]):
        # Find nearest heading ancestor
        section_heading = ""
        parent = element.find_parent(["h1", "h2", "h3", "h4", "h5", "h6"])
        if parent:
            section_heading = parent.get_text(strip=True)
        else:
            # Look for preceding heading sibling
            for sibling in element.find_all_previous(["h1", "h2", "h3", "h4", "h5", "h6"]):
                section_heading = sibling.get_text(strip=True)
                break
        
        # Handle tables
        if element.name == "table":
            table_text = []
            for row in element.find_all("tr"):
                cells = [cell.get_text(strip=True).replace("\n", " ") for cell in row.find_all(["td", "th"])]
                if cells:
                    table_text.append("| " + " | ".join(cells) + " |")
            
            if table_text:
                results.append({
                    "text": "\n".join(table_text),
                    "page": 0,
                    "section_heading": section_heading,
                    "doc_type": "html",
                    "source": source
                })
            continue
        
        # Handle code blocks
        if element.name in ["pre", "code"]:
            text = element.get_text()
            if text.strip():
                results.append({
                    "text": f"[CODE_BLOCK]{text}[/CODE_BLOCK]",
                    "page": 0,
                    "section_heading": section_heading,
                    "doc_type": "html",
                    "source": source
                })
            continue
        
        # Handle regular text
        text = element.get_text(strip=True)
        if text and len(text) > 10:  # Skip very short snippets
            # Check if contains code-like content
            if element.find(["code", "pre"]):
                text = f"[CODE_BLOCK]{text}[/CODE_BLOCK]"
            
            results.append({
                "text": text,
                "page": 0,
                "section_heading": section_heading,
                "doc_type": "html",
                "source": source
            })
    
    return results


def _parse_csv(file_path: str) -> List[Dict]:
    """
    Parse CSV files, each row becomes one dict entry.
    """
    results = []
    source = str(file_path)
    
    df = pd.read_csv(file_path, encoding="utf-8", errors="replace")
    
    for idx, row in df.iterrows():
        # Join columns as "col: val, col: val"
        row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        
        results.append({
            "text": row_text,
            "page": 0,
            "section_heading": "",
            "doc_type": "csv",
            "source": source
        })
    
    return results
