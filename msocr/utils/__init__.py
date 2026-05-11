"""Utility functions for manuscript OCR."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def validate_image_file(image_path: Path) -> bool:
    """Validate if a file is a valid image."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    if image_path.suffix.lower() not in valid_extensions:
        return False
    
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_sample_alto_xml(image_path: Path, text_content: str = "Sample text") -> str:
    """Create a sample ALTO XML file for testing."""
    image_path = Path(image_path)
    
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        width, height = 1000, 800  # Default size
    
    alto_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns="http://www.loc.gov/standards/alto/ns-v4#"
    xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-0.xsd">
    <Description>
        <sourceImageInformation>
            <fileName>{image_path.name}</fileName>
        </sourceImageInformation>
        <MeasurementUnit>pixel</MeasurementUnit>
    </Description>
    <Layout>
        <Page ID="page_1" PHYSICAL_IMG_NR="1" HEIGHT="{height}" WIDTH="{width}">
            <PrintSpace HEIGHT="{height}" WIDTH="{width}" VPOS="0" HPOS="0">
                <TextBlock ID="block_1" HPOS="50" VPOS="50" WIDTH="{width-100}" HEIGHT="{height-100}">
                    <TextLine ID="line_1" HPOS="50" VPOS="50" WIDTH="{width-100}" HEIGHT="50" 
                              BASELINE="50 70 {width-50} 70">
                        <String ID="string_1" CONTENT="{text_content}"/>
                    </TextLine>
                </TextBlock>
            </PrintSpace>
        </Page>
    </Layout>
</alto>"""
    
    return alto_xml


def create_sample_page_xml(image_path: Path, text_content: str = "Sample text") -> str:
    """Create a sample PAGE XML file for testing."""
    image_path = Path(image_path)
    
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        width, height = 1000, 800  # Default size
    
    page_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15" 
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
       xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd">
    <Metadata>
        <Creator>msocr</Creator>
        <Created>2024-01-01T00:00:00</Created>
    </Metadata>
    <Page imageFilename="{image_path.name}" imageWidth="{width}" imageHeight="{height}">
        <TextRegion id="region_1" custom="readingOrder {{index:0;}}">
            <Coords points="50,50 {width-50},50 {width-50},{height-50} 50,{height-50}"/>
            <TextLine id="line_1">
                <Baseline points="50,70 {width-50},70"/>
                <Coords points="50,50 {width-50},50 {width-50},90 50,90"/>
                <TextEquiv>
                    <Unicode>{text_content}</Unicode>
                </TextEquiv>
            </TextLine>
        </TextRegion>
    </Page>
</PcGts>"""
    
    return page_xml


def validate_xml_file(xml_path: Path) -> Dict[str, Any]:
    """Validate ALTO or PAGE XML file and return metadata."""
    xml_path = Path(xml_path)
    
    if not xml_path.exists():
        return {"valid": False, "error": "File does not exist"}
    
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Check if ALTO
        if root.tag.endswith('alto'):
            return validate_alto_xml(root)
        # Check if PAGE XML
        elif root.tag.endswith('PcGts'):
            return validate_page_xml(root)
        else:
            return {"valid": False, "error": "Unknown XML format"}
            
    except Exception as e:
        return {"valid": False, "error": str(e)}


def validate_alto_xml(root) -> Dict[str, Any]:
    """Validate ALTO XML structure."""
    try:
        # Check for required elements
        description = root.find('.//{http://www.loc.gov/standards/alto/ns-v4#}Description')
        layout = root.find('.//{http://www.loc.gov/standards/alto/ns-v4#}Layout')
        
        if description is None or layout is None:
            return {"valid": False, "error": "Missing required ALTO elements"}
        
        # Count text lines
        text_lines = root.findall('.//{http://www.loc.gov/standards/alto/ns-v4#}TextLine')
        
        return {
            "valid": True,
            "format": "ALTO",
            "text_lines": len(text_lines),
            "has_baselines": any(line.get('BASELINE') for line in text_lines)
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}


def validate_page_xml(root) -> Dict[str, Any]:
    """Validate PAGE XML structure."""
    try:
        # Check for required elements
        page = root.find('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}Page')
        
        if page is None:
            return {"valid": False, "error": "Missing Page element"}
        
        # Count text lines
        text_lines = root.findall('.//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}TextLine')
        
        return {
            "valid": True,
            "format": "PAGE",
            "text_lines": len(text_lines),
            "has_baselines": any(line.find('.//Baseline') is not None for line in text_lines)
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}


def get_supported_languages() -> Dict[str, Dict[str, Any]]:
    """Get supported languages and their configurations."""
    return {
        "sogdian": {
            "name": "Sogdian",
            "direction": "LTR",
            "script": "Sogdian",
            "config_file": "configs/sogdian_config.yaml",
            "common_characters": "ʾβγδζθκλμνξοπρστυφχψω𐼰𐼱𐼲𐼳𐼴𐼵𐼶𐼷𐼸𐼹𐼺𐼻𐼼𐼽𐼾𐼿"
        },
        "old_turkish": {
            "name": "Old Turkish",
            "direction": "RTL", 
            "script": "Old Turkic",
            "config_file": "configs/old_turkish_config.yaml",
            "common_characters": "𐰀𐰁𐰂𐰃𐰄𐰅𐰆𐰇𐰈𐰉𐰊𐰋𐰌𐰍𐰎𐰏𐰐𐰑𐰒𐰓𐰔𐰕𐰖𐰗𐰘𐰙𐰚𐰛𐰜𐰝𐰞𐰟𐰠𐰡𐰢𐰣𐰤𐰥𐰦𐰧𐰨𐰩𐰪𐰫𐰬𐰭𐰮𐰯𐰰𐰱𐰲𐰳𐰴𐰵𐰶𐰷𐰸𐰹𐰺𐰻𐰼𐰽𐰾𐰿𐱀𐱁𐱂𐱃𐱄𐱅𐱆𐱇𐱈𐱉𐱊𐱋𐱌𐱍𐱎𐱏𐱐𐱑𐱒𐱓𐱔𐱕𐱖𐱗𐱘𐱙𐱚𐱛𐱜𐱝𐱞𐱟𐱠𐱡𐱢𐱣𐱤𐱥𐱦𐱧𐱨𐱩𐱪𐱫𐱬𐱭𐱮𐱯𐱰𐱱𐱲𐱳𐱴𐱵𐱶𐱷𐱸𐱹𐱺𐱻"
        }
    }


def create_project_directories(base_path: str) -> Dict[str, Path]:
    """Create standard project directories."""
    base = Path(base_path)
    directories = {
        'data': base / 'data',
        'models': base / 'models', 
        'logs': base / 'logs',
        'notebooks': base / 'notebooks',
        'configs': base / 'configs',
        'exports': base / 'exports'
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def save_training_metadata(output_path: Path, metadata: Dict[str, Any]):
    """Save training metadata to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)