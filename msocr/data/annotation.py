"""Annotation tool integration for ground truth creation."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AnnotationExporter:
    """Export dataset for external annotation tools and import results."""
    
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
    
    def export_for_labelstudio(self, output_dir: Path, language: Optional[str] = None):
        """
        Export images in Label Studio compatible format.
        
        Creates a JSON configuration file that can be imported into Label Studio.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get filtered images
        images = self.dataset_manager.list_images(language=language)
        
        # Create Label Studio configuration
        config = {
            "project_name": f"Manuscript OCR - {language or 'All'}",
            "interface": {
                "task": "image_segmentation",
                "labels": ["text_line"]
            }
        }
        
        # Save config
        config_path = output_dir / "labelstudio_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Create tasks for Label Studio
        tasks = []
        for image_data in images:
            image_path = self.dataset_manager.get_image_path(image_data["id"])
            if image_path and image_path.exists():
                task = {
                    "data": {
                        "image": f"file://{image_path.absolute()}",
                        "image_id": image_data["id"],
                        "language": image_data["language"],
                        "manuscript_id": image_data.get("manuscript_id", "")
                    }
                }
                tasks.append(task)
        
        # Save tasks
        tasks_path = output_dir / "labelstudio_tasks.json"
        with open(tasks_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2)
        
        logger.info(f"Exported {len(tasks)} images for Label Studio annotation")
        return config_path, tasks_path
    
    def export_for_cvat(self, output_dir: Path, language: Optional[str] = None):
        """
        Export images in CVAT compatible format.
        
        Creates CVAT XML annotation file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = self.dataset_manager.list_images(language=language)
        
        # Create CVAT XML structure
        root = ET.Element("annotations")
        
        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "id").text = "1"
        ET.SubElement(task, "name").text = f"Manuscript OCR - {language or 'All'}"
        ET.SubElement(task, "size").text = str(len(images))
        ET.SubElement(task, "mode").text = "annotation"
        
        labels = ET.SubElement(meta, "labels")
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = "text_line"
        
        # Add image information
        images_element = ET.SubElement(root, "images")
        
        for idx, image_data in enumerate(images):
            image_path = self.dataset_manager.get_image_path(image_data["id"])
            if not image_path or not image_path.exists():
                continue
            
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
                
                image_elem = ET.SubElement(images_element, "image")
                image_elem.set("id", str(idx + 1))
                image_elem.set("name", image_path.name)
                image_elem.set("width", str(width))
                image_elem.set("height", str(height))
                
            except Exception as e:
                logger.warning(f"Could not get dimensions for {image_path}: {e}")
        
        # Save CVAT XML
        cvat_path = output_dir / "cvat_annotation.xml"
        tree = ET.ElementTree(root)
        tree.write(cvat_path, encoding='utf-8', xml_declaration=True)
        
        logger.info(f"Exported CVAT annotation file with {len(images)} images")
        return cvat_path
    
    def import_labelstudio_results(self, results_file: Path):
        """Import annotation results from Label Studio."""
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        imported_count = 0
        
        for result in results:
            try:
                image_id = result.get("data", {}).get("image_id")
                annotations = result.get("annotations", [])
                
                if not image_id:
                    continue
                
                # Create ALTO XML from annotations
                if annotations:
                    alto_xml = self._create_alto_from_labelstudio(image_id, annotations)
                    if alto_xml:
                        imported_count += 1
                        
                        # Update dataset manager
                        self.dataset_manager.update_image_status(
                            image_id, "annotated", annotated=True
                        )
                
            except Exception as e:
                logger.error(f"Error importing annotation for {image_id}: {e}")
        
        logger.info(f"Imported {imported_count} annotations from Label Studio")
    
    def _create_alto_from_labelstudio(self, image_id: str, annotations: List[Dict]) -> Optional[Path]:
        """Create ALTO XML from Label Studio annotations."""
        image_data = self.dataset_manager.metadata["images"].get(image_id)
        if not image_data:
            return None
        
        image_path = self.dataset_manager.get_image_path(image_id)
        if not image_path or not image_path.exists():
            return None
        
        try:
            # Get image dimensions
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
            
            # Create ALTO XML structure
            alto = ET.Element("alto", xmlns="http://www.loc.gov/standards/alto/ns-v4#")
            
            description = ET.SubElement(alto, "Description")
            measurement_unit = ET.SubElement(description, "MeasurementUnit")
            measurement_unit.text = "pixel"
            
            source_image_info = ET.SubElement(description, "sourceImageInformation")
            file_name = ET.SubElement(source_image_info, "fileName")
            file_name.text = image_path.name
            
            layout = ET.SubElement(alto, "Layout")
            page = ET.SubElement(layout, "Page")
            page.set("ID", "page_1")
            page.set("PHYSICAL_IMG_NR", "1")
            page.set("HEIGHT", str(height))
            page.set("WIDTH", str(width))
            
            print_space = ET.SubElement(page, "PrintSpace")
            print_space.set("HEIGHT", str(height))
            print_space.set("WIDTH", str(width))
            print_space.set("VPOS", "0")
            print_space.set("HPOS", "0")
            
            # Process annotations
            for idx, annotation in enumerate(annotations):
                if annotation.get("type") == "rectanglelabels":
                    result = annotation.get("result", {})
                    if not result:
                        continue
                    
                    # Extract bounding box
                    x = result.get("x", 0)
                    y = result.get("y", 0)
                    w = result.get("width", 0)
                    h = result.get("height", 0)
                    
                    transcription = result.get(" transcription", {}).get("value", "")
                    
                    # Create TextBlock and TextLine
                    textblock = ET.SubElement(print_space, "TextBlock")
                    textblock.set("ID", f"block_{idx}")
                    textblock.set("HPOS", str(x))
                    textblock.set("VPOS", str(y))
                    textblock.set("WIDTH", str(w))
                    textblock.set("HEIGHT", str(h))
                    
                    textline = ET.SubElement(textblock, "TextLine")
                    textline.set("ID", f"line_{idx}")
                    textline.set("HPOS", str(x))
                    textline.set("VPOS", str(y))
                    textline.set("WIDTH", str(w))
                    textline.set("HEIGHT", str(h))
                    
                    if transcription:
                        string_elem = ET.SubElement(textline, "String")
                        string_elem.set("ID", f"string_{idx}")
                        string_elem.set("CONTENT", transcription)
            
            # Save ALTO XML
            annotations_dir = self.dataset_manager.annotations_dir
            alto_path = annotations_dir / f"{image_id}.xml"
            
            tree = ET.ElementTree(alto)
            tree.write(alto_path, encoding='utf-8', xml_declaration=True)
            
            return alto_path
            
        except Exception as e:
            logger.error(f"Error creating ALTO XML for {image_id}: {e}")
            return None
    
    def create_annotation_template(self, image_id: str, output_format: str = "alto") -> Optional[Path]:
        """Create an empty annotation template for manual editing."""
        image_data = self.dataset_manager.metadata["images"].get(image_id)
        if not image_data:
            return None
        
        image_path = self.dataset_manager.get_image_path(image_id)
        if not image_path or not image_path.exists():
            return None
        
        if output_format.lower() == "alto":
            return self._create_alto_template(image_id, image_path)
        elif output_format.lower() == "page":
            return self._create_page_xml_template(image_id, image_path)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
    
    def _create_alto_template(self, image_id: str, image_path: Path) -> Path:
        """Create empty ALTO XML template."""
        from PIL import Image
        
        with Image.open(image_path) as img:
            width, height = img.size
        
        alto = ET.Element("alto", xmlns="http://www.loc.gov/standards/alto/ns-v4#")
        
        description = ET.SubElement(alto, "Description")
        measurement_unit = ET.SubElement(description, "MeasurementUnit")
        measurement_unit.text = "pixel"
        
        source_image_info = ET.SubElement(description, "sourceImageInformation")
        file_name = ET.SubElement(source_image_info, "fileName")
        file_name.text = image_path.name
        
        layout = ET.SubElement(alto, "Layout")
        page = ET.SubElement(layout, "Page")
        page.set("ID", "page_1")
        page.set("PHYSICAL_IMG_NR", "1")
        page.set("HEIGHT", str(height))
        page.set("WIDTH", str(width))
        
        print_space = ET.SubElement(page, "PrintSpace")
        print_space.set("HEIGHT", str(height))
        print_space.set("WIDTH", str(width))
        print_space.set("VPOS", "0")
        print_space.set("HPOS", "0")
        
        # Save template
        template_path = self.dataset_manager.annotations_dir / f"{image_id}_template.xml"
        tree = ET.ElementTree(alto)
        tree.write(template_path, encoding='utf-8', xml_declaration=True)
        
        return template_path