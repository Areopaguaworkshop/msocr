"""Training wrapper for Kraken ketos commands."""

import subprocess
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KetosTrainer:
    """Wrapper for Kraken ketos training commands."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.output_config = config.get('output', {})
        self.logging_config = config.get('logging', {})
    
    def compile_dataset(self, xml_files: List[Path], output_path: Path) -> bool:
        """Compile XML files into a binary dataset."""
        try:
            cmd = [
                "ketos", "compile", "-f", "xml", 
                "-o", str(output_path)
            ] + [str(path) for path in xml_files]
            
            logger.info(f"Compiling dataset with {len(xml_files)} files")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Dataset compilation successful: {result.stdout}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset compilation failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def train_model(self, dataset_path: Path, output_prefix: Optional[str] = None) -> bool:
        """Train a recognition model."""
        try:
            model_spec = self.model_config.get('spec')
            if not model_spec:
                raise ValueError("Model specification not found in config")
            
            output_prefix = output_prefix or self.output_config.get('model_prefix', 'model')
            
            # Build command
            cmd = [
                "ketos", "train",
                "-f", "binary",
                "-s", model_spec,
                "-o", output_prefix
            ]
            
            # Add model configuration
            if 'normalization' in self.model_config:
                cmd.extend(["--normalization", self.model_config['normalization']])
            
            if 'base_dir' in self.model_config:
                cmd.extend(["--base-dir", self.model_config['base_dir']])
            
            # Add training configuration
            cmd.extend([
                "--optimizer", self.training_config.get('optimizer', 'Adam'),
                "--lrate", str(self.training_config.get('learning_rate', 0.001)),
                "--weight-decay", str(self.training_config.get('weight_decay', 0.0001))
            ])
            
            if 'schedule' in self.training_config:
                cmd.extend(["--schedule", self.training_config['schedule']])
            
            cmd.extend([
                "--epochs", str(self.training_config.get('epochs', 100)),
                "--min-epochs", str(self.training_config.get('min_epochs', 20)),
                "--lag", str(self.training_config.get('lag', 10))
            ])
            
            if 'validation_split' in self.training_config:
                cmd.extend(["--partition", str(self.training_config['validation_split'])])
            
            # Add augmentation if enabled
            if self.training_config.get('augment', False):
                cmd.append("--augment")
            
            # Add device and precision settings
            cmd.extend([
                "--device", self.training_config.get('device', 'auto'),
                "--precision", self.training_config.get('precision', 'auto'),
                "--workers", str(self.training_config.get('workers', 4))
            ])
            
            # Add dataset
            cmd.append(str(dataset_path))
            
            logger.info(f"Starting model training")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True)
            logger.info("Training completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def test_model(self, model_path: Path, dataset_path: Path) -> Dict[str, Any]:
        """Test a trained model."""
        try:
            cmd = [
                "ketos", "test",
                "-m", str(model_path),
                "-f", "binary",
                str(dataset_path)
            ]
            
            logger.info(f"Testing model {model_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                "success": True,
                "output": result.stdout,
                "model_path": str(model_path)
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Model testing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": e.stderr if e.stderr else "",
                "model_path": str(model_path)
            }
    
    def preprocess_for_training(self, xml_files: List[Path]) -> Optional[Path]:
        """Preprocess data files if needed before training."""
        # This could implement additional preprocessing steps
        # For now, just return the first file path as the dataset path
        if not xml_files:
            logger.error("No XML files provided for preprocessing")
            return None
        
        return xml_files[0]
    
    def validate_config(self) -> List[str]:
        """Validate the training configuration."""
        errors = []
        
        # Check required fields
        if not self.model_config.get('spec'):
            errors.append("Model specification not found")
        
        if not self.dataset_config.get('format_type'):
            errors.append("Dataset format type not specified")
        
        if not self.training_config.get('epochs'):
            errors.append("Training epochs not specified")
        
        return errors
    
    def get_training_command(self, dataset_path: Path, output_prefix: Optional[str] = None) -> List[str]:
        """Get the full training command without executing it."""
        model_spec = self.model_config.get('spec')
        if not model_spec:
            raise ValueError("Model specification not found in config")
        
        output_prefix = output_prefix or self.output_config.get('model_prefix', 'model')
        
        # Build command (same as in train_model)
        cmd = [
            "ketos", "train",
            "-f", "binary",
            "-s", model_spec,
            "-o", output_prefix
        ]
        
        # Add model configuration
        if 'normalization' in self.model_config:
            cmd.extend(["--normalization", self.model_config['normalization']])
        
        if 'base_dir' in self.model_config:
            cmd.extend(["--base-dir", self.model_config['base_dir']])
        
        # Add training configuration
        cmd.extend([
            "--optimizer", self.training_config.get('optimizer', 'Adam'),
            "--lrate", str(self.training_config.get('learning_rate', 0.001)),
            "--weight-decay", str(self.training_config.get('weight_decay', 0.0001))
        ])
        
        if 'schedule' in self.training_config:
            cmd.extend(["--schedule", self.training_config['schedule']])
        
        cmd.extend([
            "--epochs", str(self.training_config.get('epochs', 100)),
            "--min-epochs", str(self.training_config.get('min_epochs', 20)),
            "--lag", str(self.training_config.get('lag', 10))
        ])
        
        if 'validation_split' in self.training_config:
            cmd.extend(["--partition", str(self.training_config['validation_split'])])
        
        # Add augmentation if enabled
        if self.training_config.get('augment', False):
            cmd.append("--augment")
        
        # Add device and precision settings
        cmd.extend([
            "--device", self.training_config.get('device', 'auto'),
            "--precision", self.training_config.get('precision', 'auto'),
            "--workers", str(self.training_config.get('workers', 4))
        ])
        
        # Add dataset
        cmd.append(str(dataset_path))
        
        return cmd
    
    def train(self, xml_files: Optional[List[Path]] = None) -> bool:
        """Complete training workflow."""
        # Validate configuration
        errors = self.validate_config()
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        if not xml_files:
            logger.error("No XML files provided for training")
            return False
        
        # Compile dataset
        dataset_path = Path(f"{self.output_config.get('model_prefix', 'dataset')}.arrow")
        if not self.compile_dataset(xml_files, dataset_path):
            return False
        
        # Train model
        return self.train_model(dataset_path)